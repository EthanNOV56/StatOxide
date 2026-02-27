//! Generalized Linear Model implementation using IRLS (Iteratively Reweighted Least Squares)

use ndarray::{Array1, Array2};
use so_linalg;
use statrs::distribution::{Normal, ContinuousCDF};

use so_core::data::DataFrame;
use so_core::formula::Formula;
use so_core::error::{Result, Error};

use super::family::{Family, Link, is_valid_link};
use super::results::GLMResults;

/// Add intercept column (column of ones) to design matrix
fn add_intercept(X: &Array2<f64>) -> Array2<f64> {
    let (n_samples, n_features) = X.dim();
    let mut X_with_intercept = Array2::zeros((n_samples, n_features + 1));
    
    // First column is intercept (ones)
    for i in 0..n_samples {
        X_with_intercept[(i, 0)] = 1.0;
    }
    
    // Copy original features
    for i in 0..n_samples {
        for j in 0..n_features {
            X_with_intercept[(i, j + 1)] = X[(i, j)];
        }
    }
    
    X_with_intercept
}

/// GLM model configuration and builder
#[derive(Debug, Clone)]
pub struct GLMModelBuilder {
    family: Family,
    link: Option<Link>,
    intercept: bool,
    max_iter: usize,
    tol: f64,
    scale: Option<f64>,
}

impl Default for GLMModelBuilder {
    fn default() -> Self {
        Self {
            family: Family::Gaussian,
            link: None,
            intercept: true,
            max_iter: 100,
            tol: 1e-6,
            scale: None,
        }
    }
}

impl GLMModelBuilder {
    /// Create a new GLM builder with default settings
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the distribution family
    pub fn family(mut self, family: Family) -> Self {
        self.family = family;
        self
    }
    
    /// Set the link function (if None, uses family's default)
    pub fn link(mut self, link: Link) -> Self {
        self.link = Some(link);
        self
    }
    
    /// Set whether to include an intercept
    pub fn intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self
    }
    
    /// Set maximum number of IRLS iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    
    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    
    /// Set fixed scale parameter (dispersion)
    pub fn scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }
    
    /// Build the GLM model with current configuration
    pub fn build(self) -> GLM {
        let link = self.link.unwrap_or_else(|| self.family.default_link());
        
        if !is_valid_link(self.family, link) {
            panic!("Invalid link-function combination: {} with {}", 
                self.family.name(), link.name());
        }
        
        GLM {
            family: self.family,
            link,
            intercept: self.intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            scale: self.scale,
        }
    }
}

/// Generalized Linear Model
#[derive(Debug, Clone)]
pub struct GLM {
    family: Family,
    link: Link,
    intercept: bool,
    max_iter: usize,
    tol: f64,
    scale: Option<f64>,
}

impl GLM {
    /// Create a new GLM builder
    pub fn new() -> GLMModelBuilder {
        GLMModelBuilder::new()
    }
    
    /// Fit the GLM using formula and data
    pub fn fit(&self, formula: &str, data: &DataFrame) -> Result<GLMResults> {
        let formula = Formula::parse(formula)
            .map_err(|e| Error::FormulaError(format!("Formula parse error: {}", e)))?;
        
        // Extract response and predictors
        let response_var = formula.response.as_ref()
            .and_then(|term| match term {
                so_core::formula::Term::Variable(name) => Some(name.clone()),
                _ => None,
            })
            .ok_or_else(|| Error::DataError(
                "Response must be a simple variable".to_string()
            ))?;
        
        let y_series = data.column(&response_var)
            .ok_or_else(|| Error::DataError(
                format!("Response variable '{}' not found in data", response_var)
            ))?;
        
        let y = y_series.data().to_owned();
        
        // Validate response values for the family
        self.family.validate_response(&y)?;
        
        // Build design matrix
        let X = formula.build_matrix(data)
            .map_err(|e| Error::DataError(format!("Design matrix error: {}", e)))?;
        
        // Add intercept if requested
        let X = if self.intercept {
            add_intercept(&X)
        } else {
            X.clone()
        };
        
        // Fit using IRLS
        self.fit_irls(&X, &y)
    }
    
    /// Fit using IRLS algorithm
    fn fit_irls(&self, X: &Array2<f64>, y: &Array1<f64>) -> Result<GLMResults> {
        let n = X.nrows();
        let p = X.ncols();
        
        if n <= p {
            return Err(Error::DataError(
                format!("Not enough observations (n={}) for p={} parameters", n, p)
            ));
        }
        
        // Initialize parameters
        let mut mu = self.family.initialize(y);
        let mut eta = mu.mapv(|mu_val| self.link.link(mu_val));
        let mut beta = Array1::zeros(p);
        
        // Working variables for IRLS
        let mut weights = Array1::zeros(n);
        let mut working_response = Array1::zeros(n);
        let mut converged = false;
        let mut iteration = 0;
        let mut deviance_old = f64::INFINITY;
        let mut last_X_weighted = Array2::zeros((n, p));
        let mut last_XtWX = Array2::zeros((p, p));
        
        // IRLS iterations
        while iteration < self.max_iter {
            // Compute working weights and response
            for i in 0..n {
                let mu_i = mu[i];
                let eta_i = eta[i];
                
                // Weight: w = 1 / (V(μ) * (g'(μ))^2)
                let variance = self.family.variance(mu_i);
                let derivative = self.link.derivative(eta_i);
                weights[i] = 1.0 / (variance * derivative.powi(2)).max(1e-10);
                
                // Working response: z = η + (y - μ) * g'(μ)
                working_response[i] = eta_i + (y[i] - mu_i) * derivative;
            }
            
            // Solve weighted least squares: β = (X'WX)^-1 X'Wz
            let X_weighted = X.clone() * weights.mapv(|w| w.sqrt()).insert_axis(ndarray::Axis(1));
            let z_weighted = &working_response * weights.mapv(|w| w.sqrt());
            
            let XtWX = X_weighted.t().dot(&X_weighted);
            let XtWz = X_weighted.t().dot(&z_weighted);
            
            let beta_new = so_linalg::solve(&XtWX, &XtWz)
                .map_err(|e| Error::LinearAlgebraError(format!("IRLS solve failed: {}", e)))?;
            
            // Save the last weighted matrices for standard error computation
            last_X_weighted = X_weighted;
            last_XtWX = XtWX;
            
            // Update parameters
            eta = X.dot(&beta_new);
            mu = eta.mapv(|eta_val| self.link.inverse_link(eta_val));
            
            // Check convergence
            let deviance = self.family.deviance(y, &mu);
            let beta_diff = (&beta_new - &beta).mapv(|x| x.abs()).sum();
            
            if (deviance_old - deviance).abs() < self.tol && beta_diff < self.tol {
                converged = true;
                break;
            }
            
            beta = beta_new;
            deviance_old = deviance;
            iteration += 1;
        }
        
        if !converged {
            return Err(Error::Message(
                format!("IRLS did not converge after {} iterations", self.max_iter)
            ));
        }
        
        // Compute final predictions and residuals
        let fitted = mu.clone();
        let residuals = y - &fitted;
        
        // Compute Pearson residuals
        let pearson_residuals: Array1<f64> = y.iter()
            .zip(fitted.iter())
            .map(|(&y_val, &mu_val)| {
                let variance = self.family.variance(mu_val);
                if variance > 0.0 {
                    (y_val - mu_val) / variance.sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        
        // Compute leverage and Cook's distance (simplified)
        let hat_matrix_diag = self.compute_leverage(&last_X_weighted);
        
        // Estimate or use provided scale
        let scale = match self.scale {
            Some(s) => s,
            None => self.family.estimate_dispersion(y, &fitted, n, p),
        };
        
        // Compute standard errors
        let cov_matrix = self.compute_covariance(&last_XtWX, scale);
        let std_errors: Array1<f64> = (0..p)
            .map(|i| cov_matrix[(i, i)].sqrt())
            .collect();
        
        // Compute z/t-values and p-values
        let (z_values, p_values) = self.compute_inference(&beta, &std_errors, n - p);
        
        // Compute model statistics
        let null_deviance = self.compute_null_deviance(y);
        let residual_deviance = deviance_old;
        let df_null = if self.intercept { n - 1 } else { n };
        let df_residual = n - p;
        let aic = self.compute_aic(y, &fitted, p);
        let bic = self.compute_bic(y, &fitted, p, n);
        
        Ok(GLMResults {
            coefficients: beta,
            std_errors,
            z_values,
            p_values,
            fitted_values: fitted,
            residuals,
            pearson_residuals,
            hat_matrix_diag,
            scale,
            deviance: residual_deviance,
            null_deviance,
            df_residual,
            df_null,
            aic,
            bic,
            converged,
            iterations: iteration,
            family: self.family,
            link: self.link,
            intercept: self.intercept,
            n_obs: n,
            n_params: p,
        })
    }
    
    /// Compute leverage (diagonal of hat matrix)
    fn compute_leverage(&self, X_weighted: &Array2<f64>) -> Array1<f64> {
        let n = X_weighted.nrows();
        let p = X_weighted.ncols();
        
        if n <= p {
            return Array1::zeros(n);
        }
        
        let xtx = X_weighted.t().dot(X_weighted);
        match so_linalg::inv(&xtx) {
            Ok(xtx_inv) => {
                let mut leverage = Array1::zeros(n);
                for i in 0..n {
                    let xi = X_weighted.row(i);
                    leverage[i] = xi.dot(&xtx_inv.dot(&xi.t()));
                }
                leverage
            },
            Err(_) => Array1::zeros(n),
        }
    }
    
    /// Compute covariance matrix of coefficients
    fn compute_covariance(&self, XtWX: &Array2<f64>, scale: f64) -> Array2<f64> {
        match so_linalg::inv(XtWX) {
            Ok(cov) => &cov * scale,
            Err(_) => Array2::zeros((XtWX.nrows(), XtWX.ncols())),
        }
    }
    
    /// Compute z/t-values and p-values for coefficients
    fn compute_inference(&self, coefficients: &Array1<f64>, std_errors: &Array1<f64>, _df_residual: usize) -> (Array1<f64>, Array1<f64>) {
        let n_coef = coefficients.len();
        let mut z_values = Array1::zeros(n_coef);
        let mut p_values = Array1::zeros(n_coef);
        
        for i in 0..n_coef {
            let se = std_errors[i];
            if se > 0.0 {
                z_values[i] = coefficients[i] / se;
                
                // Use normal distribution for p-values in GLM
                let z_abs = z_values[i].abs();
                p_values[i] = 2.0 * (1.0 - Normal::new(0.0, 1.0).unwrap().cdf(z_abs));
            } else {
                z_values[i] = f64::NAN;
                p_values[i] = f64::NAN;
            }
        }
        
        (z_values, p_values)
    }
    
    /// Compute null deviance (intercept-only model)
    fn compute_null_deviance(&self, y: &Array1<f64>) -> f64 {
        let n = y.len();
        let mu_null = if self.intercept {
            // Compute mean response (on link scale, then transform)
            let y_mean = y.mean().unwrap_or(0.0);
            let eta_mean = self.link.link(y_mean);
            Array1::from_elem(n, self.link.inverse_link(eta_mean))
        } else {
            Array1::zeros(n)
        };
        
        self.family.deviance(y, &mu_null)
    }
    
    /// Compute Akaike Information Criterion
    fn compute_aic(&self, y: &Array1<f64>, fitted: &Array1<f64>, n_params: usize) -> f64 {
        let deviance = self.family.deviance(y, fitted);
        2.0 * n_params as f64 + deviance
    }
    
    /// Compute Bayesian Information Criterion
    fn compute_bic(&self, y: &Array1<f64>, fitted: &Array1<f64>, n_params: usize, n_obs: usize) -> f64 {
        let deviance = self.family.deviance(y, fitted);
        n_params as f64 * (n_obs as f64).ln() + deviance
    }
    
    /// Predict on new data
    pub fn predict(&self, results: &GLMResults, X: &Array2<f64>) -> Array1<f64> {
        let X_with_intercept = if self.intercept {
            add_intercept(X)
        } else {
            X.clone()
        };
        
        let linear_predictor = X_with_intercept.dot(&results.coefficients);
        linear_predictor.mapv(|eta| self.link.inverse_link(eta))
    }
}