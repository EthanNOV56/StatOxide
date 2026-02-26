//! Ordinary Least Squares (OLS) linear regression
//!
//! This module implements the standard linear regression model
//! using the ordinary least squares estimator.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, QR, Solve};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

use crate::base::{
    Coefficient, ModelError, ModelResultTrait, ModelStatistics, ModelSummary, ModelType,
    ResidualStatistics, Result,
};
use crate::lm::{LinearConfig, LinearModel, StandardErrorType};
use so_core::data::DataFrame;
use so_core::formula::Formula;

// ==================== Type Definitions ====================

/// Matrix type alias for 2D arrays
pub type Matrix = Array2<f64>;

/// Vector type alias for 1D arrays
pub type Vector = Array1<f64>;

// ==================== Linear Regression Result ====================

/// OLS linear regression result
#[derive(Debug, Clone)]
pub struct LinearRegressionResult {
    /// Coefficients
    pub coefficients: Vector,
    /// Standard errors
    pub standard_errors: Vector,
    /// t-statistics
    pub t_statistics: Vector,
    /// p-values
    pub p_values: Vector,
    /// Coefficient confidence intervals (lower)
    pub ci_lower: Vector,
    /// Coefficient confidence intervals (upper)
    pub ci_upper: Vector,
    /// Fitted values
    pub fitted_values: Vector,
    /// Residuals
    pub residuals: Vector,
    /// Hat matrix diagonal (leverage)
    pub hat_diagonal: Vector,
    /// Cook's distances
    pub cooks_distance: Vector,
    /// Design matrix
    pub x: Matrix,
    /// Response vector
    pub y: Vector,
    /// Variable names
    pub variable_names: Vec<String>,
    /// Model statistics
    pub model_statistics: ModelStatistics,
    /// Has intercept
    pub has_intercept: bool,
}

impl LinearRegressionResult {
    /// Create coefficient structs from arrays
    pub fn to_coefficients(&self) -> Vec<Coefficient> {
        self.coefficients
            .iter()
            .zip(self.standard_errors.iter())
            .zip(self.t_statistics.iter())
            .zip(self.p_values.iter())
            .zip(self.ci_lower.iter())
            .zip(self.ci_upper.iter())
            .enumerate()
            .map(|(i, (((((&coef, &se), &t), &p), &ci_low), &ci_up))| {
                let name = if i < self.variable_names.len() {
                    self.variable_names[i].clone()
                } else {
                    format!("x{}", i)
                };

                let is_intercept = i == 0 && self.has_intercept;

                Coefficient::new(name, coef)
                    .with_std_error(se)
                    .with_t_stat(t)
                    .with_p_value(p)
                    .with_ci(ci_low, ci_up)
            })
            .collect()
    }
}

impl ModelResultTrait for LinearRegressionResult {
    fn coefficients(&self) -> &[Coefficient] {
        // Note: This requires storing coefficients as Coefficient structs
        // or converting on demand. We'll handle this in the LinearRegression struct.
        &[]
    }

    fn fitted_values(&self) -> &Vector {
        &self.fitted_values
    }

    fn residuals(&self) -> &Vector {
        &self.residuals
    }

    fn r_squared(&self) -> Option<f64> {
        self.model_statistics.r_squared
    }

    fn adj_r_squared(&self) -> Option<f64> {
        self.model_statistics.adj_r_squared
    }

    fn aic(&self) -> Option<f64> {
        self.model_statistics.aic
    }

    fn bic(&self) -> Option<f64> {
        self.model_statistics.bic
    }

    fn log_likelihood(&self) -> Option<f64> {
        self.model_statistics.log_likelihood
    }

    fn residual_std_error(&self) -> Option<f64> {
        self.model_statistics.residual_std_error
    }

    fn df_residual(&self) -> Option<usize> {
        self.model_statistics.df_residual
    }

    fn df_model(&self) -> Option<usize> {
        self.model_statistics.df_model
    }
}

// ==================== Linear Regression Model ====================

/// OLS linear regression model
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// Model formula
    formula: Formula,
    /// Data
    data: Option<DataFrame>,
    /// Configuration
    config: LinearConfig,
    /// Fitted result
    result: Option<LinearRegressionResult>,
    /// Whether to include intercept
    has_intercept: bool,
}

impl LinearRegression {
    /// Create a new linear regression model
    pub fn new(formula: &str) -> Result<Self> {
        let formula = Formula::parse(formula).map_err(ModelError::Formula)?;

        Ok(Self {
            formula,
            data: None,
            config: LinearConfig::default(),
            result: None,
            has_intercept: true,
        })
    }

    /// Set data for the model
    pub fn data(mut self, data: &DataFrame) -> Self {
        self.data = Some(data.clone());
        self
    }

    /// Set configuration
    pub fn config(mut self, config: LinearConfig) -> Self {
        self.config = config;
        self
    }

    /// Disable intercept
    pub fn no_intercept(mut self) -> Self {
        self.config.intercept = false;
        self.has_intercept = false;
        self
    }

    /// Use robust standard errors
    pub fn robust(mut self, se_type: StandardErrorType) -> Self {
        self.config.se_type = se_type;
        self
    }

    /// Fit the OLS model
    pub fn fit(mut self) -> Result<Self> {
        let data = self.data.as_ref().ok_or_else(|| ModelError::Custom {
            message: "No data provided".to_string(),
        })?;

        // Get design matrix and response
        let (x, y) = self
            .formula
            .design_matrix(data)
            .map_err(ModelError::Formula)?;

        let y = y.ok_or_else(|| ModelError::Custom {
            message: "No response variable in formula".to_string(),
        })?;

        // Check dimensions
        let n = x.nrows();
        let p = x.ncols();

        if n <= p {
            return Err(ModelError::InsufficientData {
                n_samples: n,
                n_predictors: p,
            });
        }

        // Set intercept flag
        self.has_intercept = self.formula.has_intercept;

        // Extract variable names
        let variable_names = self.extract_variable_names(&x, data);

        // Center and scale if requested
        let (x_scaled, _) = if self.config.center_predictors || self.config.scale_predictors {
            self.scale_matrix(&x)?
        } else {
            (x.clone(), None)
        };

        // Fit using QR decomposition (numerically stable)
        let coefficients = self.svd_solve(&x_scaled, &y)?;

        // Calculate fitted values and residuals
        let fitted_values = x_scaled.dot(&coefficients);
        let residuals = &y - &fitted_values;

        // Calculate basic statistics
        let rss = residuals.mapv(|r| r * r).sum();
        let tss = y
            .iter()
            .map(|&yi| (yi - y.mean().unwrap_or(0.0)).powi(2))
            .sum::<f64>();
        let r_squared = 1.0 - rss / tss;
        let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n as f64 - 1.0) / (n as f64 - p as f64));
        let residual_std_error = (rss / (n as f64 - p as f64)).sqrt();

        // Calculate standard errors
        let standard_errors = match self.config.se_type {
            StandardErrorType::Standard => {
                self.standard_errors(&x_scaled, &residuals, n, p, rss)?
            }
            _ => self.robust_standard_errors(&x_scaled, &residuals, n, p)?,
        };

        // Calculate inference statistics
        let (t_statistics, p_values, ci_lower, ci_upper) =
            self.calculate_inference(&coefficients, &standard_errors, n, p)?;

        // Calculate F-statistic
        let (f_statistic, f_p_value) = self.f_statistic(rss, tss, n, p)?;

        // Calculate log-likelihood
        let log_likelihood = self.log_likelihood(&residuals, n, p, rss)?;

        // Calculate information criteria
        let aic = 2.0 * p as f64 - 2.0 * log_likelihood;
        let bic = (n as f64).ln() * p as f64 - 2.0 * log_likelihood;

        // Calculate leverage (hat values)
        let hat_diagonal = self.hat_matrix_diagonal(&x_scaled)?;

        // Calculate Cook's distances
        let cooks_distance =
            self.cooks_distance(&residuals, &hat_diagonal, p, residual_std_error)?;

        // Create ModelStatistics
        let model_statistics = ModelStatistics {
            r_squared: Some(r_squared),
            adj_r_squared: Some(adj_r_squared),
            residual_std_error: Some(residual_std_error),
            f_statistic: Some(f_statistic),
            f_p_value: Some(f_p_value),
            log_likelihood: Some(log_likelihood),
            aic: Some(aic),
            bic: Some(bic),
            df_residual: Some(n - p),
            df_model: Some(p - 1),
            iterations: None,
            converged: Some(true),
            weights: None,
            null_deviance: None,
            residual_deviance: None,
            chi_squared: None,
            chi_squared_p_value: None,
            df_null: None,
        };

        // Store result
        self.result = Some(LinearRegressionResult {
            coefficients,
            standard_errors,
            t_statistics,
            p_values,
            ci_lower,
            ci_upper,
            fitted_values,
            residuals,
            hat_diagonal,
            cooks_distance,
            x: x_scaled,
            y,
            variable_names,
            model_statistics,
            has_intercept: self.has_intercept,
        });

        Ok(self)
    }

    // /// Solve using QR decomposition
    // fn qr_solve(&self, x: &Matrix, y: &Vector) -> Result<Vector> {
    //     let qr = x.qr().map_err(|e| ModelError::NumericalError {
    //         message: format!("QR decomposition failed: {}", e),
    //         operation: "qr_solve".to_string(),
    //     })?;

    //     qr.solve(y).map_err(|e| ModelError::NumericalError {
    //         message: format!("Failed to solve linear system: {}", e),
    //         operation: "qr_solve".to_string(),
    //     })
    // }

    /// Solve using SVD-based least squares (recommended for numerical stability)
    fn svd_solve(&self, x: &Matrix, y: &Vector) -> Result<Vector> {
        use ndarray_linalg::LeastSquaresSvd;

        x.least_squares(y)
            .map_err(|e| ModelError::NumericalError {
                message: format!("SVD least squares failed: {}", e),
                operation: "svd_solve".to_string(),
            })
            .map(|ls| ls.solution)
    }

    /// Calculate standard errors
    fn standard_errors(
        &self,
        x: &Matrix,
        residuals: &Vector,
        n: usize,
        p: usize,
        rss: f64,
    ) -> Result<Vector> {
        // (X'X)^{-1}
        let xtx = x.t().dot(x);
        let xtx_inv = xtx.inv().map_err(|e| ModelError::NumericalError {
            message: format!("Failed to invert X'X: {}", e),
            operation: "standard_errors".to_string(),
        })?;

        // σ² = RSS / (n - p)
        let sigma2 = rss / (n as f64 - p as f64);

        // Standard errors = sqrt(diag(σ² * (X'X)^{-1}))
        let cov_matrix = &xtx_inv * sigma2;
        let std_errors = cov_matrix.diag().mapv(|v| v.sqrt().max(1e-10));

        Ok(std_errors)
    }

    /// Calculate robust standard errors
    fn robust_standard_errors(
        &self,
        x: &Matrix,
        residuals: &Vector,
        n: usize,
        p: usize,
    ) -> Result<Vector> {
        // Implementation of White's heteroscedasticity-consistent standard errors
        // For simplicity, implementing HC0
        let xtx = x.t().dot(x);
        let xtx_inv = xtx.inv().map_err(|e| ModelError::NumericalError {
            message: format!("Failed to invert X'X: {}", e),
            operation: "robust_standard_errors".to_string(),
        })?;

        // Create diagonal matrix of squared residuals
        let mut omega = Matrix::zeros((n, n));
        for i in 0..n {
            omega[(i, i)] = residuals[i].powi(2);
        }

        // (X'X)^{-1} X' Ω X (X'X)^{-1}
        let middle = x.t().dot(&omega).dot(x);
        let cov_matrix = xtx_inv.dot(&middle).dot(&xtx_inv);

        let std_errors = cov_matrix.diag().mapv(|v| v.sqrt().max(1e-10));
        Ok(std_errors)
    }

    /// Calculate inference statistics
    fn calculate_inference(
        &self,
        coefficients: &Vector,
        std_errors: &Vector,
        n: usize,
        p: usize,
    ) -> Result<(Vector, Vector, Vector, Vector)> {
        let df = n - p;

        // Calculate t-statistics
        let t_statistics: Vector = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(&coef, &se)| coef / se)
            .collect();

        // Calculate p-values using t-distribution
        let t_dist =
            StudentsT::new(0.0, 1.0, df as f64).map_err(|e| ModelError::NumericalError {
                message: format!("Failed to create t-distribution: {}", e),
                operation: "calculate_inference".to_string(),
            })?;

        let p_values: Vector = t_statistics
            .iter()
            .map(|&t| {
                let p = 2.0 * (1.0 - t_dist.cdf(t.abs()));
                p.min(1.0).max(0.0)
            })
            .collect();

        // Calculate confidence intervals
        let alpha = 1.0 - self.config.confidence_level;
        let t_critical = t_dist.inverse_cdf(1.0 - alpha / 2.0);

        let ci_lower: Vector = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(&coef, &se)| coef - t_critical * se)
            .collect();

        let ci_upper: Vector = coefficients
            .iter()
            .zip(std_errors.iter())
            .map(|(&coef, &se)| coef + t_critical * se)
            .collect();

        Ok((t_statistics, p_values, ci_lower, ci_upper))
    }

    /// Calculate F-statistic
    fn f_statistic(&self, rss: f64, tss: f64, n: usize, p: usize) -> Result<(f64, f64)> {
        let ess = tss - rss; // Explained sum of squares
        let df_model = (p - 1) as f64;
        let df_residual = (n - p) as f64;

        let f_statistic = (ess / df_model) / (rss / df_residual);

        // F-distribution p-value
        let f_dist =
            FisherSnedecor::new(df_model, df_residual).map_err(|e| ModelError::NumericalError {
                message: format!("Failed to create F-distribution: {}", e),
                operation: "f_statistic".to_string(),
            })?;

        let f_p_value = 1.0 - f_dist.cdf(f_statistic);

        Ok((f_statistic, f_p_value))
    }

    /// Calculate log-likelihood
    fn log_likelihood(&self, residuals: &Vector, n: usize, p: usize, rss: f64) -> Result<f64> {
        let sigma2 = rss / n as f64;

        // Log-likelihood for normal distribution
        let ll = -0.5 * n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln() - 0.5 * rss / sigma2;

        Ok(ll)
    }

    /// Calculate hat matrix diagonal (leverage)
    fn hat_matrix_diagonal(&self, x: &Matrix) -> Result<Vector> {
        let xtx = x.t().dot(x);
        let xtx_inv = xtx.inv().map_err(|e| ModelError::NumericalError {
            message: format!("Failed to invert X'X: {}", e),
            operation: "hat_matrix_diagonal".to_string(),
        })?;

        // H = X(X'X)^{-1}X'
        // Diagonal elements: h_ii = x_i'(X'X)^{-1}x_i
        let mut hat_diag = Vector::zeros(x.nrows());

        for i in 0..x.nrows() {
            let xi = x.row(i);
            let hi = xi.dot(&xtx_inv).dot(&xi.t());
            hat_diag[i] = hi;
        }

        Ok(hat_diag)
    }

    /// Calculate Cook's distances
    fn cooks_distance(
        &self,
        residuals: &Vector,
        hat_diag: &Vector,
        p: usize,
        sigma: f64,
    ) -> Result<Vector> {
        let mut cooks = Vector::zeros(residuals.len());

        for i in 0..residuals.len() {
            let r = residuals[i];
            let h = hat_diag[i];
            cooks[i] = (r.powi(2) * h) / (p as f64 * sigma.powi(2) * (1.0 - h).powi(2));
        }

        Ok(cooks)
    }

    /// Scale matrix (center and/or scale)
    fn scale_matrix(&self, x: &Matrix) -> Result<(Matrix, Option<ScalingFactors>)> {
        if !self.config.center_predictors && !self.config.scale_predictors {
            return Ok((x.clone(), None));
        }

        let mut scaled = x.clone();
        let mut factors = ScalingFactors {
            means: Vec::new(),
            stds: Vec::new(),
        };

        for j in 0..x.ncols() {
            let col = x.column(j);

            if self.config.center_predictors {
                let mean = col.mean().unwrap_or(0.0);
                factors.means.push(mean);
                for i in 0..col.len() {
                    scaled[(i, j)] -= mean;
                }
            }

            if self.config.scale_predictors {
                let std = col.std(1.0);
                if std > 1e-10 {
                    factors.stds.push(std);
                    for i in 0..col.len() {
                        scaled[(i, j)] /= std;
                    }
                } else {
                    factors.stds.push(1.0);
                }
            }
        }

        Ok((scaled, Some(factors)))
    }

    /// Extract variable names
    fn extract_variable_names(&self, x: &Matrix, df: &DataFrame) -> Vec<String> {
        let mut names = Vec::new();

        if self.formula.has_intercept {
            names.push("(Intercept)".to_string());
        }

        // For each column in the design matrix, get the corresponding variable name
        // This is a simplification - in reality, we need to map columns to terms
        for term in &self.formula.terms {
            // This should be implemented based on how the design matrix is built
            names.push(term.to_string());
        }

        names
    }

    /// Get coefficients as Coefficient
    fn coefficients(&self) -> Option<&Vector> {
        self.result.as_ref().map(|r| &r.coefficients)
    }

    /// Get coefficients as Coefficient structs
    fn coefficient_structs(&self) -> Option<Vec<Coefficient>> {
        self.result.as_ref().map(|r| r.to_coefficients())
    }

    /// Get model summary
    pub fn summary(&self) -> Result<ModelSummary> {
        let result = self.result.as_ref().ok_or_else(|| ModelError::NotFitted)?;

        let coefficients = self
            .coefficient_structs()
            .ok_or_else(|| ModelError::NotFitted)?;

        // Calculate residual statistics
        let residual_stats = ResidualStatistics {
            min: *result
                .residuals
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0),
            q1: self.calculate_quantile(&result.residuals, 0.25),
            median: self.calculate_quantile(&result.residuals, 0.5),
            q3: self.calculate_quantile(&result.residuals, 0.75),
            max: *result
                .residuals
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0),
            mean: result.residuals.mean().unwrap_or(0.0),
            std_dev: result.residuals.std(1.0),
            skewness: Some(self.calculate_skewness(&result.residuals)),
            kurtosis: Some(self.calculate_kurtosis(&result.residuals)),
            durbin_watson: Some(self.calculate_durbin_watson(&result.residuals)),
        };

        Ok(ModelSummary {
            model_type: ModelType::LinearRegression,
            formula: self.formula.to_string(),
            n_obs: result.y.len(),
            n_predictors: result.coefficients.len(),
            coefficients,
            model_statistics: result.model_statistics,
            residual_statistics: residual_stats,
        })
    }

    /// Calculate quantile
    fn calculate_quantile(&self, data: &Vector, q: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = (sorted.len() as f64 - 1.0) * q;
        let lower = idx.floor() as usize;
        let upper = idx.ceil() as usize;

        if lower == upper {
            sorted[lower]
        } else {
            let weight = idx - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }

    /// Calculate skewness
    fn calculate_skewness(&self, data: &Vector) -> f64 {
        let n = data.len() as f64;
        if n < 3.0 {
            return 0.0;
        }

        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(1.0);

        if std < 1e-10 {
            return 0.0;
        }

        let sum_cubes: f64 = data.iter().map(|&x| (x - mean).powi(3)).sum();
        (sum_cubes / n) / std.powi(3)
    }

    /// Calculate kurtosis
    fn calculate_kurtosis(&self, data: &Vector) -> f64 {
        let n = data.len() as f64;
        if n < 4.0 {
            return 0.0;
        }

        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(1.0);

        if std < 1e-10 {
            return 0.0;
        }

        let sum_quarts: f64 = data.iter().map(|&x| (x - mean).powi(4)).sum();
        (sum_quarts / n) / std.powi(4) - 3.0
    }

    /// Calculate Durbin-Watson statistic
    fn calculate_durbin_watson(&self, residuals: &Vector) -> f64 {
        let n = residuals.len();
        if n < 2 {
            return 0.0;
        }

        let mut sum_sq_diff = 0.0;
        for i in 1..n {
            let diff = residuals[i] - residuals[i - 1];
            sum_sq_diff += diff * diff;
        }

        let sum_sq: f64 = residuals.iter().map(|&r| r * r).sum();

        if sum_sq < 1e-10 {
            0.0
        } else {
            sum_sq_diff / sum_sq
        }
    }
}

impl LinearModel for LinearRegression {
    fn fit(&mut self) -> Result<&mut Self> {
        let fitted = self.clone().fit()?;
        *self = fitted;
        Ok(self)
    }

    fn coefficients(&self) -> Option<&Vector> {
        self.result.as_ref().map(|r| &r.coefficients)
    }

    fn standard_errors(&self) -> Option<&Vector> {
        self.result.as_ref().map(|r| &r.standard_errors)
    }

    fn predict(&self, data: &DataFrame) -> Result<Vector> {
        let result = self.result.as_ref().ok_or_else(|| ModelError::NotFitted)?;

        // Get design matrix (without response)
        let (x, _) = self
            .formula
            .design_matrix(data)
            .map_err(ModelError::Formula)?;

        // Note: We should apply the same scaling as during training
        // For now, assume no scaling or that scaling factors are stored
        Ok(x.dot(&result.coefficients))
    }

    fn fitted_values(&self) -> Option<&Vector> {
        self.result.as_ref().map(|r| &r.fitted_values)
    }

    fn residuals(&self) -> Option<&Vector> {
        self.result.as_ref().map(|r| &r.residuals)
    }
}

/// Scaling factors for centering and scaling
#[derive(Debug, Clone)]
struct ScalingFactors {
    means: Vec<f64>,
    stds: Vec<f64>,
}
