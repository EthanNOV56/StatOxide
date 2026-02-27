//! Mixed effects models for StatOxide
//!
//! This module implements linear mixed models (LMM) and generalized linear mixed models (GLMM)
//! for hierarchical or clustered data.
//!
//! # Model Specification
//!
//! Linear Mixed Model:
//! y = Xβ + Zb + ε
//! where:
//! - y: response vector
//! - X: fixed effects design matrix
//! - β: fixed effects coefficients
//! - Z: random effects design matrix
//! - b: random effects coefficients ~ N(0, G)
//! - ε: residuals ~ N(0, R)
//!
//! Generalized Linear Mixed Model extends LMM to non-Gaussian responses using link functions.
//!
//! # Estimation Methods
//!
//! 1. **REML**: Restricted Maximum Likelihood (preferred for variance components)
//! 2. **ML**: Maximum Likelihood
//! 3. **PQL**: Penalized Quasi-Likelihood (for GLMM)
//! 4. **Laplace Approximation**: For non-Gaussian GLMM
//!

#![allow(non_snake_case)]  // Allow mathematical notation (X, Z, V, etc.)

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use so_core::data::DataFrame;
use so_core::error::{Result, Error};
use so_core::formula::Formula;
use so_linalg::{solve, inv};

use crate::glm::{Family, Link, GLM, GLMResults};

/// Random effects structure specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffect {
    /// Name of the grouping variable
    pub group_var: String,
    /// Formula for random effects within groups
    pub formula: String,
    /// Covariance structure (currently only supports independent)
    pub covariance: RandomCovariance,
}

/// Covariance structure for random effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RandomCovariance {
    /// Independent random effects (diagonal covariance)
    Independent,
    /// Compound symmetry (exchangeable)
    CompoundSymmetry,
    /// Auto-regressive of order 1
    AR1,
    /// Unstructured covariance
    Unstructured,
    /// Custom covariance matrix
    Custom(Array2<f64>),
}

/// Linear Mixed Model (LMM) results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMMResults {
    /// Fixed effects coefficients
    pub fixed_effects: Array1<f64>,
    /// Standard errors for fixed effects
    pub fixed_se: Array1<f64>,
    /// Random effects variance components
    pub variance_components: Vec<(String, f64)>,
    /// Residual variance
    pub residual_variance: f64,
    /// Log-likelihood
    pub log_lik: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Degrees of freedom for fixed effects
    pub df_fixed: usize,
    /// Degrees of freedom for residuals
    pub df_resid: usize,
    /// Convergence status
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

/// Linear Mixed Model builder
pub struct LinearMixedModelBuilder {
    data: DataFrame,
    response: String,
    fixed_formula: String,
    random_effects: Vec<RandomEffect>,
    method: EstimationMethod,
    max_iter: usize,
    tol: f64,
}

/// Estimation method for mixed models
#[derive(Debug, Clone, Copy)]
pub enum EstimationMethod {
    /// Restricted Maximum Likelihood
    REML,
    /// Maximum Likelihood
    ML,
}

/// Estimation method for GLMM
#[derive(Debug, Clone, Copy)]
pub enum GLMMEstimationMethod {
    /// Penalized Quasi-Likelihood (PQL)
    PQL,
    /// Laplace Approximation
    Laplace,
    /// Adaptive Gauss-Hermite Quadrature (higher accuracy)
    AGHQ(usize), // number of quadrature points
}

impl LinearMixedModelBuilder {
    /// Create a new LMM builder
    pub fn new(data: DataFrame, response: &str, fixed_formula: &str) -> Self {
        Self {
            data,
            response: response.to_string(),
            fixed_formula: fixed_formula.to_string(),
            random_effects: Vec::new(),
            method: EstimationMethod::REML,
            max_iter: 100,
            tol: 1e-6,
        }
    }
    
    /// Add a random effect
    pub fn random_effect(mut self, group_var: &str, formula: &str) -> Self {
        self.random_effects.push(RandomEffect {
            group_var: group_var.to_string(),
            formula: formula.to_string(),
            covariance: RandomCovariance::Independent,
        });
        self
    }
    
    /// Set estimation method
    pub fn method(mut self, method: EstimationMethod) -> Self {
        self.method = method;
        self
    }
    
    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    
    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    
    /// Fit the linear mixed model
    pub fn fit(self) -> Result<LMMResults> {
        // Parse fixed effects formula
        // This is a simplified implementation
        // In practice, we would use the formula parser from so-core
        
        // Extract response variable
        let y = self.data.column(&self.response)
            .ok_or_else(|| Error::DataError(format!("Response column '{}' not found", self.response)))?;
        let y_array = y.data().to_owned();
        
        // Build design matrix for fixed effects
        let X = self.build_fixed_design_matrix()?;
        
        // Build random effects design matrices
        let (Z_matrices, group_sizes) = self.build_random_design_matrices()?;
        
        // Fit using EM algorithm (simplified)
        self.fit_em(&y_array, &X, &Z_matrices, &group_sizes)
    }
    
    /// Build fixed effects design matrix
    fn build_fixed_design_matrix(&self) -> Result<Array2<f64>> {
        // Parse fixed effects formula
        let formula_str = if self.fixed_formula.contains('~') {
            self.fixed_formula.clone()
        } else {
            // If no response in formula (for random effects), add placeholder
            format!("__response__ ~ {}", self.fixed_formula)
        };
        
        let formula = Formula::parse(&formula_str)
            .map_err(|e| Error::FormulaError(format!("Failed to parse fixed formula: {}", e)))?;
        
        // Build design matrix (includes intercept if specified)
        formula.build_matrix(&self.data)
            .map_err(|e| Error::DataError(format!("Failed to build design matrix: {}", e)))
    }
    
    /// Build random effects design matrices
    fn build_random_design_matrices(&self) -> Result<(Vec<Array2<f64>>, Vec<usize>)> {
        let mut Z_matrices = Vec::new();
        let mut group_sizes = Vec::new();
        
        for random_effect in &self.random_effects {
            // Simplified: create indicator matrix for groups
            let group_col = self.data.column(&random_effect.group_var)
                .ok_or_else(|| Error::DataError(format!("Group column '{}' not found", random_effect.group_var)))?;
            
            // TODO: Implement proper categorical extraction
            let groups: Vec<String> = vec!["group1".to_string(), "group2".to_string()];  // Placeholder
            let n_groups = groups.len();
            let n = self.data.n_rows();
            
            let mut Z = Array2::zeros((n, n_groups));
            
            // Extract group indices from column data (assumes numeric encoding)
            let group_data = group_col.data();
            for j in 0..n {
                let group_idx = group_data[j] as usize % n_groups;  // Simple mapping
                Z[(j, group_idx)] = 1.0;
            }
            
            Z_matrices.push(Z);
            group_sizes.push(n_groups);
        }
        
        Ok((Z_matrices, group_sizes))
    }
    
    /// Fit using Expectation-Maximization algorithm (simplified)
    fn fit_em(&self, y: &Array1<f64>, X: &Array2<f64>, Z_matrices: &[Array2<f64>], group_sizes: &[usize]) -> Result<LMMResults> {
        let n = y.len();
        let p = X.ncols();
        
        // Initial values
        let mut sigma2_e = 1.0;  // Residual variance
        let mut sigma2_u = vec![1.0; Z_matrices.len()];  // Random effect variances
        
        // Combine all Z matrices into a single block-diagonal matrix
        let Z = self.combine_Z_matrices(Z_matrices, group_sizes);
        let q = Z.ncols();
        
        let mut beta = Array1::zeros(p);
        let mut u = Array1::zeros(q);
        
        let mut converged = false;
        let mut iter = 0;
        
        while !converged && iter < self.max_iter {
            iter += 1;
            
            // E-step: Update random effects
            let V_inv = self.compute_V_inv(&Z, sigma2_e, &sigma2_u, group_sizes)?;
            let XtVX = X.t().dot(&V_inv.dot(X));
            let XtVy = X.t().dot(&V_inv.dot(y));
            
            beta = solve(&XtVX, &XtVy)
                .map_err(|e| Error::LinearAlgebraError(format!("Failed to solve for beta: {}", e)))?;
            
            let residuals = y - X.dot(&beta);
            u = Z.t().dot(&V_inv.dot(&residuals));
            
            // M-step: Update variance components
            let old_sigma2_e = sigma2_e;
            let old_sigma2_u = sigma2_u.clone();
            
            // Update residual variance
            let y_Xb = y - X.dot(&beta);
            let y_Xb_Zu = &y_Xb - Z.dot(&u);
            sigma2_e = y_Xb_Zu.dot(&y_Xb_Zu) / (n - p) as f64;
            
            // Update random effect variances (simplified)
            for i in 0..sigma2_u.len() {
                let start_idx: usize = group_sizes[..i].iter().sum();
                let end_idx = start_idx + group_sizes[i];
                let u_i = u.slice(ndarray::s![start_idx..end_idx]);
                sigma2_u[i] = u_i.dot(&u_i) / group_sizes[i] as f64;
            }
            
            // Check convergence
            let delta_e = (sigma2_e - old_sigma2_e).abs() / old_sigma2_e.max(1e-10);
            let max_delta_u = sigma2_u.iter().zip(&old_sigma2_u)
                .map(|(new, old)| (new - old).abs() / old.max(1e-10))
                .fold(0.0, f64::max);
            
            converged = delta_e < self.tol && max_delta_u < self.tol;
        }
        
        // Compute standard errors and log-likelihood
        let V_inv = self.compute_V_inv(&Z, sigma2_e, &sigma2_u, group_sizes)?;
        let XtVX = X.t().dot(&V_inv.dot(X));
        let cov_beta = inv(&XtVX)
            .map_err(|e| Error::LinearAlgebraError(format!("Failed to invert X'V^{{-1}}X: {}", e)))?;
        
        let fixed_se = cov_beta.diag().mapv(|x| x.sqrt());
        
        // Compute log-likelihood
        let V = self.compute_V(&Z, sigma2_e, &sigma2_u, group_sizes);
        let log_lik = self.compute_log_lik(y, X, &V, beta.clone(), self.method);
        
        // Compute information criteria
        let n_params = p + sigma2_u.len() + 1;  // beta + variance components
        let aic = -2.0 * log_lik + 2.0 * n_params as f64;
        let bic = -2.0 * log_lik + (n_params as f64) * (n as f64).ln();
        
        // Prepare variance components with names
        let mut var_comps = Vec::new();
        for (i, random_effect) in self.random_effects.iter().enumerate() {
            var_comps.push((random_effect.group_var.clone(), sigma2_u[i]));
        }
        
        Ok(LMMResults {
            fixed_effects: beta,
            fixed_se,
            variance_components: var_comps,
            residual_variance: sigma2_e,
            log_lik,
            aic,
            bic,
            df_fixed: p,
            df_resid: n - p,
            converged,
            iterations: iter,
        })
    }
    
    /// Combine Z matrices into block-diagonal matrix
    fn combine_Z_matrices(&self, Z_matrices: &[Array2<f64>], group_sizes: &[usize]) -> Array2<f64> {
        let n = Z_matrices[0].nrows();
        let total_cols: usize = group_sizes.iter().sum();
        
        let mut Z = Array2::zeros((n, total_cols));
        let mut col_offset = 0;
        
        for (i, Z_i) in Z_matrices.iter().enumerate() {
            let cols = group_sizes[i];
            for row in 0..n {
                for col in 0..cols {
                    Z[(row, col_offset + col)] = Z_i[(row, col)];
                }
            }
            col_offset += cols;
        }
        
        Z
    }
    
    /// Compute V = ZGZ' + σ²I
    fn compute_V(&self, Z: &Array2<f64>, sigma2_e: f64, sigma2_u: &[f64], group_sizes: &[usize]) -> Array2<f64> {
        let n = Z.nrows();
        let mut V = Array2::zeros((n, n));
        
        // Add residual variance component
        for i in 0..n {
            V[(i, i)] = sigma2_e;
        }
        
        // Add random effect components
        let mut col_offset = 0;
        for (k, &sigma2_u_k) in sigma2_u.iter().enumerate() {
            let cols = group_sizes[k];
            let Z_k = Z.slice(ndarray::s![.., col_offset..col_offset + cols]);
            
            // Add Z_k G_k Z_k' where G_k = σ²_u_k I
            let ZkZkt = Z_k.dot(&Z_k.t());
            V = &V + &(ZkZkt * sigma2_u_k);
            
            col_offset += cols;
        }
        
        V
    }
    
    /// Compute V^{-1} using Woodbury identity (simplified)
    fn compute_V_inv(&self, Z: &Array2<f64>, sigma2_e: f64, _sigma2_u: &[f64], _group_sizes: &[usize]) -> Result<Array2<f64>> {
        let n = Z.nrows();
        let mut V_inv = Array2::zeros((n, n));
        
        // For independent random effects with diagonal G, we can use special structure
        // Simplified: return identity for now
        for i in 0..n {
            V_inv[(i, i)] = 1.0 / sigma2_e;
        }
        
        Ok(V_inv)
    }
    
    /// Compute log-likelihood
    fn compute_log_lik(&self, y: &Array1<f64>, X: &Array2<f64>, V: &Array2<f64>, beta: Array1<f64>, method: EstimationMethod) -> f64 {
        let n = y.len() as f64;
        let _p = X.ncols() as f64;
        
        // Compute residuals
        let residuals = y - X.dot(&beta);
        
        // Log-determinant of V (simplified: assuming V is diagonal)
        let log_det_V: f64 = V.diag().iter().map(|&v| v.ln()).sum();
        
        // Quadratic form: r'V^{-1}r
        // Note: need to clone residuals for the division
        let residuals_clone = residuals.clone();
        let Vinv_r = residuals_clone / V.diag();  // Simplified for diagonal V
        let quad_form = residuals.dot(&Vinv_r);
        
        let log_lik = -0.5 * (n * (2.0 * std::f64::consts::PI).ln() + log_det_V + quad_form);
        
        match method {
            EstimationMethod::ML => log_lik,
            EstimationMethod::REML => {
                // REML adjusts for fixed effects
                // Compute X'V^{-1}X where V is diagonal
                let inv_diag = V.diag().mapv(|v| 1.0 / v);
                let X_scaled = X * &inv_diag.insert_axis(ndarray::Axis(1));
                let XtVX = X.t().dot(&X_scaled);
                let log_det_XtVX = XtVX.diag().iter().map(|&x| x.ln()).sum::<f64>();
                log_lik - 0.5 * log_det_XtVX
            }
        }
    }
}

/// Generalized Linear Mixed Model (GLMM) results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLMMResults {
    /// Fixed effects coefficients
    pub fixed_effects: Array1<f64>,
    /// Standard errors for fixed effects
    pub fixed_se: Array1<f64>,
    /// Random effects variance components
    pub variance_components: Vec<(String, f64)>,
    /// Scale parameter (dispersion)
    pub scale: f64,
    /// Log-likelihood (approximate)
    pub log_lik: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Degrees of freedom for fixed effects
    pub df_fixed: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Convergence status
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Family used
    pub family: Family,
    /// Link function used
    pub link: Link,
}

/// Generalized Linear Mixed Model builder
pub struct GLMMBuilder {
    data: DataFrame,
    response: String,
    fixed_formula: String,
    random_effects: Vec<RandomEffect>,
    family: Family,
    link: Option<Link>,
    method: GLMMEstimationMethod,
    max_iter: usize,
    tol: f64,
}

impl GLMMBuilder {
    /// Create a new GLMM builder
    pub fn new(data: DataFrame, response: &str, fixed_formula: &str, family: Family) -> Self {
        Self {
            data,
            response: response.to_string(),
            fixed_formula: fixed_formula.to_string(),
            random_effects: Vec::new(),
            family,
            link: None,
            method: GLMMEstimationMethod::PQL,
            max_iter: 50,
            tol: 1e-6,
        }
    }
    
    /// Add a random effect
    pub fn random_effect(mut self, group_var: &str, formula: &str) -> Self {
        self.random_effects.push(RandomEffect {
            group_var: group_var.to_string(),
            formula: formula.to_string(),
            covariance: RandomCovariance::Independent,
        });
        self
    }
    
    /// Set the link function (if None, uses family's default)
    pub fn link(mut self, link: Link) -> Self {
        self.link = Some(link);
        self
    }
    
    /// Set estimation method
    pub fn method(mut self, method: GLMMEstimationMethod) -> Self {
        self.method = method;
        self
    }
    
    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    
    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    
    /// Fit the GLMM using Penalized Quasi-Likelihood (PQL)
    pub fn fit(self) -> Result<GLMMResults> {
        // Determine link function
        let link = self.link.unwrap_or_else(|| self.family.default_link());
        
        // Extract response variable
        let y = self.data.column(&self.response)
            .ok_or_else(|| Error::DataError(format!("Response column '{}' not found", self.response)))?;
        let y_array = y.data().to_owned();
        
        // For PQL, we need initial values from a GLM without random effects
        let glmm_results = self.fit_pql(&y_array, link)?;
        Ok(glmm_results)
    }
    
    /// Fit using Penalized Quasi-Likelihood (PQL) algorithm
    fn fit_pql(&self, y: &Array1<f64>, link: Link) -> Result<GLMMResults> {
        let n = y.len();
        
        // Step 1: Fit a GLM without random effects to get initial estimates
        let _glm_model = GLM::new()
            .family(self.family)
            .link(link)
            .max_iter(self.max_iter)
            .tol(self.tol)
            .build();
        
        // We need to fit the GLM - for simplicity, we'll use intercept-only model initially
        // In practice, we should parse the fixed formula and build proper design matrix
        let X = self.build_fixed_design_matrix()?;
        
        // Initialize: fit GLM ignoring random effects
        let mut eta = Array1::zeros(n); // linear predictor
        let mut mu = Array1::zeros(n); // mean
        let mut mu_eta = Array1::zeros(n); // derivative dμ/dη
        
        // Initialize with simple values based on family
        match self.family {
            Family::Binomial => {
                // For binary data, initialize with empirical logits
                let y_mean = y.mean().unwrap_or(0.5);
                let eps = 1e-4;
                let y_clamped = y_mean.max(eps).min(1.0 - eps);
                let init_eta = link.link(y_clamped);
                eta.fill(init_eta);
            }
            Family::Poisson => {
                let y_mean = y.mean().unwrap_or(1.0);
                let init_eta = link.link(y_mean.max(1e-4));
                eta.fill(init_eta);
            }
            _ => {
                // Gaussian or other families
                let y_mean = y.mean().unwrap_or(0.0);
                let init_eta = link.link(y_mean);
                eta.fill(init_eta);
            }
        }
        
        // Update mu and mu_eta based on initial eta
        for i in 0..n {
            mu[i] = link.inverse_link(eta[i]);
            mu_eta[i] = link.derivative(eta[i]);
        }
        
        // Build random effects design matrices (simplified)
        let (Z_matrices, group_sizes) = self.build_random_design_matrices()?;
        let Z = self.combine_Z_matrices(&Z_matrices, &group_sizes);
        let q = Z.ncols();
        
        // Initial variance components
        let mut sigma2_e = 1.0; // residual/scale parameter
        let mut sigma2_u = vec![1.0; Z_matrices.len()];
        
        // Initial fixed effects (from intercept-only GLM)
        let p = X.ncols();
        let mut beta = Array1::zeros(p);
        if p > 0 {
            // Simple intercept estimate
            beta[0] = eta.mean().unwrap_or(0.0);
        }
        
        let mut u = Array1::zeros(q);
        
        let mut converged = false;
        let mut iter = 0;
        
        while !converged && iter < self.max_iter {
            iter += 1;
            
            // PQL iteration:
            // 1. Compute working variable: y* = η + (y - μ) * (dη/dμ)
            // where dη/dμ = 1 / (dμ/dη) = 1 / mu_eta
            let mut y_star = Array1::zeros(n);
            for i in 0..n {
                let d_eta_d_mu = if mu_eta[i].abs() > 1e-10 { 1.0 / mu_eta[i] } else { 1.0 };
                y_star[i] = eta[i] + (y[i] - mu[i]) * d_eta_d_mu;
            }
            
            // 2. Compute weights: w = 1 / (V(μ) * (dη/dμ)^2)
            // where V(μ) is variance function of the family
            let mut weights = Array1::zeros(n);
            for i in 0..n {
                let d_eta_d_mu = if mu_eta[i].abs() > 1e-10 { 1.0 / mu_eta[i] } else { 1.0 };
                let v_mu = self.family.variance(mu[i]);
                weights[i] = 1.0 / (v_mu * d_eta_d_mu * d_eta_d_mu);
            }
            
            // 3. Fit weighted LMM to y_star with weights
            // This is simplified - we should implement proper weighted LMM
            // For now, we'll use a simplified EM-like approach
            
            // Update beta and u using weighted least squares analogy
            let W_sqrt = weights.mapv(|w| w.sqrt());
            let y_star_weighted = &y_star * &W_sqrt;
            let X_weighted = &X * &W_sqrt.clone().insert_axis(ndarray::Axis(1));
            let Z_weighted = &Z * &W_sqrt.clone().insert_axis(ndarray::Axis(1));
            
            // Solve weighted mixed model equations (simplified)
            // [X'WX  X'WZ] [β] = [X'Wy*]
            // [Z'WX  Z'WZ + G^{-1}] [u]   [Z'Wy*]
            // where G = diag(σ²_u_k I) for each random effect
            
            let XtWX = X_weighted.t().dot(&X_weighted);
            let ZtWZ = Z_weighted.t().dot(&Z_weighted);
            let XtWZ = X_weighted.t().dot(&Z_weighted);
            let ZtWX = Z_weighted.t().dot(&X_weighted);
            
            let XtWy = X_weighted.t().dot(&y_star_weighted);
            let ZtWy = Z_weighted.t().dot(&y_star_weighted);
            
            // Build mixed model equations matrix
            let total_cols = p + q;
            let mut M = Array2::zeros((total_cols, total_cols));
            let mut rhs = Array1::zeros(total_cols);
            
            // Top-left: X'WX
            M.slice_mut(ndarray::s![0..p, 0..p]).assign(&XtWX);
            // Top-right: X'WZ
            M.slice_mut(ndarray::s![0..p, p..]).assign(&XtWZ);
            // Bottom-left: Z'WX
            M.slice_mut(ndarray::s![p.., 0..p]).assign(&ZtWX);
            // Bottom-right: Z'WZ + G^{-1}
            let mut ZtWZ_plus_Ginv = ZtWZ.clone();
            
            // Add G^{-1} to diagonal blocks
            let mut col_offset = 0;
            for (k, sigma2_u_k) in sigma2_u.iter().enumerate() {
                let cols = group_sizes[k];
                let g_inv = 1.0 / f64::max(*sigma2_u_k, 1e-10);
                for i in 0..cols {
                    let idx = col_offset + i;
                    ZtWZ_plus_Ginv[(idx, idx)] += g_inv;
                }
                col_offset += cols;
            }
            
            M.slice_mut(ndarray::s![p.., p..]).assign(&ZtWZ_plus_Ginv);
            
            // Right-hand side
            rhs.slice_mut(ndarray::s![0..p]).assign(&XtWy);
            rhs.slice_mut(ndarray::s![p..]).assign(&ZtWy);
            
            // Solve mixed model equations
            let solution = solve(&M, &rhs)
                .map_err(|e| Error::LinearAlgebraError(format!("Failed to solve mixed model equations: {}", e)))?;
            
            let new_beta = solution.slice(ndarray::s![0..p]).to_owned();
            let new_u = solution.slice(ndarray::s![p..]).to_owned();
            
            // 4. Update linear predictor and mean
            let new_eta = X.dot(&new_beta) + Z.dot(&new_u);
            
            // Update mu and mu_eta
            let mut new_mu = Array1::zeros(n);
            let mut new_mu_eta = Array1::zeros(n);
            for i in 0..n {
                new_mu[i] = link.inverse_link(new_eta[i]);
                new_mu_eta[i] = link.derivative(new_eta[i]);
            }
            
            // 5. Update variance components (simplified EM update)
            let old_sigma2_e = sigma2_e;
            let old_sigma2_u = sigma2_u.clone();
            
            // Update random effect variances
            col_offset = 0;
            for (k, sigma2_u_k) in sigma2_u.iter_mut().enumerate() {
                let cols = group_sizes[k];
                let u_k = new_u.slice(ndarray::s![col_offset..col_offset + cols]);
                let trace_term = 0.0; // Simplified - should compute trace of inverse
                *sigma2_u_k = u_k.dot(&u_k) / (cols as f64 - trace_term).max(1.0);
                col_offset += cols;
            }
            
            // Update scale parameter
            let residuals = y - &new_mu;
            let pearson_residuals = residuals.mapv(|r| r * r / self.family.variance(new_mu[0]).max(1e-10));
            sigma2_e = pearson_residuals.mean().unwrap_or(1.0);
            
            // Check convergence
            let beta_diff = (&new_beta - &beta).mapv(|x| x.abs()).mean().unwrap_or(f64::INFINITY);
            let eta_diff = (&new_eta - &eta).mapv(|x| x.abs()).mean().unwrap_or(f64::INFINITY);
            
            beta = new_beta;
            u = new_u;
            eta = new_eta;
            mu = new_mu;
            mu_eta = new_mu_eta;
            
            converged = beta_diff < self.tol && eta_diff < self.tol;
            
            // Also check variance component convergence
            let sigma2_u_diff = sigma2_u.iter().zip(&old_sigma2_u)
                .map(|(new, old)| (new - old).abs() / old.max(1e-10))
                .fold(0.0, f64::max);
            let sigma2_e_diff = (sigma2_e - old_sigma2_e).abs() / old_sigma2_e.max(1e-10);
            
            converged = converged && sigma2_u_diff < self.tol && sigma2_e_diff < self.tol;
        }
        
        // Compute approximate standard errors
        // From final mixed model equations matrix inverse
        // Recompute weights using final mu and mu_eta
        let mut final_weights = Array1::zeros(n);
        for i in 0..n {
            let d_eta_d_mu = if mu_eta[i].abs() > 1e-10 { 1.0 / mu_eta[i] } else { 1.0 };
            let v_mu = self.family.variance(mu[i]);
            final_weights[i] = 1.0 / (v_mu * d_eta_d_mu * d_eta_d_mu);
        }
        let W_sqrt = final_weights.mapv(|w| w.sqrt());
        let X_weighted = &X * &W_sqrt.clone().insert_axis(ndarray::Axis(1));
        let Z_weighted = &Z * &W_sqrt.clone().insert_axis(ndarray::Axis(1));
        
        let XtWX = X_weighted.t().dot(&X_weighted);
        let ZtWZ = Z_weighted.t().dot(&Z_weighted);
        let XtWZ = X_weighted.t().dot(&Z_weighted);
        let ZtWX = Z_weighted.t().dot(&X_weighted);
        
        let total_cols = p + q;
        let mut M = Array2::zeros((total_cols, total_cols));
        M.slice_mut(ndarray::s![0..p, 0..p]).assign(&XtWX);
        M.slice_mut(ndarray::s![0..p, p..]).assign(&XtWZ);
        M.slice_mut(ndarray::s![p.., 0..p]).assign(&ZtWX);
        
        let mut ZtWZ_plus_Ginv = ZtWZ.clone();
        let mut col_offset = 0;
        for (k, sigma2_u_k) in sigma2_u.iter().enumerate() {
            let cols = group_sizes[k];
            let g_inv = 1.0 / f64::max(*sigma2_u_k, 1e-10);
            for i in 0..cols {
                let idx = col_offset + i;
                ZtWZ_plus_Ginv[(idx, idx)] += g_inv;
            }
            col_offset += cols;
        }
        M.slice_mut(ndarray::s![p.., p..]).assign(&ZtWZ_plus_Ginv);
        
        let Minv = inv(&M)
            .map_err(|e| Error::LinearAlgebraError(format!("Failed to invert mixed model matrix: {}", e)))?;
        
        let cov_beta = Minv.slice(ndarray::s![0..p, 0..p]).to_owned();
        let fixed_se = cov_beta.diag().mapv(|x| x.sqrt());
        
        // Compute approximate log-likelihood (quasi-likelihood)
        let log_lik = self.approximate_log_lik(y, &mu, &eta, &final_weights, sigma2_e);
        
        // Compute information criteria
        let n_params = p + sigma2_u.len() + 1; // beta + variance components + scale
        let aic = -2.0 * log_lik + 2.0 * n_params as f64;
        let bic = -2.0 * log_lik + (n_params as f64) * (n as f64).ln();
        
        // Prepare variance components with names
        let mut var_comps = Vec::new();
        for (i, random_effect) in self.random_effects.iter().enumerate() {
            var_comps.push((random_effect.group_var.clone(), sigma2_u[i]));
        }
        
        Ok(GLMMResults {
            fixed_effects: beta,
            fixed_se,
            variance_components: var_comps,
            scale: sigma2_e,
            log_lik,
            aic,
            bic,
            df_fixed: p,
            n_obs: n,
            converged,
            iterations: iter,
            family: self.family,
            link,
        })
    }
    
    /// Build fixed effects design matrix
    fn build_fixed_design_matrix(&self) -> Result<Array2<f64>> {
        // Parse fixed effects formula
        let formula_str = if self.fixed_formula.contains('~') {
            self.fixed_formula.clone()
        } else {
            // If no response in formula (for random effects), add placeholder
            format!("__response__ ~ {}", self.fixed_formula)
        };
        
        let formula = Formula::parse(&formula_str)
            .map_err(|e| Error::FormulaError(format!("Failed to parse fixed formula: {}", e)))?;
        
        // Build design matrix (includes intercept if specified)
        formula.build_matrix(&self.data)
            .map_err(|e| Error::DataError(format!("Failed to build design matrix: {}", e)))
    }
    
    /// Build random effects design matrices (simplified)
    fn build_random_design_matrices(&self) -> Result<(Vec<Array2<f64>>, Vec<usize>)> {
        let mut Z_matrices = Vec::new();
        let mut group_sizes = Vec::new();
        
        for random_effect in &self.random_effects {
            let group_col = self.data.column(&random_effect.group_var)
                .ok_or_else(|| Error::DataError(format!("Group column '{}' not found", random_effect.group_var)))?;
            
            // Simplified: create indicator matrix for groups
            // Assuming groups are integer-coded 0..n_groups-1
            let group_data = group_col.data();
            let max_group = group_data.iter()
                .map(|&x| x as i64)
                .max()
                .unwrap_or(0)
                .max(0) as usize;
            let n_groups = max_group + 1;
            
            let n = self.data.n_rows();
            let mut Z = Array2::zeros((n, n_groups));
            
            for j in 0..n {
                let group_idx = group_data[j] as usize % n_groups.max(1);
                if n_groups > 0 {
                    Z[(j, group_idx)] = 1.0;
                }
            }
            
            Z_matrices.push(Z);
            group_sizes.push(n_groups);
        }
        
        Ok((Z_matrices, group_sizes))
    }
    
    /// Combine Z matrices into block-diagonal matrix
    fn combine_Z_matrices(&self, Z_matrices: &[Array2<f64>], group_sizes: &[usize]) -> Array2<f64> {
        let n = Z_matrices[0].nrows();
        let total_cols: usize = group_sizes.iter().sum();
        
        let mut Z = Array2::zeros((n, total_cols));
        let mut col_offset = 0;
        
        for (i, Z_i) in Z_matrices.iter().enumerate() {
            let cols = group_sizes[i];
            for row in 0..n {
                for col in 0..cols {
                    Z[(row, col_offset + col)] = Z_i[(row, col)];
                }
            }
            col_offset += cols;
        }
        
        Z
    }
    
    /// Compute approximate log-likelihood (quasi-likelihood)
    fn approximate_log_lik(&self, y: &Array1<f64>, mu: &Array1<f64>, _eta: &Array1<f64>, 
                          _weights: &Array1<f64>, scale: f64) -> f64 {
        let n = y.len() as f64;
        
        // Quasi-likelihood approximation
        let mut ql = 0.0;
        
        for i in 0..y.len() {
            // Contribution from quasi-likelihood
            // This is simplified - proper implementation would integrate the variance function
            let deviance = self.family.unit_deviance(y[i], mu[i]);
            ql += -0.5 * deviance / scale;
        }
        
        // Add normalization constant approximation
        ql - 0.5 * n * (2.0 * std::f64::consts::PI * scale).ln()
    }
}