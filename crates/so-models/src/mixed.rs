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
use so_linalg::{solve, inv};

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
        // Simplified: just intercept for now
        // In practice, parse formula and create design matrix
        let n = self.data.n_rows();
        Ok(Array2::ones((n, 1)))  // Intercept only
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

/// Generalized Linear Mixed Model builder (placeholder)
pub struct GLMMBuilder {
    // To be implemented
}

impl GLMMBuilder {
    /// Create a new GLMM builder
    pub fn new(_data: DataFrame, _response: &str, _family: crate::glm::Family) -> Self {
        Self {
            // To be implemented
        }
    }
    
    /// Fit the GLMM (placeholder)
    pub fn fit(self) -> Result<()> {
        Err(Error::Message("GLMM not yet implemented".to_string()))
    }
}