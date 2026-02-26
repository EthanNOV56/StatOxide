//! GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
//!
//! This module implements GARCH models for volatility clustering in time series.
//!
//! # Model Specification
//!
//! GARCH(p, q) models conditional variance as:
//! σₜ² = ω + Σᵢ₌₁ᵖ αᵢ εₜ₋ᵢ² + Σⱼ₌₁ᵠ βⱼ σₜ₋ⱼ²
//!
//! where:
//! - εₜ = σₜ zₜ, zₜ ∼ i.i.d. with mean 0, variance 1
//! - ω > 0, αᵢ ≥ 0, βⱼ ≥ 0 for stationarity
//! - Σ(αᵢ + βⱼ) < 1 for covariance stationarity
//!
//! Special cases:
//! - ARCH(q): GARCH(0, q) - only past squared errors
//! - GARCH(1,1): Most commonly used specification
//!
//! # Distributions for Innovations
//!
//! 1. **Normal**: zₜ ∼ N(0, 1)
//! 2. **Student's t**: zₜ ∼ t(ν) with ν degrees of freedom
//! 3. **Generalized Error Distribution (GED)**: Flexible tail behavior

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use statrs::function::gamma;
use super::timeseries::TimeSeries;
use so_core::error::{Result, Error};

/// Distribution for GARCH innovations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GARCHDistribution {
    /// Standard normal distribution
    Normal,
    /// Student's t distribution with ν degrees of freedom
    StudentsT(f64),
    /// Generalized Error Distribution with shape parameter ν
    GED(f64),
}

/// GARCH model order
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GARCHOrder {
    /// ARCH order (q) - lagged squared errors
    pub p: usize,
    /// GARCH order (p) - lagged conditional variances
    pub q: usize,
}

/// GARCH model configuration
#[derive(Debug, Clone)]
pub struct GARCHConfig {
    /// Model order
    pub order: GARCHOrder,
    /// Distribution for innovations
    pub distribution: GARCHDistribution,
    /// Include constant in mean equation
    pub with_constant: bool,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl Default for GARCHConfig {
    fn default() -> Self {
        Self {
            order: GARCHOrder { p: 1, q: 1 },
            distribution: GARCHDistribution::Normal,
            with_constant: false,
            max_iter: 100,
            tol: 1e-6,
        }
    }
}

/// GARCH model results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GARCHResults {
    /// Constant in variance equation (ω)
    pub omega: f64,
    /// ARCH coefficients (α₁, ..., α_q)
    pub arch_coef: Array1<f64>,
    /// GARCH coefficients (β₁, ..., β_p)
    pub garch_coef: Array1<f64>,
    /// Constant in mean equation (if included)
    pub mu: Option<f64>,
    /// Degrees of freedom (for t/GED distributions)
    pub df: Option<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Residuals (εₜ)
    pub residuals: Array1<f64>,
    /// Conditional variances (σₜ²)
    pub conditional_variances: Array1<f64>,
    /// Standardized residuals (zₜ = εₜ/σₜ)
    pub standardized_residuals: Array1<f64>,
}

/// GARCH model builder
pub struct GARCHBuilder {
    config: GARCHConfig,
}

impl GARCHBuilder {
    /// Create new GARCH builder
    pub fn new(p: usize, q: usize) -> Self {
        Self {
            config: GARCHConfig {
                order: GARCHOrder { p, q },
                ..Default::default()
            },
        }
    }
    
    /// Create ARCH builder (GARCH with p=0)
    pub fn arch(q: usize) -> Self {
        Self::new(0, q)
    }
    
    /// Set distribution for innovations
    pub fn distribution(mut self, distribution: GARCHDistribution) -> Self {
        self.config.distribution = distribution;
        self
    }
    
    /// Include constant in mean equation
    pub fn with_constant(mut self, include: bool) -> Self {
        self.config.with_constant = include;
        self
    }
    
    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }
    
    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }
    
    /// Fit GARCH model
    pub fn fit(self, ts: &TimeSeries) -> Result<GARCHResults> {
        let mut garch = GARCH::new(self.config);
        garch.fit(ts)
    }
}

/// GARCH model
pub struct GARCH {
    config: GARCHConfig,
}

impl GARCH {
    /// Create new GARCH model
    pub fn new(config: GARCHConfig) -> Self {
        Self { config }
    }
    
    /// Create GARCH builder
    pub fn builder(p: usize, q: usize) -> GARCHBuilder {
        GARCHBuilder::new(p, q)
    }
    
    /// Create ARCH builder
    pub fn arch(q: usize) -> GARCHBuilder {
        GARCHBuilder::arch(q)
    }
    
    /// Fit GARCH model to time series
    pub fn fit(&mut self, ts: &TimeSeries) -> Result<GARCHResults> {
        let y = ts.values();
        let n = y.len();
        let order = self.config.order;
        
        if n < order.p.max(order.q) + 10 {
            return Err(Error::DataError(
                format!("Not enough observations for GARCH({},{}), need at least {}, got {}",
                    order.p, order.q, order.p.max(order.q) + 10, n)
            ));
        }
        
        // Estimate mean equation if constant is included
        let (residuals, mu) = if self.config.with_constant {
            let mean = y.mean().unwrap_or(0.0);
            let residuals = y - mean;
            (residuals, Some(mean))
        } else {
            (y.clone(), None)
        };
        
        // Initial parameter estimates
        let mut params = self.initial_parameters(&residuals);
        
        // Maximize log-likelihood
        let (final_params, log_lik) = self.maximize_likelihood(&residuals, &mut params)?;
        
        // Extract parameters
        let (omega, arch_coef, garch_coef, df) = self.extract_parameters(&final_params);
        
        // Calculate conditional variances and standardized residuals
        let (conditional_variances, standardized_residuals) = 
            self.calculate_conditional_variances(&residuals, omega, &arch_coef, &garch_coef);
        
        // Calculate information criteria
        let n_params = 1 + order.q + order.p + if self.config.with_constant { 1 } else { 0 } + 
            match self.config.distribution {
                GARCHDistribution::Normal => 0,
                GARCHDistribution::StudentsT(_) => 1,
                GARCHDistribution::GED(_) => 1,
            };
        
        let aic = 2.0 * n_params as f64 - 2.0 * log_lik;
        let bic = (n as f64).ln() * n_params as f64 - 2.0 * log_lik;
        
        Ok(GARCHResults {
            omega,
            arch_coef,
            garch_coef,
            mu,
            df,
            log_likelihood: log_lik,
            aic,
            bic,
            n_obs: n,
            residuals: residuals.clone(),
            conditional_variances,
            standardized_residuals,
        })
    }
    
    /// Generate initial parameter estimates
    fn initial_parameters(&self, residuals: &Array1<f64>) -> Array1<f64> {
        let order = self.config.order;
        let n_params = 1 + order.q + order.p; // ω + α + β
        
        let mut params = Array1::zeros(n_params);
        
        // Initial variance (ω)
        let variance = residuals.var(1.0);
        params[0] = variance * 0.1;
        
        // ARCH coefficients (α) - sum to 0.1
        if order.q > 0 {
            let alpha_sum = 0.1;
            for i in 0..order.q {
                params[1 + i] = alpha_sum / order.q as f64;
            }
        }
        
        // GARCH coefficients (β) - sum to 0.8
        if order.p > 0 {
            let beta_sum = 0.8;
            for i in 0..order.p {
                params[1 + order.q + i] = beta_sum / order.p as f64;
            }
        }
        
        // Add distribution parameter if needed
        match self.config.distribution {
            GARCHDistribution::StudentsT(_) => {
                let mut extended = Array1::zeros(n_params + 1);
                extended.slice_mut(ndarray::s![..n_params]).assign(&params);
                extended[n_params] = 8.0; // Initial degrees of freedom
                extended
            },
            GARCHDistribution::GED(_) => {
                let mut extended = Array1::zeros(n_params + 1);
                extended.slice_mut(ndarray::s![..n_params]).assign(&params);
                extended[n_params] = 1.5; // Initial shape parameter
                extended
            },
            GARCHDistribution::Normal => params,
        }
    }
    
    /// Maximize log-likelihood using gradient-based optimization
    fn maximize_likelihood(
        &self,
        residuals: &Array1<f64>,
        params: &mut Array1<f64>,
    ) -> Result<(Array1<f64>, f64)> {
        let _n = residuals.len();
        let order = self.config.order;
        
        let mut log_lik_old = f64::NEG_INFINITY;
        let mut iteration = 0;
        
        while iteration < self.config.max_iter {
            // Calculate log-likelihood and gradient
            let (log_lik, gradient) = self.log_likelihood_and_gradient(residuals, params);
            
            // Check convergence
            if (log_lik - log_lik_old).abs() < self.config.tol {
                return Ok((params.clone(), log_lik));
            }
            
            // Simple gradient ascent (in practice would use BFGS or Newton)
            let step_size = 0.01;
            for i in 0..params.len() {
                params[i] += step_size * gradient[i];
                
                // Apply constraints
                if i == 0 {
                    // ω > 0
                    params[i] = params[i].max(1e-8);
                } else if i <= order.q {
                    // α ≥ 0
                    params[i] = params[i].max(0.0);
                } else if i <= order.q + order.p {
                    // β ≥ 0
                    params[i] = params[i].max(0.0);
                } else if let GARCHDistribution::StudentsT(_) = self.config.distribution {
                    // ν > 2 for finite variance
                    params[i] = params[i].max(2.1);
                } else if let GARCHDistribution::GED(_) = self.config.distribution {
                    // ν > 0
                    params[i] = params[i].max(0.1);
                }
            }
            
            // Ensure stationarity: Σ(α + β) < 1
            let alpha_sum: f64 = (1..=order.q).map(|i| params[i]).sum();
            let beta_sum: f64 = (1..=order.p).map(|i| params[order.q + i]).sum();
            
            if alpha_sum + beta_sum >= 1.0 {
                // Scale down coefficients
                let scale = 0.99 / (alpha_sum + beta_sum);
                for i in 1..=order.q {
                    params[i] *= scale;
                }
                for i in 1..=order.p {
                    params[order.q + i] *= scale;
                }
            }
            
            log_lik_old = log_lik;
            iteration += 1;
        }
        
        Err(Error::ModelError(
            format!("GARCH optimization did not converge after {} iterations", self.config.max_iter)
        ))
    }
    
    /// Calculate log-likelihood and gradient
    fn log_likelihood_and_gradient(
        &self,
        residuals: &Array1<f64>,
        params: &Array1<f64>,
    ) -> (f64, Array1<f64>) {
        let _n = residuals.len();
        let order = self.config.order;
        
        // Extract parameters
        let (omega, arch_coef, garch_coef, df) = self.extract_parameters(params);
        
        // Calculate conditional variances
        let conditional_variances = self.calculate_variances(residuals, omega, &arch_coef, &garch_coef);
        
        // Calculate log-likelihood
        let mut log_lik = 0.0;
        let mut gradient = Array1::zeros(params.len());
        
        for t in order.q.max(order.p)..n {
            let sigma2 = conditional_variances[t];
            let sigma = sigma2.sqrt();
            let z = residuals[t] / sigma;
            
            // Log-likelihood contribution
            match self.config.distribution {
                GARCHDistribution::Normal => {
                    log_lik += -0.5 * (2.0 * std::f64::consts::PI * sigma2).ln() - 0.5 * z.powi(2);
                },
                GARCHDistribution::StudentsT(nu) => {
                    // Student's t log-likelihood
                    let nu = nu.max(2.1);
                    let constant = gamma::ln_gamma((nu + 1.0) / 2.0) - gamma::ln_gamma(nu / 2.0)
                        - 0.5 * (std::f64::consts::PI * (nu - 2.0)).ln();
                    log_lik += constant - 0.5 * sigma2.ln() 
                        - ((nu + 1.0) / 2.0) * (1.0 + z.powi(2) / (nu - 2.0)).ln();
                },
                GARCHDistribution::GED(nu) => {
                    // GED log-likelihood
                    let nu = nu.max(0.1);
                    let lambda = (2.0f64.powf(-2.0 / nu) * gamma::gamma(1.0 / nu) / gamma::gamma(3.0 / nu)).sqrt();
                    let constant = -0.5 * (lambda.powi(2) * gamma::gamma(1.0 / nu) / gamma::gamma(3.0 / nu)).ln()
                        - gamma::ln_gamma(1.0 + 1.0 / nu);
                    log_lik += constant - 0.5 * sigma2.ln() 
                        - 0.5 * (z.abs() / lambda).powf(nu);
                },
            }
            
            // Gradient calculation (simplified)
            // In practice would implement full gradient
            let eps = 1e-8;
            for i in 0..params.len() {
                let mut params_plus = params.clone();
                params_plus[i] += eps;
                let (omega_p, arch_coef_p, garch_coef_p, _) = self.extract_parameters(&params_plus);
                let sigma2_p = self.calculate_variance_t(residuals, t, omega_p, &arch_coef_p, &garch_coef_p);
                
                let mut params_minus = params.clone();
                params_minus[i] -= eps;
                let (omega_m, arch_coef_m, garch_coef_m, _) = self.extract_parameters(&params_minus);
                let sigma2_m = self.calculate_variance_t(residuals, t, omega_m, &arch_coef_m, &garch_coef_m);
                
                let deriv = (sigma2_p - sigma2_m) / (2.0 * eps);
                gradient[i] += deriv * (-0.5 / sigma2 + 0.5 * z.powi(2) / sigma2.powi(2));
            }
        }
        
        (log_lik, gradient)
    }
    
    /// Extract parameters from parameter vector
    fn extract_parameters(&self, params: &Array1<f64>) -> (f64, Array1<f64>, Array1<f64>, Option<f64>) {
        let order = self.config.order;
        
        let omega = params[0];
        
        let arch_coef = if order.q > 0 {
            params.slice(ndarray::s![1..1+order.q]).to_owned()
        } else {
            Array1::zeros(0)
        };
        
        let garch_coef = if order.p > 0 {
            params.slice(ndarray::s![1+order.q..1+order.q+order.p]).to_owned()
        } else {
            Array1::zeros(0)
        };
        
        let _df = match self.config.distribution {
            GARCHDistribution::Normal => None,
            GARCHDistribution::StudentsT(_) => Some(params[params.len() - 1]),
            GARCHDistribution::GED(_) => Some(params[params.len() - 1]),
        };
        
        (omega, arch_coef, garch_coef, df)
    }
    
    /// Calculate conditional variances for all time points
    fn calculate_variances(
        &self,
        residuals: &Array1<f64>,
        omega: f64,
        arch_coef: &Array1<f64>,
        garch_coef: &Array1<f64>,
    ) -> Array1<f64> {
        let _n = residuals.len();
        let p = garch_coef.len();
        let q = arch_coef.len();
        let max_lag = p.max(q);
        
        let mut variances = Array1::zeros(n);
        
        // Initial variance (unconditional)
        let initial_variance = residuals.var(1.0).max(1e-8);
        
        for t in 0..n {
            if t < max_lag {
                variances[t] = initial_variance;
            } else {
                let mut variance = omega;
                
                // ARCH terms
                for lag in 1..=q {
                    variance += arch_coef[lag - 1] * residuals[t - lag].powi(2);
                }
                
                // GARCH terms
                for lag in 1..=p {
                    variance += garch_coef[lag - 1] * variances[t - lag];
                }
                
                variances[t] = variance.max(1e-8);
            }
        }
        
        variances
    }
    
    /// Calculate conditional variance at specific time point
    fn calculate_variance_t(
        &self,
        residuals: &Array1<f64>,
        t: usize,
        omega: f64,
        arch_coef: &Array1<f64>,
        garch_coef: &Array1<f64>,
    ) -> f64 {
        let p = garch_coef.len();
        let q = arch_coef.len();
        
        let mut variance = omega;
        
        // ARCH terms
        for lag in 1..=q {
            if t >= lag {
                variance += arch_coef[lag - 1] * residuals[t - lag].powi(2);
            }
        }
        
        // GARCH terms (would need previous variances)
        // For gradient calculation, we approximate
        for lag in 1..=p {
            variance += garch_coef[lag - 1] * omega / (1.0 - arch_coef.sum() - garch_coef.sum());
        }
        
        variance.max(1e-8)
    }
    
    /// Calculate conditional variances and standardized residuals
    fn calculate_conditional_variances(
        &self,
        residuals: &Array1<f64>,
        omega: f64,
        arch_coef: &Array1<f64>,
        garch_coef: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let variances = self.calculate_variances(residuals, omega, arch_coef, garch_coef);
        let standardized = residuals / &variances.mapv(|v| v.sqrt());
        
        (variances, standardized)
    }
    
    /// Forecast conditional variances
    pub fn forecast_variances(
        &self,
        results: &GARCHResults,
        steps: usize,
    ) -> Array1<f64> {
        let n = results.n_obs;
        let order = self.config.order;
        
        let mut forecasts = Array1::zeros(steps);
        let mut past_variances = results.conditional_variances.clone();
        let mut past_residuals = results.residuals.clone();
        
        // Unconditional variance
        let unconditional_variance = results.omega / 
            (1.0 - results.arch_coef.sum() - results.garch_coef.sum());
        
        for h in 0..steps {
            let mut variance = results.omega;
            
            // ARCH terms
            for lag in 1..=order.q {
                let idx = n + h - lag;
                if idx < n {
                    variance += results.arch_coef[lag - 1] * past_residuals[idx].powi(2);
                } else if idx < n + h {
                    // Use forecasted residual (assume zero for squared residual expectation)
                    variance += results.arch_coef[lag - 1] * unconditional_variance;
                }
            }
            
            // GARCH terms
            for lag in 1..=order.p {
                let idx = n + h - lag;
                if idx < n {
                    variance += results.garch_coef[lag - 1] * past_variances[idx];
                } else if idx < n + h {
                    variance += results.garch_coef[lag - 1] * forecasts[idx - n];
                }
            }
            
            forecasts[h] = variance.max(1e-8);
            
            // Extend arrays for next forecast
            past_variances = ndarray::concatenate(
                ndarray::Axis(0),
                &[past_variances.view(), ndarray::array![variance].view()]
            ).unwrap();
            
            past_residuals = ndarray::concatenate(
                ndarray::Axis(0),
                &[past_residuals.view(), ndarray::array![0.0].view()]
            ).unwrap();
        }
        
        forecasts
    }
}

/// ARCH model (special case of GARCH with p=0)
pub type ARCH = GARCH;

/// Extension trait for TimeSeries
pub trait GARCHExt {
    /// Fit GARCH model
    fn garch(&self, p: usize, q: usize) -> Result<GARCHResults>;
    
    /// Fit ARCH model
    fn arch(&self, q: usize) -> Result<GARCHResults>;
}

impl GARCHExt for TimeSeries {
    fn garch(&self, p: usize, q: usize) -> Result<GARCHResults> {
        GARCH::builder(p, q).fit(self)
    }
    
    fn arch(&self, q: usize) -> Result<GARCHResults> {
        GARCH::arch(q).fit(self)
    }
}