//! Results structures for mixed models

#![allow(unused_imports)]

use std::collections::HashMap;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::model::{EstimationMethod, LMMConfig};
use super::random_effects::RandomEffect;

/// Results for a single random effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffectResults {
    /// Random effect specification
    pub effect: RandomEffect,
    /// Estimated variance components
    pub variances: Array1<f64>,
    /// Estimated covariance parameters (if applicable)
    pub covariances: Option<Array2<f64>>,
    /// Standard errors for variance components (if available)
    pub std_errors: Option<Array1<f64>>,
    /// Best Linear Unbiased Predictors (BLUPs) for random effects
    pub blups: Option<Array1<f64>>,
    /// Predicted values for each group
    pub group_predictions: Option<HashMap<String, f64>>,
}

/// Main results for a mixed effects model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedModelResults {
    /// Model configuration
    pub config: LMMConfig,
    /// Fixed effects coefficients
    pub fixed_coefficients: Array1<f64>,
    /// Standard errors for fixed effects
    pub fixed_std_errors: Option<Array1<f64>>,
    /// t-statistics for fixed effects
    pub fixed_t_values: Option<Array1<f64>>,
    /// p-values for fixed effects
    pub fixed_p_values: Option<Array1<f64>>,
    /// Random effect results
    pub random_results: Vec<RandomEffectResults>,
    /// Residual variance
    pub residual_variance: f64,
    /// Residual standard error
    pub residual_std_error: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Restricted log-likelihood (for REML)
    pub restricted_log_likelihood: Option<f64>,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion  
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of groups for each random effect
    pub n_groups: Vec<usize>,
    /// Convergence status
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Residuals (observed - fitted)
    pub residuals: Array1<f64>,
    /// Fitted values
    pub fitted_values: Array1<f64>,
    /// Model degrees of freedom
    pub df_model: usize,
    /// Residual degrees of freedom
    pub df_residual: usize,
    /// R-squared (marginal - fixed effects only)
    pub r_squared_marginal: f64,
    /// R-squared (conditional - fixed + random effects)
    pub r_squared_conditional: f64,
    /// Variance-covariance matrix of fixed effects
    pub vcov_fixed: Option<Array2<f64>>,
    /// Random effects covariance matrices
    pub vcov_random: Vec<Array2<f64>>,
}

impl MixedModelResults {
    /// Create a summary string similar to R's summary()
    pub fn summary(&self, feature_names: &[String]) -> String {
        let mut summary = String::new();
        
        summary.push_str(&format!("Mixed Effects Model Results\n"));
        summary.push_str(&format!("===========================\n"));
        
        // Model information
        summary.push_str(&format!("Method: {}\n", 
            match self.config.method {
                super::model::EstimationMethod::ML => "Maximum Likelihood",
                super::model::EstimationMethod::REML => "Restricted Maximum Likelihood",
            }));
        
        summary.push_str(&format!("Observations: {}\n", self.n_obs));
        summary.push_str(&format!("Groups: {}\n", self.n_groups.iter().sum::<usize>()));
        
        for (i, n_groups) in self.n_groups.iter().enumerate() {
            if i < self.config.random_effects.len() {
                let effect = &self.config.random_effects[i];
                summary.push_str(&format!("  {}: {} groups\n", effect.grouping, n_groups));
            }
        }
        
        // Fit statistics
        summary.push_str(&format!("\nFit Statistics:\n"));
        summary.push_str(&format!("  Log-Likelihood: {:.3}\n", self.log_likelihood));
        if let Some(reml_ll) = self.restricted_log_likelihood {
            summary.push_str(&format!("  REML Log-Likelihood: {:.3}\n", reml_ll));
        }
        summary.push_str(&format!("  AIC: {:.3}\n", self.aic));
        summary.push_str(&format!("  BIC: {:.3}\n", self.bic));
        summary.push_str(&format!("  R² (marginal): {:.4}\n", self.r_squared_marginal));
        summary.push_str(&format!("  R² (conditional): {:.4}\n", self.r_squared_conditional));
        summary.push_str(&format!("  Residual Std. Error: {:.4}\n", self.residual_std_error));
        
        // Fixed effects
        summary.push_str(&format!("\nFixed Effects:\n"));
        summary.push_str(&format!("              Estimate Std. Error t value Pr(>|t|)\n"));
        
        for i in 0..self.fixed_coefficients.len() {
            let name = if i == 0 && self.config.intercept {
                "(Intercept)".to_string()
            } else if self.config.intercept {
                feature_names.get(i).cloned().unwrap_or_else(|| format!("x{}", i))
            } else {
                feature_names.get(i).cloned().unwrap_or_else(|| format!("x{}", i))
            };
            
            let coef = self.fixed_coefficients[i];
            let se = self.fixed_std_errors.as_ref().map_or(f64::NAN, |se| se[i]);
            let t = self.fixed_t_values.as_ref().map_or(f64::NAN, |t| t[i]);
            let p = self.fixed_p_values.as_ref().map_or(f64::NAN, |p| p[i]);
            
            let significance = if p < 0.001 {
                "***"
            } else if p < 0.01 {
                "**"
            } else if p < 0.05 {
                "*"
            } else if p < 0.1 {
                "."
            } else {
                ""
            };
            
            summary.push_str(&format!("{:15} {:8.4} {:8.4} {:7.3} {:8.4} {}\n",
                name, coef, se, t, p, significance));
        }
        
        // Random effects
        summary.push_str(&format!("\nRandom Effects:\n"));
        
        for (_i, random_result) in self.random_results.iter().enumerate() {
            let effect = &random_result.effect;
            summary.push_str(&format!("  {}:\n", effect.grouping));
            
            match effect.effect_type {
                super::random_effects::RandomEffectType::Intercept => {
                    summary.push_str(&format!("    Intercept variance: {:.4}\n", 
                        random_result.variances[0]));
                }
                super::random_effects::RandomEffectType::Slope => {
                    if let Some(ref predictor) = effect.predictor {
                        summary.push_str(&format!("    {} slope variance: {:.4}\n", 
                            predictor, random_result.variances[0]));
                    }
                }
                super::random_effects::RandomEffectType::InterceptSlope => {
                    if let Some(ref predictor) = effect.predictor {
                        summary.push_str(&format!("    Intercept variance: {:.4}\n", 
                            random_result.variances[0]));
                        summary.push_str(&format!("    {} slope variance: {:.4}\n", 
                            predictor, random_result.variances[1]));
                        if let Some(ref cov) = random_result.covariances {
                            summary.push_str(&format!("    Correlation: {:.4}\n", 
                                cov[(0, 1)] / (random_result.variances[0] * random_result.variances[1]).sqrt()));
                        }
                    }
                }
                super::random_effects::RandomEffectType::Uncorrelated => {
                    if let Some(ref predictor) = effect.predictor {
                        summary.push_str(&format!("    Intercept variance: {:.4}\n", 
                            random_result.variances[0]));
                        summary.push_str(&format!("    {} slope variance: {:.4}\n", 
                            predictor, random_result.variances[1]));
                    }
                }
            }
        }
        
        // Residual variance
        summary.push_str(&format!("\nResidual Variance: {:.4}\n", self.residual_variance));
        
        // Convergence
        summary.push_str(&format!("\nConvergence: {}\n", 
            if self.converged { "Yes" } else { "No" }));
        summary.push_str(&format!("Iterations: {}\n", self.iterations));
        
        summary.push_str(&format!("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n"));
        
        summary
    }
    
    /// Predict using fitted model
    pub fn predict(&self, X: &Array2<f64>, Z: Option<&[Array2<f64>]>, 
                   groups: Option<&[Array1<usize>]>) -> Array1<f64> {
        // Fixed effects prediction
        let y_pred = X.dot(&self.fixed_coefficients);
        
        // Add random effects if provided
        if let (Some(_z_matrices), Some(_group_ids)) = (Z, groups) {
            for (_i, _z_matrix) in _z_matrices.iter().enumerate() {
                if _i < self.random_results.len() {
                    if let Some(ref _blups) = self.random_results[_i].blups {
                        // TODO: Implement proper BLUP addition
                        // This is simplified - in practice would need to map BLUPs to observations
                    }
                }
            }
        }
        
        y_pred
    }
    
    /// Get variance-covariance matrix of fixed effects
    pub fn vcov_fixed(&self) -> Option<&Array2<f64>> {
        self.vcov_fixed.as_ref()
    }
    
    /// Get intraclass correlation coefficient (ICC)
    pub fn icc(&self) -> f64 {
        // ICC = random effect variance / (random effect variance + residual variance)
        let total_random_variance: f64 = self.random_results.iter()
            .map(|r| r.variances.sum())
            .sum();
        
        if total_random_variance + self.residual_variance == 0.0 {
            0.0
        } else {
            total_random_variance / (total_random_variance + self.residual_variance)
        }
    }
    
    /// Perform likelihood ratio test against null model
    pub fn lrt_test(&self, null_model: &MixedModelResults) -> (f64, f64) {
        let chi2 = 2.0 * (self.log_likelihood - null_model.log_likelihood);
        let df = (self.df_model - null_model.df_model) as f64;
        
        // p-value from chi-squared distribution
        let p_value = 1.0 - statrs::function::gamma::gamma_ur(df / 2.0, chi2 / 2.0);
        
        (chi2, p_value)
    }
}