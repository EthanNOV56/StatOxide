//! Statistical structures for model results

use serde::{Deserialize, Serialize};

/// Model statistics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelStatistics {
    /// R-squared
    pub r_squared: Option<f64>,
    /// Adjusted R-squared
    pub adj_r_squared: Option<f64>,
    /// Residual standard error
    pub residual_std_error: Option<f64>,
    /// F-statistic
    pub f_statistic: Option<f64>,
    /// F-statistic p-value
    pub f_p_value: Option<f64>,
    /// Log-likelihood
    pub log_likelihood: Option<f64>,
    /// AIC
    pub aic: Option<f64>,
    /// BIC
    pub bic: Option<f64>,
    /// Null deviance
    pub null_deviance: Option<f64>,
    /// Residual deviance
    pub residual_deviance: Option<f64>,
    /// Chi-squared statistic
    pub chi_squared: Option<f64>,
    /// Chi-squared p-value
    pub chi_squared_p_value: Option<f64>,
    /// Residual degrees of freedom
    pub df_residual: Option<usize>,
    /// Model degrees of freedom
    pub df_model: Option<usize>,
    /// Null model degrees of freedom
    pub df_null: Option<usize>,
    /// Number of iterations
    pub iterations: Option<usize>,
    /// Convergence status
    pub converged: Option<bool>,
    /// Model weights
    pub weights: Option<f64>,
}

impl Default for ModelStatistics {
    fn default() -> Self {
        Self {
            r_squared: None,
            adj_r_squared: None,
            residual_std_error: None,
            f_statistic: None,
            f_p_value: None,
            log_likelihood: None,
            aic: None,
            bic: None,
            null_deviance: None,
            residual_deviance: None,
            chi_squared: None,
            chi_squared_p_value: None,
            df_residual: None,
            df_model: None,
            df_null: None,
            iterations: None,
            converged: None,
            weights: None,
        }
    }
}

/// Residual statistics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResidualStatistics {
    /// Minimum residual
    pub min: f64,
    /// First quartile
    pub q1: f64,
    /// Median
    pub median: f64,
    /// Third quartile
    pub q3: f64,
    /// Maximum residual
    pub max: f64,
    /// Mean residual
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Skewness
    pub skewness: Option<f64>,
    /// Kurtosis
    pub kurtosis: Option<f64>,
    /// Durbin-Watson statistic
    pub durbin_watson: Option<f64>,
}

impl Default for ResidualStatistics {
    fn default() -> Self {
        Self {
            min: 0.0,
            q1: 0.0,
            median: 0.0,
            q3: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            skewness: None,
            kurtosis: None,
            durbin_watson: None,
        }
    }
}
