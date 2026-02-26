//! Linear regression models
//!
//! This module provides various linear regression models:
//! - Ordinary Least Squares (OLS)
//! - Weighted Least Squares (WLS)
//! - Ridge Regression
//! - LASSO Regression
//! - Elastic Net
//!
//! All models support R-style formula syntax via the formula module.

pub mod diagnostics;
pub mod elastic_net;
pub mod lasso;
pub mod ols;
pub mod result;
pub mod ridge;
pub mod wls;

#[cfg(test)]
mod tests;

// Re-exports
// pub use elastic_net::ElasticNetRegression;
// pub use lasso::LassoRegression;
pub use ols::LinearRegression;
// pub use ridge::RidgeRegression;
// pub use wls::WeightedLinearRegression;

// Common types
use crate::base::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use so_core::data::DataFrame;

/// Linear model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearConfig {
    /// Whether to include intercept
    pub intercept: bool,
    /// Standard error type
    pub se_type: StandardErrorType,
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Whether to center predictors
    pub center_predictors: bool,
    /// Whether to scale predictors
    pub scale_predictors: bool,
    /// Robust regression method
    pub robust_method: Option<RobustMethod>,
}

/// Standard error types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StandardErrorType {
    /// Standard errors assuming homoscedasticity
    Standard,
    /// Heteroscedasticity-consistent standard errors
    HC0,
    HC1,
    HC2,
    HC3,
    /// Heteroscedasticity and autocorrelation consistent
    HAC,
}

/// Robust regression methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RobustMethod {
    Huber,
    Bisquare,
    Hampel,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            intercept: true,
            se_type: StandardErrorType::Standard,
            confidence_level: 0.95,
            center_predictors: false,
            scale_predictors: false,
            robust_method: None,
        }
    }
}

/// Linear model trait
pub trait LinearModel {
    /// Fit the model
    fn fit(&mut self) -> Result<&mut Self>;

    /// Get coefficients
    fn coefficients(&self) -> Option<&Array1<f64>>;

    /// Get standard errors
    fn standard_errors(&self) -> Option<&Array1<f64>>;

    /// Get predictions
    fn predict(&self, data: &DataFrame) -> Result<Array1<f64>>;

    /// Get fitted values
    fn fitted_values(&self) -> Option<&Array1<f64>>;

    /// Get residuals
    fn residuals(&self) -> Option<&Array1<f64>>;
}

/// Convenience function for OLS regression
pub fn lm(formula: &str, data: &DataFrame) -> Result<LinearRegression> {
    LinearRegression::new(formula)?.data(data).fit()
}
