//! Regression models
//!
//! This module provides various regression models:
//! - Ordinary Least Squares (OLS)
//! - Ridge Regression (L2 regularization)
//! - Lasso Regression (L1 regularization) - TODO
//! - Elastic Net (L1 + L2 regularization) - TODO

mod linear;

// Re-exports
pub use linear::{LinearModelBuilder, LinearRegressionResults, OLS, Ridge};
