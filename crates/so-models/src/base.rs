//! Core traits and types for statistical models
//!
//! This module defines the core interfaces and data structures
//! for statistical models in StatOxide.

use ndarray::Array1;
use serde::Serialize;

use so_core::data::DataFrame;

// Re-export core types
pub use coefficient::Coefficient;
pub use statistics::ModelStatistics;
pub use statistics::ResidualStatistics;
pub use summary::ModelSummary;
pub use summary::ModelType;

pub use crate::error::ModelError;

pub mod coefficient;
pub mod statistics;
pub mod summary;

/// Result type for model operations
pub type Result<T> = std::result::Result<T, ModelError>;

/// Base trait for all statistical models
pub trait StatisticalModel: Send + Sync {
    /// Model parameters type
    type Params: Clone + Serialize;
    /// Model result type
    type Result: ModelResultTrait;

    /// Fit the model to data
    fn fit(&mut self) -> Result<()>;

    /// Predict using the fitted model
    fn predict(&self, data: &DataFrame) -> Result<Array1<f64>>;

    /// Get model parameters
    fn params(&self) -> Option<&Self::Params>;

    /// Get model result
    fn result(&self) -> Option<&Self::Result>;

    /// Generate model summary
    fn summary(&self) -> Result<ModelSummary>;

    /// Check if model is fitted
    fn is_fitted(&self) -> bool;
}

/// Trait for model results
pub trait ModelResultTrait: Send + Sync {
    /// Get coefficients
    fn coefficients(&self) -> &[Coefficient];

    /// Get fitted values
    fn fitted_values(&self) -> &Array1<f64>;

    /// Get residuals
    fn residuals(&self) -> &Array1<f64>;

    /// Get R-squared value
    fn r_squared(&self) -> Option<f64>;

    /// Get adjusted R-squared value
    fn adj_r_squared(&self) -> Option<f64>;

    /// Get AIC
    fn aic(&self) -> Option<f64>;

    /// Get BIC
    fn bic(&self) -> Option<f64>;

    /// Get log-likelihood
    fn log_likelihood(&self) -> Option<f64>;

    /// Get residual standard error
    fn residual_std_error(&self) -> Option<f64>;

    /// Get degrees of freedom for residuals
    fn df_residual(&self) -> Option<usize>;

    /// Get degrees of freedom for model
    fn df_model(&self) -> Option<usize>;
}
