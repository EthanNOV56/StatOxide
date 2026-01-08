//! Model-related error types

use thiserror::Error;

use so_core::data::DataError;
use so_core::formula::error::FormulaError;

/// Model-related errors
#[derive(Debug, Error)]
pub enum ModelError {
    /// Formula parsing or evaluation error
    #[error("Formula error: {0}")]
    Formula(#[from] FormulaError),

    /// Data-related error
    #[error("Data error: {0}")]
    Data(#[from] DataError),

    /// Numerical computation error
    #[error("Numerical error: {message} (operation: {operation})")]
    NumericalError {
        /// Error message
        message: String,
        /// Operation that failed
        operation: String,
    },

    /// Insufficient data for model fitting
    #[error("Not enough data: {n_samples} samples for {n_predictors} predictors")]
    InsufficientData {
        /// Number of samples
        n_samples: usize,
        /// Number of predictors
        n_predictors: usize,
    },

    /// Model fitting failed to converge
    #[error("Failed to converge after {max_iter} iterations")]
    NotConverged {
        /// Maximum number of iterations attempted
        max_iter: usize,
    },

    /// Singular matrix encountered
    #[error("Singular matrix encountered")]
    SingularMatrix,

    /// Invalid model configuration
    #[error("Invalid model configuration: {message}")]
    InvalidConfig {
        /// Configuration error message
        message: String,
    },

    /// Model not fitted yet
    #[error("Model not fitted yet")]
    NotFitted,

    /// Prediction error
    #[error("Prediction error: {message}")]
    PredictionError {
        /// Prediction error message
        message: String,
    },

    /// Custom error
    #[error("{message}")]
    Custom {
        /// Custom error message
        message: String,
    },
}
