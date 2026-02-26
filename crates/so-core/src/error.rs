//! Error types for so-core

use thiserror::Error;

/// Core error type
#[derive(Error, Debug)]
pub enum Error {
    /// Dimension mismatch in arrays or data structures
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    
    /// Invalid data (NaN, Inf, etc.)
    #[error("Invalid data: {0}")]
    DataError(String),
    
    /// Invalid formula syntax
    #[error("Formula error: {0}")]
    FormulaError(String),
    
    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinearAlgebraError(String),
    
    /// General error with message
    #[error("{0}")]
    Message(String),
    
    /// Wrapper for other errors
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, Error>;