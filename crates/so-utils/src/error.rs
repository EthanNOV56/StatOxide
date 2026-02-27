//! Error types for utility operations

use thiserror::Error;

/// Main error type for utility operations
#[derive(Error, Debug)]
pub enum UtilsError {
    #[error("Data error: {0}")]
    DataError(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// Result type alias for utility operations
pub type Result<T> = std::result::Result<T, UtilsError>;
