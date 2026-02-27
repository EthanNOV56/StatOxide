//! Error types for linear algebra operations

use thiserror::Error;

/// Linear algebra errors
#[derive(Error, Debug)]
pub enum LinalgError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Matrix must be square, got {rows}x{cols}")]
    NotSquare { rows: usize, cols: usize },

    #[error("Singular matrix encountered")]
    SingularMatrix,

    #[error("Linear algebra operation failed: {0}")]
    OperationFailed(String),

    #[error("Backend not available: {0}")]
    BackendUnavailable(String),
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, LinalgError>;
