use crate::data::DataError;

#[derive(thiserror::Error, Debug)]
pub enum StatOxideError {
    #[error("Data error: {0}")]
    Data(#[from] DataError),

    #[error("Linear algebra error: {0}")]
    LinearAlgebra(String),

    #[error("Statistical error: {0}")]
    Stats(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Formula parsing error: {0}")]
    Formula(String),
}
