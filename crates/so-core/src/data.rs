//! Core data structures for StatOxide
//!
//! This module provides the foundational data structures for statistical computing,
//! optimized for numerical operations and statistical modeling.

mod builder;
mod dataframe;
mod index;
mod ops;
mod series;
mod traits;
mod utils;
mod view;

#[cfg(test)]
mod tests;

// Re-exports
pub use builder::{DataFrameBuilder, SeriesBuilder};
pub use dataframe::{DataFrame, Row};
pub use index::{Index, RangeIndex};
pub use ops::{Arrange, Filter, Mutate, Select};
pub use series::{Series, SeriesStats, SeriesValue};
pub use traits::{AsMatrix, DataView, SeriesTrait};
pub use utils::{ColumnType, Dimensions, Shape};
pub use view::{
    DataFrameColumnIter, DataFrameRowIter, DataFrameView, FilteredDataFrameView, FilteredRowIter,
    RowView, SeriesView, SeriesViewIter, SeriesViewMut,
};

// Type aliases for common use cases
pub type FloatArray = ndarray::Array1<f64>;
pub type IntArray = ndarray::Array1<i64>;
pub type BoolArray = ndarray::Array1<bool>;
pub type StringArray = Vec<String>;
pub type Matrix = ndarray::Array2<f64>;
pub type FloatMatrix = ndarray::Array2<f64>;
pub type IntMatrix = ndarray::Array2<i64>;

/// Error types specific to data operations
#[derive(thiserror::Error, Debug)]
pub enum DataError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("Column '{0}' not found")]
    ColumnNotFound(String),

    #[error("Index out of bounds: index {index}, length {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("Invalid column type: expected {expected}, got {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },

    #[error("Duplicate column name: {0}")]
    DuplicateColumn(String),

    #[error("Operation requires numeric data, got {0}")]
    NonNumericData(&'static str),

    #[error("Missing data in column: {0}")]
    MissingData(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Result type for data operations
pub type Result<T> = std::result::Result<T, DataError>;
