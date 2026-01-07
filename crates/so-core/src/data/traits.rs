//! Traits for data operations

use super::*;

/// Trait for data structures that can be viewed as tabular data
pub trait DataView: Send + Sync {
    /// Get the shape of the data
    fn shape(&self) -> (usize, usize);

    /// Get column names
    fn column_names(&self) -> Vec<&str>;

    /// Get a column by name
    fn get_column(&self, name: &str) -> Option<&Series>;

    /// Get numeric data as matrix
    fn numeric_matrix(&self) -> Result<Matrix>;

    /// Get descriptive statistics
    fn describe(&self) -> Result<DataFrame>;

    /// Compute correlation matrix
    fn corr(&self) -> Result<Matrix>;

    /// Compute covariance matrix
    fn cov(&self, ddof: usize) -> Result<Matrix>;
}

impl DataView for DataFrame {
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }

    fn column_names(&self) -> Vec<&str> {
        self.column_names()
    }

    fn get_column(&self, name: &str) -> Option<&Series> {
        self.get_column(name)
    }

    fn numeric_matrix(&self) -> Result<Matrix> {
        self.numeric_matrix()
    }

    fn describe(&self) -> Result<DataFrame> {
        self.describe()
    }

    fn corr(&self) -> Result<Matrix> {
        self.corr()
    }

    fn cov(&self, ddof: usize) -> Result<Matrix> {
        self.cov(ddof)
    }
}

/// Trait for series operations
pub trait SeriesTrait: Send + Sync {
    /// Get length of series
    fn len(&self) -> usize;

    /// Check if series is empty
    fn is_empty(&self) -> bool;

    /// Get data type
    fn dtype(&self) -> &'static str;

    /// Get value at index
    fn get(&self, idx: usize) -> Option<SeriesValue>;

    /// Compute mean
    fn mean(&self) -> Result<f64>;

    /// Compute standard deviation
    fn std(&self, ddof: usize) -> Result<f64>;

    /// Compute sum
    fn sum(&self) -> Result<f64>;
}

impl SeriesTrait for Series {
    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn dtype(&self) -> &'static str {
        self.dtype()
    }

    fn get(&self, idx: usize) -> Option<SeriesValue> {
        self.get(idx)
    }

    fn mean(&self) -> Result<f64> {
        self.mean()
    }

    fn std(&self, ddof: usize) -> Result<f64> {
        self.std(ddof)
    }

    fn sum(&self) -> Result<f64> {
        self.sum()
    }
}

/// Trait for converting to matrix
pub trait AsMatrix {
    /// Convert to matrix
    fn as_matrix(&self) -> Result<Matrix>;

    /// Get dimensions
    fn dim(&self) -> (usize, usize);

    /// Check if matrix is empty
    fn is_empty(&self) -> bool;
}

impl AsMatrix for DataFrame {
    fn as_matrix(&self) -> Result<Matrix> {
        self.numeric_matrix()
    }

    fn dim(&self) -> (usize, usize) {
        self.shape()
    }

    fn is_empty(&self) -> bool {
        self.nrows() == 0 || self.ncols() == 0
    }
}
