//! Utility types and functions for data operations

use std::fmt;

/// Shape of a DataFrame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl Shape {
    /// Create a new Shape
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    /// Check if shape is empty
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }
}

impl From<(usize, usize)> for Shape {
    fn from((rows, cols): (usize, usize)) -> Self {
        Self { rows, cols }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}×{}", self.rows, self.cols)
    }
}

/// Dimensions of an array
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dimensions {
    pub ndim: usize,
    pub shape: Vec<usize>,
}

impl Dimensions {
    /// Create new Dimensions
    pub fn new(shape: Vec<usize>) -> Self {
        let ndim = shape.len();
        Self { ndim, shape }
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Column type information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Float,
    Int,
    Bool,
    String,
    Categorical,
    Unknown,
}

impl ColumnType {
    /// Check if type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(self, ColumnType::Float | ColumnType::Int | ColumnType::Bool)
    }

    /// Check if type is categorical
    pub fn is_categorical(&self) -> bool {
        matches!(self, ColumnType::Categorical)
    }

    /// Check if type is string
    pub fn is_string(&self) -> bool {
        matches!(self, ColumnType::String)
    }
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnType::Float => write!(f, "float64"),
            ColumnType::Int => write!(f, "int64"),
            ColumnType::Bool => write!(f, "bool"),
            ColumnType::String => write!(f, "string"),
            ColumnType::Categorical => write!(f, "categorical"),
            ColumnType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Utility functions
pub mod utils {
    #[allow(unused_imports)]
    use super::*;
    use crate::data::{DataError, Result};

    /// Check if all slices have the same length
    pub fn check_equal_lengths(slices: &[&[impl Sized]]) -> Result<()> {
        if slices.len() < 2 {
            return Ok(());
        }

        let first_len = slices[0].len();
        for slice in slices.iter().skip(1) {
            if slice.len() != first_len {
                return Err(DataError::DimensionMismatch {
                    expected: format!("length {}", first_len),
                    actual: format!("length {}", slice.len()),
                });
            }
        }

        Ok(())
    }

    /// Convert a slice to a vector of f64
    pub fn to_f64_slice<T>(data: &[T]) -> Vec<f64>
    where
        T: Copy + Into<f64>,
    {
        data.iter().map(|&x| x.into()).collect()
    }

    /// Compute the unique values in a slice
    pub fn unique_values<T: Eq + std::hash::Hash + Clone + Ord>(data: &[T]) -> Vec<T> {
        use std::collections::HashSet;
        let set: HashSet<_> = data.iter().cloned().collect();
        let mut vec: Vec<_> = set.into_iter().collect();
        vec.sort();
        vec
    }

    /// Compute the mode of a slice
    pub fn mode<T: Eq + std::hash::Hash + Clone>(data: &[T]) -> Option<T> {
        use std::collections::HashMap;

        let mut counts = HashMap::new();
        for item in data {
            *counts.entry(item).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(value, _)| value.clone())
    }
}
