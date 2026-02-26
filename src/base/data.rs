//! Data structures for statistical computing
//!
//! This module provides columnar data structures optimized for
//! statistical operations, with interoperability with numpy and pandas.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::tools::formula::Formula;

// ============================================================================
// Series - Vector with metadata
// ============================================================================

/// A Series represents a single column of data with a name and dtype
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Series {
    name: String,
    data: Array1<f64>,
    /// Optional factor/categorical encoding
    levels: Option<Vec<String>>,
}

impl Series {
    /// Create a new numeric series
    pub fn new(name: impl Into<String>, data: Array1<f64>) -> Self {
        Self {
            name: name.into(),
            data,
            levels: None,
        }
    }

    /// Create a factor/categorical series
    pub fn factor(name: impl Into<String>, levels: Vec<String>, indices: Array1<usize>) -> Self {
        Self {
            name: name.into(),
            data: indices.mapv(|i| i as f64),
            levels: Some(levels),
        }
    }

    /// Get series name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get data as array view
    pub fn data(&self) -> ArrayView1<f64> {
        self.data.view()
    }

    /// Get length of series
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if series is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Compute mean
    pub fn mean(&self) -> Option<f64> {
        if self.is_empty() {
            None
        } else {
            Some(self.data.mean().unwrap_or(f64::NAN))
        }
    }

    /// Compute variance
    pub fn var(&self, ddof: f64) -> Option<f64> {
        if self.len() <= 1 {
            None
        } else {
            Some(self.data.var(ddof))
        }
    }

    /// Compute standard deviation
    pub fn std(&self, ddof: f64) -> Option<f64> {
        self.var(ddof).map(|v| v.sqrt())
    }

    /// Compute minimum value
    pub fn min(&self) -> Option<f64> {
        self.data.fold(f64::INFINITY, |a, &b| a.min(b)).into()
    }

    /// Compute maximum value
    pub fn max(&self) -> Option<f64> {
        self.data.fold(-f64::INFINITY, |a, &b| a.max(b)).into()
    }

    /// Compute quantile using R's type 7 method
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.is_empty() || !(0.0..=1.0).contains(&q) {
            return None;
        }

        let mut sorted = self.data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let index = (n - 1) as f64 * q;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            Some(sorted[lower])
        } else {
            let weight = index - lower as f64;
            Some((1.0 - weight) * sorted[lower] + weight * sorted[upper])
        }
    }

    /// Apply a function element-wise
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            name: self.name.clone(),
            data: self.data.mapv(f),
            levels: self.levels.clone(),
        }
    }

    /// Standardize (z-score normalization)
    pub fn standardize(&self) -> Option<Self> {
        let mean = self.mean()?;
        let std = self.std(1.0)?;
        
        if std == 0.0 {
            return None;
        }

        Some(self.map(|x| (x - mean) / std))
    }
}

// ============================================================================
// DataFrame - Collection of Series
// ============================================================================

/// A DataFrame represents a collection of named Series (columns)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFrame {
    columns: HashMap<String, Series>,
    n_rows: usize,
}

impl DataFrame {
    /// Create a new empty DataFrame
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
            n_rows: 0,
        }
    }

    /// Create a DataFrame from a map of column names to Series
    pub fn from_series(columns: HashMap<String, Series>) -> Result<Self, String> {
        let mut n_rows = 0;
        for (name, series) in &columns {
            if n_rows == 0 {
                n_rows = series.len();
            } else if series.len() != n_rows {
                return Err(format!(
                    "Column '{}' has length {}, expected {}",
                    name,
                    series.len(),
                    n_rows
                ));
            }
        }

        Ok(Self { columns, n_rows })
    }

    /// Get number of rows
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Get number of columns
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        self.columns.keys().cloned().collect()
    }

    /// Get a column by name
    pub fn column(&self, name: &str) -> Option<&Series> {
        self.columns.get(name)
    }

    /// Get a mutable reference to a column
    pub fn column_mut(&mut self, name: &str) -> Option<&mut Series> {
        self.columns.get_mut(name)
    }

    /// Add a column to the DataFrame
    pub fn with_column(mut self, series: Series) -> Result<Self, String> {
        let name = series.name().to_string();
        
        if self.n_rows == 0 {
            self.n_rows = series.len();
        } else if series.len() != self.n_rows {
            return Err(format!(
                "Column '{}' has length {}, expected {}",
                name,
                series.len(),
                self.n_rows
            ));
        }

        self.columns.insert(name, series);
        Ok(self)
    }

    /// Remove a column
    pub fn drop_column(mut self, name: &str) -> Self {
        self.columns.remove(name);
        self
    }

    /// Select specific columns
    pub fn select(&self, col_names: &[&str]) -> Result<Self, String> {
        let mut new_columns = HashMap::new();
        for &name in col_names {
            if let Some(series) = self.columns.get(name) {
                new_columns.insert(name.to_string(), series.clone());
            } else {
                return Err(format!("Column '{}' not found", name));
            }
        }
        Self::from_series(new_columns)
    }

    /// Filter rows based on a boolean mask
    pub fn filter(&self, mask: &[bool]) -> Result<Self, String> {
        if mask.len() != self.n_rows {
            return Err(format!(
                "Mask length {} doesn't match DataFrame rows {}",
                mask.len(),
                self.n_rows
            ));
        }

        let mut new_columns = HashMap::new();
        for (name, series) in &self.columns {
            let mut filtered_data = Vec::new();
            for (i, &value) in series.data.iter().enumerate() {
                if mask[i] {
                    filtered_data.push(value);
                }
            }
            new_columns.insert(
                name.clone(),
                Series::new(name.clone(), Array1::from_vec(filtered_data)),
            );
        }

        Self::from_series(new_columns)
    }

    /// Get design matrix for regression
    pub fn design_matrix(&self, formula: &Formula) -> Result<Array2<f64>, String> {
        formula.build_matrix(self)
    }
}

// ============================================================================
// Data loading and conversion utilities
// ============================================================================

/// Trait for converting external data formats to StatOxide structures
pub trait FromData<T> {
    fn from_data(data: T) -> Result<Self, String>
    where
        Self: Sized;
}

/// Trait for exporting StatOxide structures to external formats
pub trait ToData<T> {
    fn to_data(&self) -> Result<T, String>;
}

impl FromData<Vec<Vec<f64>>> for DataFrame {
    fn from_data(data: Vec<Vec<f64>>) -> Result<Self, String> {
        if data.is_empty() {
            return Ok(Self::new());
        }

        let n_rows = data[0].len();
        let mut columns = HashMap::new();

        for (i, column_data) in data.iter().enumerate() {
            if column_data.len() != n_rows {
                return Err(format!(
                    "Column {} has length {}, expected {}",
                    i,
                    column_data.len(),
                    n_rows
                ));
            }

            columns.insert(
                format!("x{}", i),
                Series::new(format!("x{}", i), Array1::from_vec(column_data.clone())),
            );
        }

        Self::from_series(columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_series_basic() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let series = Series::new("test", data);

        assert_eq!(series.name(), "test");
        assert_eq!(series.len(), 5);
        assert_eq!(series.mean(), Some(3.0));
        assert_eq!(series.std(1.0).unwrap(), 1.5811388300841898);
        assert_eq!(series.min(), Some(1.0));
        assert_eq!(series.max(), Some(5.0));
    }

    #[test]
    fn test_dataframe_basic() {
        let mut columns = HashMap::new();
        columns.insert(
            "x".to_string(),
            Series::new("x", arr1(&[1.0, 2.0, 3.0])),
        );
        columns.insert(
            "y".to_string(),
            Series::new("y", arr1(&[4.0, 5.0, 6.0])),
        );

        let df = DataFrame::from_series(columns).unwrap();
        assert_eq!(df.n_rows(), 3);
        assert_eq!(df.n_cols(), 2);
        assert!(df.column("x").is_some());
        assert!(df.column("z").is_none());
    }
}