//! DataFrame implementation for tabular data
//!
//! A DataFrame is a 2-dimensional labeled data structure with columns of
//! potentially different types. It's optimized for statistical operations.

use super::*;

use indexmap::IndexMap;
use ndarray::{Axis, stack};
use rand::seq::SliceRandom;

use std::collections::HashMap;

/// Main DataFrame structure
#[derive(Clone, Debug)]
pub struct DataFrame {
    pub(crate) columns: IndexMap<String, Series>,
    pub(crate) index: Option<Index>,
    pub(crate) nrows: usize,
    // Cache for numeric data matrix
    pub(crate) numeric_cache: Option<Matrix>,
}

impl DataFrame {
    /// Create an empty DataFrame
    pub fn new() -> Self {
        Self {
            columns: IndexMap::new(),
            index: None,
            nrows: 0,
            numeric_cache: None,
        }
    }

    /// Create DataFrame from columns
    pub fn from_columns<I, S>(columns: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S, Series)>,
        S: Into<String>,
    {
        let mut builder = DataFrameBuilder::new();

        for (name, series) in columns.into_iter() {
            builder = builder.with_column(name, series)?;
        }

        builder.build()
    }

    /// Get the shape of the DataFrame (rows, columns)
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.columns.len())
    }

    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(|k| k.as_str()).collect()
    }

    /// Get a reference to a column
    pub fn get_column(&self, name: &str) -> Option<&Series> {
        self.columns.get(name)
    }

    /// Get mutable reference to a column
    pub fn get_column_mut(&mut self, name: &str) -> Option<&mut Series> {
        // Invalidate cache when column is modified
        self.numeric_cache = None;
        self.columns.get_mut(name)
    }

    /// Check if column exists
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Select specific columns
    pub fn select<I, S>(&self, names: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut builder = DataFrameBuilder::new();

        for name in names.into_iter() {
            let name = name.as_ref();
            match self.columns.get(name) {
                Some(series) => {
                    builder = builder.with_column(name, series.clone())?;
                }
                None => {
                    return Err(DataError::ColumnNotFound(name.to_string()));
                }
            }
        }

        builder.build()
    }

    /// Select columns by index positions
    pub fn select_by_index<I>(&self, indices: I) -> Result<Self>
    where
        I: IntoIterator<Item = usize>,
    {
        let names: Vec<&str> = indices
            .into_iter()
            .filter_map(|idx| self.columns.get_index(idx).map(|(name, _)| name.as_str()))
            .collect();

        self.select(names)
    }

    /// Filter rows with a boolean mask
    pub fn filter(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.nrows {
            return Err(DataError::DimensionMismatch {
                expected: format!("mask length {}", self.nrows),
                actual: format!("mask length {}", mask.len()),
            });
        }

        let mut builder = DataFrameBuilder::new();

        for (name, series) in &self.columns {
            let filtered = series.filter(mask)?;
            builder = builder.with_column(name.clone(), filtered)?;
        }

        builder.build()
    }

    /// Filter rows with a predicate
    pub fn filter_with<F>(&self, predicate: F) -> Result<Self>
    where
        F: Fn(&Row) -> bool,
    {
        let mut mask = Vec::with_capacity(self.nrows);

        for i in 0..self.nrows {
            let row = self.get_row(i)?;
            mask.push(predicate(&row));
        }

        self.filter(&mask)
    }

    /// Add a new column
    pub fn with_column<S: Into<String>>(mut self, name: S, series: Series) -> Result<Self> {
        let name = name.into();

        if self.columns.contains_key(&name) {
            return Err(DataError::DuplicateColumn(name));
        }

        if !self.columns.is_empty() && series.len() != self.nrows {
            return Err(DataError::DimensionMismatch {
                expected: format!("{} rows", self.nrows),
                actual: format!("{} rows", series.len()),
            });
        }

        if self.columns.is_empty() {
            self.nrows = series.len();
        }

        self.columns.insert(name, series);
        self.numeric_cache = None; // Invalidate cache

        Ok(self)
    }

    /// Mutate an existing column or add a new one
    pub fn mutate<S, F>(&mut self, name: S, mutator: F) -> Result<&mut Self>
    where
        S: Into<String>,
        F: Fn(&Series) -> Result<Series>,
    {
        let name = name.into();

        if let Some(existing) = self.columns.get(&name) {
            let new_series = mutator(existing)?;
            if new_series.len() != self.nrows {
                return Err(DataError::DimensionMismatch {
                    expected: format!("{} rows", self.nrows),
                    actual: format!("{} rows", new_series.len()),
                });
            }
            self.columns.insert(name, new_series);
        } else {
            // For new columns, we need to handle differently
            return Err(DataError::ColumnNotFound(name));
        }

        self.numeric_cache = None; // Invalidate cache
        Ok(self)
    }

    /// Rename columns
    pub fn rename<S1, S2>(mut self, mapping: &[(S1, S2)]) -> Result<Self>
    where
        S1: AsRef<str>,
        String: for<'a> From<&'a S2>,
    {
        for (old_name, new_name) in mapping {
            let old_name = old_name.as_ref();
            let new_name = new_name.into();

            if let Some(series) = self.columns.shift_remove(old_name) {
                if self.columns.contains_key(&new_name) {
                    return Err(DataError::DuplicateColumn(new_name));
                }
                self.columns.insert(new_name, series);
            } else {
                return Err(DataError::ColumnNotFound(old_name.to_string()));
            }
        }

        Ok(self)
    }

    /// Drop columns
    pub fn drop<S: AsRef<str>>(mut self, names: &[S]) -> Result<Self> {
        for name in names {
            let name = name.as_ref();
            if self.columns.shift_remove(name).is_none() {
                return Err(DataError::ColumnNotFound(name.to_string()));
            }
        }

        if self.columns.is_empty() {
            self.nrows = 0;
        }

        self.numeric_cache = None; // Invalidate cache
        Ok(self)
    }

    /// Get a row as a Row view
    pub fn get_row(&self, idx: usize) -> Result<Row<'_>> {
        if idx >= self.nrows {
            return Err(DataError::IndexOutOfBounds {
                index: idx,
                len: self.nrows,
            });
        }

        Ok(Row {
            df: self,
            row_idx: idx,
        })
    }

    /// Iterate over rows
    pub fn rows(&self) -> RowIter<'_> {
        RowIter {
            df: self,
            current: 0,
        }
    }

    /// Get numeric columns as a matrix
    pub fn numeric_matrix(&self) -> Result<Matrix> {
        // Check cache first
        if let Some(cached) = &self.numeric_cache {
            return Ok(cached.clone());
        }

        let numeric_cols: Vec<&Series> = self
            .columns
            .values()
            .filter(|s| matches!(s, Series::Float(_) | Series::Int(_) | Series::Bool(_)))
            .collect();

        if numeric_cols.is_empty() {
            return Ok(Matrix::zeros((self.nrows, 0)));
        }

        // Convert all numeric columns to float
        let float_cols: Result<Vec<FloatArray>> = numeric_cols
            .iter()
            .map(|&s| match s {
                Series::Float(arr) => Ok(arr.clone()),
                Series::Int(arr) => Ok(arr.iter().map(|&v| v as f64).collect()),
                Series::Bool(arr) => Ok(arr.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect()),
                _ => Err(DataError::NonNumericData("non-numeric")),
            })
            .collect();

        let float_cols = float_cols?;

        // Stack columns into a matrix
        let arrays: Vec<ndarray::ArrayView1<f64>> =
            float_cols.iter().map(|arr| arr.view()).collect();

        let matrix = stack(Axis(1), &arrays).map_err(|e| DataError::DimensionMismatch {
            expected: "compatible dimensions".to_string(),
            actual: e.to_string(),
        })?;

        // Cache the result
        let matrix = matrix.to_owned();
        let mut df = self.clone();
        df.numeric_cache = Some(matrix.clone());

        Ok(matrix)
    }

    /// Get descriptive statistics for all numeric columns
    pub fn describe(&self) -> Result<DataFrame> {
        let mut stats = Vec::new();

        for (name, series) in &self.columns {
            match series.describe() {
                Ok(series_stats) => {
                    // Convert stats to a row
                    let mut row = HashMap::new();
                    row.insert("column".to_string(), name.clone());
                    row.insert("count".to_string(), series_stats.count.to_string());
                    row.insert("mean".to_string(), series_stats.mean.to_string());
                    row.insert("std".to_string(), series_stats.std.to_string());
                    row.insert("min".to_string(), series_stats.min.to_string());
                    row.insert("25%".to_string(), series_stats.q25.to_string());
                    row.insert("50%".to_string(), series_stats.q50.to_string());
                    row.insert("75%".to_string(), series_stats.q75.to_string());
                    row.insert("max".to_string(), series_stats.max.to_string());

                    if let Some(unique_count) = series_stats.unique_count {
                        row.insert("unique".to_string(), unique_count.to_string());
                    }

                    stats.push(row);
                }
                Err(_) => {
                    // For non-numeric columns, just get count
                    let mut row = HashMap::new();
                    row.insert("column".to_string(), name.clone());
                    row.insert("count".to_string(), series.len().to_string());
                    row.insert("dtype".to_string(), series.dtype().to_string());
                    stats.push(row);
                }
            }
        }

        // Convert to DataFrame
        let mut builder = DataFrameBuilder::new();

        // Get all unique keys
        let all_keys: std::collections::HashSet<&String> =
            stats.iter().flat_map(|row| row.keys()).collect();

        for &key in &all_keys {
            let values: Vec<String> = stats
                .iter()
                .map(|row| row.get(key).cloned().unwrap_or_default())
                .collect();

            builder = builder.with_column(key, Series::string(values))?;
        }

        builder.build()
    }

    /// Compute correlation matrix for numeric columns
    pub fn corr(&self) -> Result<Matrix> {
        let matrix = self.numeric_matrix()?;
        let nrows = matrix.nrows() as f64;

        if nrows <= 1.0 {
            return Ok(Matrix::zeros((0, 0)));
        }

        // Center the data
        let means = matrix.mean_axis(Axis(0)).unwrap();
        let centered = &matrix - &means;

        // Compute covariance
        let cov = centered.t().dot(&centered) / (nrows - 1.0);

        // Compute correlation from covariance
        let stds = cov.diag().mapv(|v| v.sqrt());
        let mut corr = Matrix::zeros(cov.dim());

        for i in 0..corr.nrows() {
            for j in 0..corr.ncols() {
                if stds[i] > 0.0 && stds[j] > 0.0 {
                    corr[(i, j)] = cov[(i, j)] / (stds[i] * stds[j]);
                } else {
                    corr[(i, j)] = f64::NAN;
                }
            }
        }

        Ok(corr)
    }

    /// Compute covariance matrix for numeric columns
    pub fn cov(&self, ddof: usize) -> Result<Matrix> {
        let matrix = self.numeric_matrix()?;
        let nrows = matrix.nrows() as f64;

        if nrows <= ddof as f64 {
            return Ok(Matrix::zeros((0, 0)));
        }

        // Center the data
        let means = matrix.mean_axis(Axis(0)).unwrap();
        let centered = &matrix - &means;

        // Compute covariance
        let cov = centered.t().dot(&centered) / (nrows - ddof as f64);

        Ok(cov)
    }

    /// Sample rows
    pub fn sample(&self, n: usize, replace: bool, seed: Option<u64>) -> Result<Self> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        if n > self.nrows && !replace {
            return Err(DataError::DimensionMismatch {
                expected: format!("sample size <= {}", self.nrows),
                actual: format!("sample size {}", n),
            });
        }

        let mut rng = if let Some(seed) = seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_os_rng()
        };

        let indices: Vec<usize> = if replace {
            (0..n).map(|_| rng.random_range(0..self.nrows)).collect()
        } else {
            let mut indices: Vec<usize> = (0..self.nrows).collect();
            indices.shuffle(&mut rng);
            indices.truncate(n);
            indices
        };

        // Create mask
        let mut mask = vec![false; self.nrows];
        for &idx in &indices {
            mask[idx] = true;
        }

        self.filter(&mask)
    }

    /// Create a read-only view of the DataFrame
    pub fn view(&self) -> DataFrameView<'_> {
        DataFrameView::from_dataframe(self)
    }

    /// Get a mutable view of a column
    pub fn get_column_mut_view(&mut self, name: &str) -> Option<SeriesViewMut<'_>> {
        self.numeric_cache = None; // Invalidate cache
        self.columns
            .get_mut(name)
            .and_then(|series| Some(SeriesViewMut::from_series_mut(series)))
    }

    // /// Create a view of numeric columns as a matrix
    // pub fn numeric_matrix_view(&self) -> Result<ArrayView2<f64>> {
    //     self.view().numeric_matrix_view()
    // }

    // /// Create a filtered view
    // pub fn filter_view(&self, mask: &[bool]) -> Result<FilteredDataFrameView> {
    //     if mask.len() != self.nrows {
    //         return Err(DataError::DimensionMismatch {
    //             expected: format!("mask length {}", self.nrows),
    //             actual: format!("mask length {}", mask.len()),
    //         });
    //     }

    //     Ok(FilteredDataFrameView {
    //         base: self,
    //         mask: mask.to_vec(),
    //     })
    // }

    // /// Get a slice view of the DataFrame
    // pub fn slice_view(&self, range: std::ops::Range<usize>) -> Option<DataFrameView> {
    //     self.view().slice(range)
    // }

    pub fn reorder_rows(&self, indices: &[usize]) -> Result<Self> {
        for &idx in indices {
            if idx >= self.nrows {
                return Err(DataError::IndexOutOfBounds {
                    index: idx,
                    len: self.nrows,
                });
            }
        }

        let mut builder = DataFrameBuilder::new();

        for (name, series) in &self.columns {
            let reordered = series.reorder(indices)?;
            builder = builder.with_column(name.clone(), reordered)?;
        }

        builder.build()
    }
}

/// Row view into a DataFrame
pub struct Row<'a> {
    df: &'a DataFrame,
    row_idx: usize,
}

impl<'a> Row<'a> {
    /// Get a value from the row
    pub fn get(&self, col: &str) -> Result<Option<SeriesValue>> {
        match self.df.columns.get(col) {
            Some(series) => Ok(series.get(self.row_idx)),
            None => Err(DataError::ColumnNotFound(col.to_string())),
        }
    }

    /// Get value as float
    pub fn get_float(&self, col: &str) -> Result<Option<f64>> {
        match self.get(col)? {
            Some(SeriesValue::Float(v)) => Ok(Some(v)),
            Some(SeriesValue::Int(v)) => Ok(Some(v as f64)),
            Some(SeriesValue::Bool(v)) => Ok(Some(if v { 1.0 } else { 0.0 })),
            _ => Ok(None),
        }
    }

    /// Get value as integer
    pub fn get_int(&self, col: &str) -> Result<Option<i64>> {
        match self.get(col)? {
            Some(SeriesValue::Int(v)) => Ok(Some(v)),
            Some(SeriesValue::Float(v)) => Ok(Some(v as i64)),
            Some(SeriesValue::Bool(v)) => Ok(Some(if v { 1 } else { 0 })),
            _ => Ok(None),
        }
    }

    /// Get value as boolean
    pub fn get_bool(&self, col: &str) -> Result<Option<bool>> {
        match self.get(col)? {
            Some(SeriesValue::Bool(v)) => Ok(Some(v)),
            _ => Ok(None),
        }
    }

    /// Get value as string
    pub fn get_string(&self, col: &str) -> Result<Option<String>> {
        match self.get(col)? {
            Some(SeriesValue::String(v)) => Ok(Some(v)),
            _ => Ok(None),
        }
    }
}

/// Iterator over rows
pub struct RowIter<'a> {
    df: &'a DataFrame,
    current: usize,
}

impl<'a> Iterator for RowIter<'a> {
    type Item = Row<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.df.nrows {
            let row = Row {
                df: self.df,
                row_idx: self.current,
            };
            self.current += 1;
            Some(row)
        } else {
            None
        }
    }
}

impl std::fmt::Display for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DataFrame({} rows × {} cols)", self.nrows, self.ncols())
    }
}
