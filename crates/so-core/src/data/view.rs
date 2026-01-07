//! Zero-copy views for DataFrames and Series
//!
//! Views allow read-only access to data without copying, which is essential
//! for high-performance statistical operations.

use super::*;
use ndarray::{Array2, ArrayView1, ArrayViewMut1};
use std::ops::Range;

/// A read-only view into a Series
#[derive(Debug, Clone)]
pub enum SeriesView<'a> {
    Float(ArrayView1<'a, f64>),
    Int(ArrayView1<'a, i64>),
    Bool(ArrayView1<'a, bool>),
    String(&'a [String]),
    Categorical(ArrayView1<'a, u32>, &'a [String]), // encoded values, categories
}

impl<'a> SeriesView<'a> {
    /// Create a view from a Series
    pub fn from_series(series: &'a Series) -> SeriesView<'a> {
        match series {
            Series::Float(arr) => SeriesView::Float(arr.view()),
            Series::Int(arr) => SeriesView::Int(arr.view()),
            Series::Bool(arr) => SeriesView::Bool(arr.view()),
            Series::String(vec) => SeriesView::String(vec.as_slice()),
            Series::Categorical(arr, cats) => SeriesView::Categorical(arr.view(), cats.as_slice()),
        }
    }

    /// Get the length of the view
    pub fn len(&self) -> usize {
        match self {
            SeriesView::Float(arr) => arr.len(),
            SeriesView::Int(arr) => arr.len(),
            SeriesView::Bool(arr) => arr.len(),
            SeriesView::String(arr) => arr.len(),
            SeriesView::Categorical(arr, _) => arr.len(),
        }
    }

    /// Check if the view is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a value at index
    pub fn get(&self, idx: usize) -> Option<SeriesValue> {
        if idx >= self.len() {
            return None;
        }

        match self {
            SeriesView::Float(arr) => arr.get(idx).map(|&v| SeriesValue::Float(v)),
            SeriesView::Int(arr) => arr.get(idx).map(|&v| SeriesValue::Int(v)),
            SeriesView::Bool(arr) => arr.get(idx).map(|&v| SeriesValue::Bool(v)),
            SeriesView::String(arr) => arr.get(idx).map(|v| SeriesValue::String(v.clone())),
            SeriesView::Categorical(arr, cats) => arr
                .get(idx)
                .and_then(|&code| cats.get(code as usize))
                .map(|cat| SeriesValue::String(cat.clone())),
        }
    }

    /// Get a slice of the view
    pub fn slice(&'a self, range: Range<usize>) -> Option<SeriesView<'a>> {
        if range.start >= self.len() || range.end > self.len() || range.start > range.end {
            return None;
        }

        match self {
            SeriesView::Float(arr) => Some(SeriesView::Float(
                arr.slice(ndarray::s![range.start..range.end]),
            )),
            SeriesView::Int(arr) => Some(SeriesView::Int(
                arr.slice(ndarray::s![range.start..range.end]),
            )),
            SeriesView::Bool(arr) => Some(SeriesView::Bool(
                arr.slice(ndarray::s![range.start..range.end]),
            )),
            SeriesView::String(arr) => Some(SeriesView::String(&arr[range])),
            SeriesView::Categorical(arr, cats) => Some(SeriesView::Categorical(
                arr.slice(ndarray::s![range.start..range.end]),
                cats,
            )),
        }
    }

    /// Convert to float view if possible
    pub fn as_float_view(&self) -> Option<ArrayView1<'a, f64>> {
        match self {
            SeriesView::Float(arr) => Some(*arr),
            _ => None,
        }
    }

    /// Convert to integer view if possible
    pub fn as_int_view(&self) -> Option<ArrayView1<'a, i64>> {
        match self {
            SeriesView::Int(arr) => Some(*arr),
            _ => None,
        }
    }

    /// Convert to boolean view if possible
    pub fn as_bool_view(&self) -> Option<ArrayView1<'a, bool>> {
        match self {
            SeriesView::Bool(arr) => Some(*arr),
            _ => None,
        }
    }

    /// Get iterator over values
    pub fn iter(&'a self) -> SeriesViewIter<'a> {
        SeriesViewIter {
            view: self,
            index: 0,
        }
    }

    /// Compute statistics on the view
    pub fn stats(&self) -> Result<SeriesStats> {
        match self {
            SeriesView::Float(arr) => {
                if arr.is_empty() {
                    return Ok(SeriesStats::empty());
                }

                let mean = arr.mean().unwrap_or(f64::NAN);
                let std = arr.std(1.0);
                let min = arr.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = arr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let q25 = quantile(arr, 0.25).unwrap_or(f64::NAN);
                let q50 = quantile(arr, 0.5).unwrap_or(f64::NAN);
                let q75 = quantile(arr, 0.75).unwrap_or(f64::NAN);

                Ok(SeriesStats {
                    count: arr.len(),
                    mean,
                    std,
                    min,
                    q25,
                    q50,
                    q75,
                    max,
                    unique_count: None,
                })
            }
            _ => Err(DataError::NonNumericData(self.dtype())),
        }
    }

    /// Get data type
    pub fn dtype(&self) -> &'static str {
        match self {
            SeriesView::Float(_) => "float64",
            SeriesView::Int(_) => "int64",
            SeriesView::Bool(_) => "bool",
            SeriesView::String(_) => "string",
            SeriesView::Categorical(_, _) => "categorical",
        }
    }
}

/// Iterator over SeriesView values
pub struct SeriesViewIter<'a> {
    view: &'a SeriesView<'a>,
    index: usize,
}

impl<'a> Iterator for SeriesViewIter<'a> {
    type Item = SeriesValue;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.view.len() {
            let value = self.view.get(self.index);
            self.index += 1;
            value
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.len() - self.index;
        (remaining, Some(remaining))
    }
}

/// A read-only view into a DataFrame
#[derive(Debug, Clone)]
pub struct DataFrameView<'a> {
    columns: Vec<(&'a str, SeriesView<'a>)>,
    nrows: usize,
    index: Option<Index>,
}

impl<'a> DataFrameView<'a> {
    /// Create a view from a DataFrame
    pub fn from_dataframe(df: &'a DataFrame) -> Self {
        let columns: Vec<(&'a str, SeriesView<'a>)> = df
            .columns
            .iter()
            .map(|(name, series)| (name.as_str(), SeriesView::from_series(series)))
            .collect();

        Self {
            columns,
            nrows: df.nrows,
            index: df.index.clone(),
        }
    }

    /// Get the shape of the view
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.columns.len())
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<&'a str> {
        self.columns.iter().map(|(name, _)| *name).collect()
    }

    /// Get a column view
    pub fn column(&self, name: &str) -> Option<SeriesView<'a>> {
        self.columns
            .iter()
            .find(|(col_name, _)| *col_name == name)
            .map(|(_, view)| view.clone())
    }

    /// Get a row as a view
    pub fn row(&'a self, idx: usize) -> Option<RowView<'a>> {
        if idx >= self.nrows {
            return None;
        }

        Some(RowView {
            df_view: self,
            row_idx: idx,
        })
    }

    /// Get iterator over rows
    pub fn rows(&'a self) -> DataFrameRowIter<'a> {
        DataFrameRowIter {
            df_view: self,
            current: 0,
        }
    }

    /// Get iterator over columns
    pub fn columns(&'a self) -> DataFrameColumnIter<'a> {
        DataFrameColumnIter {
            df_view: self,
            current: 0,
        }
    }

    /// Get numeric columns as a matrix view
    pub fn numeric_matrix(&self) -> Result<Array2<f64>> {
        let numeric_cols: Vec<ArrayView1<'a, f64>> = self
            .columns
            .iter()
            .filter_map(|(_, view)| view.as_float_view())
            .collect();

        if numeric_cols.is_empty() {
            return Array2::from_shape_vec((self.nrows, 0), vec![0.0]).map_err(|e| {
                DataError::DimensionMismatch {
                    expected: "compatible dimensions".to_string(),
                    actual: e.to_string(),
                }
            });
        }

        // Stack columns into a matrix
        let arrays: Vec<ArrayView1<f64>> = numeric_cols.into_iter().collect();
        let matrix = ndarray::stack(ndarray::Axis(1), &arrays).map_err(|e| {
            DataError::DimensionMismatch {
                expected: "compatible dimensions".to_string(),
                actual: e.to_string(),
            }
        })?;

        Ok(matrix)
    }

    /// Select columns by name
    pub fn select<'b, I>(&'a self, names: I) -> Result<Self>
    where
        I: IntoIterator<Item = &'b str>,
        'a: 'b,
        'b: 'a,
    {
        let mut selected_columns = Vec::new();

        for name in names.into_iter() {
            if let Some(view) = self.column(name) {
                selected_columns.push((name, view));
            } else {
                return Err(DataError::ColumnNotFound(name.to_string()));
            }
        }

        Ok(Self {
            columns: selected_columns,
            nrows: self.nrows,
            index: self.index.clone(),
        })
    }

    /// Filter rows with a boolean mask
    pub fn filter(&'a self, mask: &[bool]) -> Result<FilteredDataFrameView<'a>> {
        if mask.len() != self.nrows {
            return Err(DataError::DimensionMismatch {
                expected: format!("mask length {}", self.nrows),
                actual: format!("mask length {}", mask.len()),
            });
        }

        Ok(FilteredDataFrameView {
            base: self,
            mask: mask.to_vec(),
        })
    }

    /// Get a slice of the view
    pub fn slice(&'a self, range: Range<usize>) -> Option<DataFrameView<'a>> {
        if range.start >= self.nrows || range.end > self.nrows || range.start > range.end {
            return None;
        }

        let sliced_columns: Vec<(&'a str, SeriesView<'a>)> = self
            .columns
            .iter()
            .filter_map(|(name, view)| {
                view.slice(range.clone())
                    .map(|sliced_view| (*name, sliced_view))
            })
            .collect();

        if sliced_columns.len() != self.columns.len() {
            return None;
        }

        Some(Self {
            columns: sliced_columns,
            nrows: range.end - range.start,
            index: self.index.clone().map(|idx| idx.slice(range)),
        })
    }
}

/// A filtered view of a DataFrame
#[derive(Debug, Clone)]
pub struct FilteredDataFrameView<'a> {
    pub(crate) base: &'a DataFrameView<'a>,
    pub(crate) mask: Vec<bool>,
}

impl<'a> FilteredDataFrameView<'a> {
    /// Get the shape of the filtered view
    pub fn shape(&self) -> (usize, usize) {
        let filtered_rows = self.mask.iter().filter(|&&b| b).count();
        (filtered_rows, self.base.columns.len())
    }

    /// Get iterator over filtered rows
    pub fn rows(&'a self) -> FilteredRowIter<'a> {
        FilteredRowIter {
            base: self.base,
            mask: &self.mask,
            current: 0,
        }
    }
}

/// Iterator over DataFrame rows
pub struct DataFrameRowIter<'a> {
    df_view: &'a DataFrameView<'a>,
    current: usize,
}

impl<'a> Iterator for DataFrameRowIter<'a> {
    type Item = RowView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.df_view.nrows {
            let row = RowView {
                df_view: self.df_view,
                row_idx: self.current,
            };
            self.current += 1;
            Some(row)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.df_view.nrows - self.current;
        (remaining, Some(remaining))
    }
}

/// Iterator over DataFrame columns
pub struct DataFrameColumnIter<'a> {
    df_view: &'a DataFrameView<'a>,
    current: usize,
}

impl<'a> Iterator for DataFrameColumnIter<'a> {
    type Item = (&'a str, SeriesView<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.df_view.columns.len() {
            let (name, view) = &self.df_view.columns[self.current];
            self.current += 1;
            Some((*name, view.clone()))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.df_view.columns.len() - self.current;
        (remaining, Some(remaining))
    }
}

/// Iterator over filtered rows
pub struct FilteredRowIter<'a> {
    base: &'a DataFrameView<'a>,
    mask: &'a [bool],
    current: usize,
}

impl<'a> FilteredRowIter<'a> {
    fn find_next_true(&self, start: usize) -> Option<usize> {
        (start..self.mask.len()).find(|&i| self.mask[i])
    }
}

impl<'a> Iterator for FilteredRowIter<'a> {
    type Item = RowView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.find_next_true(self.current) {
            self.current = idx + 1;
            Some(RowView {
                df_view: self.base,
                row_idx: idx,
            })
        } else {
            None
        }
    }
}

/// A view of a single row in a DataFrame
#[derive(Debug, Clone)]
pub struct RowView<'a> {
    df_view: &'a DataFrameView<'a>,
    row_idx: usize,
}

impl<'a> RowView<'a> {
    /// Get the row index
    pub fn index(&self) -> usize {
        self.row_idx
    }

    /// Get a value from the row
    pub fn get(&self, col: &str) -> Option<SeriesValue> {
        self.df_view
            .column(col)
            .and_then(|view| view.get(self.row_idx))
    }

    /// Get value as float
    pub fn get_float(&self, col: &str) -> Option<f64> {
        match self.get(col)? {
            SeriesValue::Float(v) => Some(v),
            SeriesValue::Int(v) => Some(v as f64),
            SeriesValue::Bool(v) => Some(if v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Get value as integer
    pub fn get_int(&self, col: &str) -> Option<i64> {
        match self.get(col)? {
            SeriesValue::Int(v) => Some(v),
            SeriesValue::Float(v) => Some(v as i64),
            SeriesValue::Bool(v) => Some(if v { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Get value as boolean
    pub fn get_bool(&self, col: &str) -> Option<bool> {
        match self.get(col)? {
            SeriesValue::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// Get value as string
    pub fn get_string(&self, col: &str) -> Option<String> {
        match self.get(col)? {
            SeriesValue::String(v) => Some(v),
            _ => None,
        }
    }

    /// Convert row to a vector of values
    pub fn to_vec(&self) -> Vec<Option<SeriesValue>> {
        self.df_view
            .columns
            .iter()
            .map(|(_, view)| view.get(self.row_idx))
            .collect()
    }
}

/// Mutable view of a Series (for in-place operations)
#[derive(Debug)]
pub enum SeriesViewMut<'a> {
    Float(ArrayViewMut1<'a, f64>),
    Int(ArrayViewMut1<'a, i64>),
    Bool(ArrayViewMut1<'a, bool>),
    String(&'a mut [String]),
    Categorical(ArrayViewMut1<'a, u32>, &'a mut [String]),
}

impl<'a> SeriesViewMut<'a> {
    /// Create a mutable view from a mutable Series reference
    pub fn from_series_mut(series: &'a mut Series) -> Self {
        match series {
            Series::Float(arr) => SeriesViewMut::Float(arr.view_mut()),
            Series::Int(arr) => SeriesViewMut::Int(arr.view_mut()),
            Series::Bool(arr) => SeriesViewMut::Bool(arr.view_mut()),
            Series::String(vec) => SeriesViewMut::String(vec.as_mut_slice()),
            Series::Categorical(arr, cats) => {
                SeriesViewMut::Categorical(arr.view_mut(), cats.as_mut_slice())
            }
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        match self {
            SeriesViewMut::Float(arr) => arr.len(),
            SeriesViewMut::Int(arr) => arr.len(),
            SeriesViewMut::Bool(arr) => arr.len(),
            SeriesViewMut::String(arr) => arr.len(),
            SeriesViewMut::Categorical(arr, _) => arr.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Set value at index
    pub fn set(&mut self, idx: usize, value: SeriesValue) -> Result<()> {
        if idx >= self.len() {
            return Err(DataError::IndexOutOfBounds {
                index: idx,
                len: self.len(),
            });
        }

        let dty = self.dtype();

        match (self, value) {
            (SeriesViewMut::Float(arr), SeriesValue::Float(v)) => {
                arr[idx] = v;
                Ok(())
            }
            (SeriesViewMut::Int(arr), SeriesValue::Int(v)) => {
                arr[idx] = v;
                Ok(())
            }
            (SeriesViewMut::Bool(arr), SeriesValue::Bool(v)) => {
                arr[idx] = v;
                Ok(())
            }
            (SeriesViewMut::String(arr), SeriesValue::String(v)) => {
                if idx < arr.len() {
                    arr[idx] = v;
                    Ok(())
                } else {
                    Err(DataError::IndexOutOfBounds {
                        index: idx,
                        len: arr.len(),
                    })
                }
            }
            (SeriesViewMut::Categorical(arr, cats), SeriesValue::String(v)) => {
                if let Some(code) = cats.iter().position(|cat| cat == &v) {
                    arr[idx] = code as u32;
                    Ok(())
                } else {
                    Err(DataError::InvalidParameter(format!(
                        "Category {} not found",
                        v
                    )))
                }
            }
            _ => Err(DataError::TypeMismatch {
                expected: dty,
                actual: "different type",
            }),
        }
    }

    /// Get data type
    pub fn dtype(&self) -> &'static str {
        match self {
            SeriesViewMut::Float(_) => "float64",
            SeriesViewMut::Int(_) => "int64",
            SeriesViewMut::Bool(_) => "bool",
            SeriesViewMut::String(_) => "string",
            SeriesViewMut::Categorical(_, _) => "categorical",
        }
    }
}

/// Helper function for quantile calculation
fn quantile<'a>(arr: &ArrayView1<'a, f64>, q: f64) -> Option<f64> {
    if arr.is_empty() {
        return None;
    }

    let mut sorted: Vec<f64> = arr.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let index = (n as f64 - 1.0) * q;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        Some(sorted[lower])
    } else {
        let weight = index - lower as f64;
        Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
    }
}
