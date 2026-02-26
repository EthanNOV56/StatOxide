//! Series data structure for holding homogeneous data
//!
//! A Series is a one-dimensional array that can hold data of a specific type.
//! It's the building block of DataFrames.

use super::*;

use ndarray::Array1;

/// A Series is a typed, one-dimensional array of data
#[derive(Clone, Debug, PartialEq)]
pub enum Series {
    /// Floating point numbers (f64)
    Float(FloatArray),
    /// Integer numbers (i64)
    Int(IntArray),
    /// Boolean values
    Bool(BoolArray),
    /// String values
    String(StringArray),
    /// Categorical data (encoded as u32)
    Categorical(Array1<u32>, Vec<String>), // values, categories
}

impl Series {
    /// Create a new Float series
    pub fn float(data: impl Into<FloatArray>) -> Self {
        Series::Float(data.into())
    }

    /// Create a new Int series
    pub fn int(data: impl Into<IntArray>) -> Self {
        Series::Int(data.into())
    }

    /// Create a new Bool series
    pub fn bool(data: impl Into<BoolArray>) -> Self {
        Series::Bool(data.into())
    }

    /// Create a new String series
    pub fn string(data: impl Into<StringArray>) -> Self {
        Series::String(data.into())
    }

    /// Create a new Categorical series
    pub fn categorical<T: AsRef<str>>(data: &[T]) -> Self {
        let categories: Vec<String> = data
            .iter()
            .map(|s| s.as_ref().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Sort categories for consistent encoding
        let mut categories = categories;
        categories.sort();

        // Create mapping from category to code
        let category_map: std::collections::HashMap<String, u32> = categories
            .iter()
            .enumerate()
            .map(|(i, cat)| (cat.clone(), i as u32))
            .collect();

        // Encode data
        let encoded: Array1<u32> = data
            .iter()
            .map(|s| *category_map.get(s.as_ref()).unwrap())
            .collect();

        Series::Categorical(encoded, categories)
    }

    /// Get the length of the series
    pub fn len(&self) -> usize {
        match self {
            Series::Float(arr) => arr.len(),
            Series::Int(arr) => arr.len(),
            Series::Bool(arr) => arr.len(),
            Series::String(arr) => arr.len(),
            Series::Categorical(arr, _) => arr.len(),
        }
    }

    /// Check if the series is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the type name of the series
    pub fn dtype(&self) -> &'static str {
        match self {
            Series::Float(_) => "float64",
            Series::Int(_) => "int64",
            Series::Bool(_) => "bool",
            Series::String(_) => "string",
            Series::Categorical(_, _) => "categorical",
        }
    }

    /// Get a value at index
    pub fn get(&self, idx: usize) -> Option<SeriesValue> {
        if idx >= self.len() {
            return None;
        }

        match self {
            Series::Float(arr) => arr.get(idx).map(|&v| SeriesValue::Float(v)),
            Series::Int(arr) => arr.get(idx).map(|&v| SeriesValue::Int(v)),
            Series::Bool(arr) => arr.get(idx).map(|&v| SeriesValue::Bool(v)),
            Series::String(arr) => arr.get(idx).map(|v| SeriesValue::String(v.clone())),
            Series::Categorical(arr, cats) => arr
                .get(idx)
                .and_then(|&code| cats.get(code as usize))
                .map(|cat| SeriesValue::String(cat.clone())),
        }
    }

    /// Get a slice of the series
    pub fn slice(&self, range: std::ops::Range<usize>) -> Result<Self> {
        if range.end > self.len() {
            return Err(DataError::IndexOutOfBounds {
                index: range.end,
                len: self.len(),
            });
        }

        match self {
            Series::Float(arr) => Ok(Series::Float(
                arr.slice(ndarray::s![range.start..range.end]).to_owned(),
            )),
            Series::Int(arr) => Ok(Series::Int(
                arr.slice(ndarray::s![range.start..range.end]).to_owned(),
            )),
            Series::Bool(arr) => Ok(Series::Bool(
                arr.slice(ndarray::s![range.start..range.end]).to_owned(),
            )),
            Series::String(arr) => Ok(Series::String(arr[range.start..range.end].to_vec())),
            Series::Categorical(arr, cats) => {
                let sliced = arr.slice(ndarray::s![range.start..range.end]).to_owned();
                Ok(Series::Categorical(sliced, cats.clone()))
            }
        }
    }

    /// Filter the series with a boolean mask
    pub fn filter(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.len() {
            return Err(DataError::DimensionMismatch {
                expected: format!("mask length {}", self.len()),
                actual: format!("mask length {}", mask.len()),
            });
        }

        match self {
            Series::Float(arr) => {
                let filtered: FloatArray = arr
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, keep)| **keep)
                    .map(|(&val, _)| val)
                    .collect();
                Ok(Series::Float(filtered))
            }
            Series::Int(arr) => {
                let filtered: IntArray = arr
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, keep)| **keep)
                    .map(|(&val, _)| val)
                    .collect();
                Ok(Series::Int(filtered))
            }
            Series::Bool(arr) => {
                let filtered: BoolArray = arr
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, keep)| **keep)
                    .map(|(&val, _)| val)
                    .collect();
                Ok(Series::Bool(filtered))
            }
            Series::String(arr) => {
                let filtered: StringArray = arr
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, keep)| **keep)
                    .map(|(val, _)| val.clone())
                    .collect();
                Ok(Series::String(filtered))
            }
            Series::Categorical(arr, cats) => {
                let filtered: Array1<u32> = arr
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, keep)| **keep)
                    .map(|(&val, _)| val)
                    .collect();
                Ok(Series::Categorical(filtered, cats.clone()))
            }
        }
    }

    /// Convert to float series if possible
    pub fn to_float(&self) -> Result<Series> {
        match self {
            Series::Float(arr) => Ok(Series::Float(arr.clone())),
            Series::Int(arr) => {
                let float_arr: FloatArray = arr.iter().map(|&v| v as f64).collect();
                Ok(Series::Float(float_arr))
            }
            Series::Bool(arr) => {
                let float_arr: FloatArray =
                    arr.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
                Ok(Series::Float(float_arr))
            }
            Series::Categorical(arr, _) => {
                let float_arr: FloatArray = arr.iter().map(|&v| v as f64).collect();
                Ok(Series::Float(float_arr))
            }
            Series::String(_) => Err(DataError::NonNumericData("string")),
        }
    }

    /// Compute basic statistics for numeric series
    pub fn describe(&self) -> Result<SeriesStats> {
        match self {
            Series::Float(arr) => {
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
            Series::Int(arr) => {
                // Convert to float for statistics
                let float_arr: FloatArray = arr.iter().map(|&v| v as f64).collect();
                Series::Float(float_arr).describe()
            }
            Series::Bool(arr) => {
                // Convert to float for statistics
                let float_arr: FloatArray =
                    arr.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
                Series::Float(float_arr).describe()
            }
            Series::Categorical(arr, _) => {
                // Convert to float for statistics
                let float_arr: FloatArray = arr.iter().map(|&v| v as f64).collect();
                Series::Float(float_arr).describe()
            }
            Series::String(arr) => {
                // For strings, we can compute count and unique values
                let unique_count = arr.iter().collect::<std::collections::HashSet<_>>().len();
                Ok(SeriesStats {
                    count: arr.len(),
                    unique_count: Some(unique_count),
                    ..SeriesStats::empty()
                })
            }
        }
    }

    /// Compute mean of numeric series
    pub fn mean(&self) -> Result<f64> {
        match self {
            Series::Float(arr) => Ok(arr.mean().unwrap_or(f64::NAN)),
            Series::Int(arr) => Ok(arr.iter().map(|&v| v as f64).sum::<f64>() / arr.len() as f64),
            Series::Bool(arr) => {
                let sum: f64 = arr.iter().map(|&v| if v { 1.0 } else { 0.0 }).sum();
                Ok(sum / arr.len() as f64)
            }
            Series::Categorical(arr, _) => {
                let sum: f64 = arr.iter().map(|&v| v as f64).sum();
                Ok(sum / arr.len() as f64)
            }
            Series::String(_) => Err(DataError::NonNumericData("string")),
        }
    }

    /// Compute standard deviation
    pub fn std(&self, ddof: usize) -> Result<f64> {
        match self {
            Series::Float(arr) => Ok(arr.std(ddof as f64)),
            Series::Int(arr) => {
                let float_arr: FloatArray = arr.iter().map(|&v| v as f64).collect();
                Ok(float_arr.std(ddof as f64))
            }
            Series::Bool(arr) => {
                let float_arr: FloatArray =
                    arr.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
                Ok(float_arr.std(ddof as f64))
            }
            Series::Categorical(arr, _) => {
                let float_arr: FloatArray = arr.iter().map(|&v| v as f64).collect();
                Ok(float_arr.std(ddof as f64))
            }
            Series::String(_) => Err(DataError::NonNumericData("string")),
        }
    }

    /// Compute sum
    pub fn sum(&self) -> Result<f64> {
        match self {
            Series::Float(arr) => Ok(arr.sum()),
            Series::Int(arr) => Ok(arr.iter().map(|&v| v as f64).sum()),
            Series::Bool(arr) => Ok(arr.iter().map(|&v| if v { 1.0 } else { 0.0 }).sum()),
            Series::Categorical(arr, _) => Ok(arr.iter().map(|&v| v as f64).sum()),
            Series::String(_) => Err(DataError::NonNumericData("string")),
        }
    }

    /// Create a read-only view of the Series
    pub fn view(&self) -> SeriesView<'_> {
        SeriesView::from_series(self)
    }

    /// Create a mutable view of the Series
    pub fn view_mut(&mut self) -> SeriesViewMut<'_> {
        SeriesViewMut::from_series_mut(self)
    }

    /// Get a slice view of the Series
    pub fn slice_view(&self, range: std::ops::Range<usize>) -> Option<SeriesView<'_>> {
        if range.end > self.len() {
            return None;
        }

        match self {
            Series::Float(arr) => {
                let slice = arr.slice(ndarray::s![range]);
                Some(SeriesView::Float(slice))
            }
            Series::Int(arr) => {
                let slice = arr.slice(ndarray::s![range]);
                Some(SeriesView::Int(slice))
            }
            Series::Bool(arr) => {
                let slice = arr.slice(ndarray::s![range]);
                Some(SeriesView::Bool(slice))
            }
            Series::String(vec) => Some(SeriesView::String(&vec[range])),
            Series::Categorical(arr, cats) => {
                let slice = arr.slice(ndarray::s![range]);
                Some(SeriesView::Categorical(slice, cats))
            }
        }
    }

    pub fn reorder(&self, indices: &[usize]) -> Result<Self> {
        match self {
            Series::Float(arr) => {
                let values: Vec<f64> = indices
                    .iter()
                    .map(|&idx| if idx < arr.len() { arr[idx] } else { f64::NAN })
                    .collect();
                Ok(Series::Float(ndarray::Array1::from(values)))
            }
            Series::Int(arr) => {
                let values: Vec<i64> = indices
                    .iter()
                    .map(|&idx| if idx < arr.len() { arr[idx] } else { 0 })
                    .collect();
                Ok(Series::Int(ndarray::Array1::from(values)))
            }
            Series::Bool(arr) => {
                let values: Vec<bool> = indices
                    .iter()
                    .map(|&idx| if idx < arr.len() { arr[idx] } else { false })
                    .collect();
                Ok(Series::Bool(ndarray::Array1::from(values)))
            }
            Series::String(vec) => {
                let values: Vec<String> = indices
                    .iter()
                    .map(|&idx| {
                        if idx < vec.len() {
                            vec[idx].clone()
                        } else {
                            String::new()
                        }
                    })
                    .collect();
                Ok(Series::String(values))
            }
            Series::Categorical(arr, cats) => {
                let values: Vec<u32> = indices
                    .iter()
                    .map(|&idx| if idx < arr.len() { arr[idx] } else { 0 })
                    .collect();
                Ok(Series::Categorical(
                    ndarray::Array1::from(values),
                    cats.clone(),
                ))
            }
        }
    }
}

/// Helper function to compute quantile
fn quantile(arr: &FloatArray, q: f64) -> Option<f64> {
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

/// Statistical summary of a series
#[derive(Debug, Clone)]
pub struct SeriesStats {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub q25: f64,
    pub q50: f64,
    pub q75: f64,
    pub max: f64,
    pub unique_count: Option<usize>,
}

impl SeriesStats {
    pub(crate) fn empty() -> Self {
        Self {
            count: 0,
            mean: f64::NAN,
            std: f64::NAN,
            min: f64::NAN,
            q25: f64::NAN,
            q50: f64::NAN,
            q75: f64::NAN,
            max: f64::NAN,
            unique_count: None,
        }
    }
}

/// Enum for type-safe value access
#[derive(Debug, Clone, PartialEq)]
pub enum SeriesValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
}

impl std::fmt::Display for SeriesValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeriesValue::Float(v) => write!(f, "{}", v),
            SeriesValue::Int(v) => write!(f, "{}", v),
            SeriesValue::Bool(v) => write!(f, "{}", v),
            SeriesValue::String(v) => write!(f, "{}", v),
        }
    }
}
