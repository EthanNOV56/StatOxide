//! Builder pattern for constructing DataFrames and Series

use indexmap::IndexMap;

use super::*;

/// Builder for creating Series
pub struct SeriesBuilder {
    data: Option<Series>,
    name: Option<String>,
}

impl SeriesBuilder {
    /// Create a new SeriesBuilder
    pub fn new() -> Self {
        Self {
            data: None,
            name: None,
        }
    }

    /// Set the data as floats
    pub fn floats(mut self, data: impl Into<FloatArray>) -> Self {
        self.data = Some(Series::float(data));
        self
    }

    /// Set the data as integers
    pub fn ints(mut self, data: impl Into<IntArray>) -> Self {
        self.data = Some(Series::int(data));
        self
    }

    /// Set the data as booleans
    pub fn bools(mut self, data: impl Into<BoolArray>) -> Self {
        self.data = Some(Series::bool(data));
        self
    }

    /// Set the data as strings
    pub fn strings(mut self, data: impl Into<StringArray>) -> Self {
        self.data = Some(Series::string(data));
        self
    }

    /// Set the data as categorical
    pub fn categorical<T: AsRef<str>>(mut self, data: &[T]) -> Self {
        self.data = Some(Series::categorical(data));
        self
    }

    /// Set the name of the series
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Build the Series
    pub fn build(self) -> Result<Series> {
        self.data.ok_or_else(|| DataError::DimensionMismatch {
            expected: "some data".to_string(),
            actual: "no data".to_string(),
        })
    }
}

/// Builder for creating DataFrames
pub struct DataFrameBuilder {
    columns: IndexMap<String, Series>,
    nrows: Option<usize>,
    index: Option<Index>,
}

impl DataFrameBuilder {
    /// Create a new DataFrameBuilder
    pub fn new() -> Self {
        Self {
            columns: IndexMap::new(),
            nrows: None,
            index: None,
        }
    }

    /// Add a column to the DataFrame
    pub fn with_column<S: Into<String>>(mut self, name: S, series: Series) -> Result<Self> {
        let name = name.into();

        if self.columns.contains_key(&name) {
            return Err(DataError::DuplicateColumn(name));
        }

        // Check dimension consistency
        match self.nrows {
            Some(n) if series.len() != n => {
                return Err(DataError::DimensionMismatch {
                    expected: format!("{} rows", n),
                    actual: format!("{} rows", series.len()),
                });
            }
            None => {
                self.nrows = Some(series.len());
            }
            _ => {}
        }

        self.columns.insert(name, series);
        Ok(self)
    }

    /// Set the index
    pub fn with_index(mut self, index: Index) -> Self {
        if index.len() == self.nrows.unwrap_or(0) {
            self.index = Some(index);
        }
        self
    }

    /// Build the DataFrame
    pub fn build(self) -> Result<DataFrame> {
        let nrows = self.nrows.unwrap_or(0);

        Ok(DataFrame {
            columns: self.columns,
            index: self.index,
            nrows,
            numeric_cache: None,
        })
    }
}
