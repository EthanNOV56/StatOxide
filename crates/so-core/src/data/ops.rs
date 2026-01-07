//! Data operations for DataFrames

use super::*;

/// Filter operation
pub struct Filter<'a> {
    df: &'a DataFrame,
    mask: Vec<bool>,
}

impl<'a> Filter<'a> {
    /// Create a new filter operation
    pub fn new(df: &'a DataFrame) -> Self {
        Self {
            df,
            mask: vec![true; df.nrows()],
        }
    }

    /// Filter by column value
    pub fn by_column<F>(mut self, col: &str, predicate: F) -> Result<Self>
    where
        F: Fn(Option<SeriesValue>) -> bool,
    {
        for i in 0..self.df.nrows() {
            let value = self.df.get_column(col).and_then(|s| s.get(i));

            if !predicate(value) {
                self.mask[i] = false;
            }
        }
        Ok(self)
    }

    /// Filter by predicate on row
    pub fn by_row<F>(mut self, predicate: F) -> Result<Self>
    where
        F: Fn(&Row) -> bool,
    {
        for (i, row) in self.df.rows().enumerate() {
            if !predicate(&row) {
                self.mask[i] = false;
            }
        }
        Ok(self)
    }

    /// Execute the filter
    pub fn execute(self) -> Result<DataFrame> {
        self.df.filter(&self.mask)
    }
}

/// Select operation
pub struct Select<'a> {
    df: &'a DataFrame,
    columns: Vec<String>,
}

impl<'a> Select<'a> {
    /// Create a new select operation
    pub fn new(df: &'a DataFrame) -> Self {
        Self {
            df,
            columns: Vec::new(),
        }
    }

    /// Add columns to select
    pub fn columns<S: AsRef<str>>(mut self, cols: &[S]) -> Self {
        self.columns
            .extend(cols.iter().map(|s| s.as_ref().to_string()));
        self
    }

    /// Add a column to select
    pub fn column<S: AsRef<str>>(mut self, col: S) -> Self {
        self.columns.push(col.as_ref().to_string());
        self
    }

    /// Execute the select
    pub fn execute(self) -> Result<DataFrame> {
        self.df.select(self.columns)
    }
}

/// Mutate operation
pub struct Mutate<'a> {
    df: &'a mut DataFrame,
    mutations: Vec<(String, Box<dyn Fn(&Series) -> Result<Series> + 'a>)>,
}

impl<'a> Mutate<'a> {
    /// Create a new mutate operation
    pub fn new(df: &'a mut DataFrame) -> Self {
        Self {
            df: df,
            mutations: Vec::new(),
        }
    }

    /// Add a mutation
    pub fn mutate<S, F>(mut self, col: S, mutator: F) -> Self
    where
        S: Into<String>,
        F: Fn(&Series) -> Result<Series> + 'a,
    {
        self.mutations.push((col.into(), Box::new(mutator)));
        self
    }

    /// Execute the mutations
    pub fn execute(mut self) -> Result<DataFrame> {
        for (col, mutator) in self.mutations {
            self.df = self.df.mutate(col, |s| mutator(s))?;
        }
        Ok(self.df.clone())
    }
}

/// Arrange (sort) operation
pub struct Arrange<'a> {
    df: &'a DataFrame,
    columns: Vec<(String, bool)>, // (column_name, ascending)
}

impl<'a> Arrange<'a> {
    /// Create a new arrange operation
    pub fn new(df: &'a DataFrame) -> Self {
        Self {
            df,
            columns: Vec::new(),
        }
    }

    /// Add a column to sort by
    pub fn by<S: Into<String>>(mut self, col: S, ascending: bool) -> Self {
        self.columns.push((col.into(), ascending));
        self
    }

    /// Execute the arrange
    pub fn execute(self) -> Result<DataFrame> {
        if self.columns.is_empty() {
            return Ok(self.df.clone());
        }

        // Get sort indices
        let mut indices: Vec<usize> = (0..self.df.nrows()).collect();

        indices.sort_by(|&a, &b| {
            for (col, ascending) in &self.columns {
                let val_a = self.df.get_column(col).and_then(|s| s.get(a));
                let val_b = self.df.get_column(col).and_then(|s| s.get(b));

                match (val_a, val_b) {
                    (Some(v1), Some(v2)) => {
                        let cmp = compare_values(&v1, &v2);
                        if cmp != std::cmp::Ordering::Equal {
                            return if *ascending { cmp } else { cmp.reverse() };
                        }
                    }
                    (None, Some(_)) => return std::cmp::Ordering::Less,
                    (Some(_), None) => return std::cmp::Ordering::Greater,
                    (None, None) => continue,
                }
            }
            std::cmp::Ordering::Equal
        });

        self.df.reorder_rows(&indices)
    }
}

fn compare_values(v1: &SeriesValue, v2: &SeriesValue) -> std::cmp::Ordering {
    match (v1, v2) {
        (SeriesValue::Float(f1), SeriesValue::Float(f2)) => {
            f1.partial_cmp(f2).unwrap_or(std::cmp::Ordering::Equal)
        }
        (SeriesValue::Int(i1), SeriesValue::Int(i2)) => i1.cmp(i2),
        (SeriesValue::Bool(b1), SeriesValue::Bool(b2)) => b1.cmp(b2),
        (SeriesValue::String(s1), SeriesValue::String(s2)) => s1.cmp(s2),
        _ => std::cmp::Ordering::Equal, // Different types are considered equal
    }
}
