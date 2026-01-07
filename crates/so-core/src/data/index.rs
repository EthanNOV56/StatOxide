//! Index structures for DataFrames

use std::fmt;

/// Index trait for DataFrame indexing
pub trait IndexTrait: Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Option<IndexValue>;
    fn slice(&self, range: std::ops::Range<usize>) -> Box<dyn IndexTrait>;
}

/// Index value enum
#[derive(Debug, Clone, PartialEq)]
pub enum IndexValue {
    Integer(usize),
    String(String),
    Float(f64),
}

impl fmt::Display for IndexValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexValue::Integer(v) => write!(f, "{}", v),
            IndexValue::String(v) => write!(f, "{}", v),
            IndexValue::Float(v) => write!(f, "{}", v),
        }
    }
}

/// Default range index
#[derive(Debug, Clone)]
pub struct RangeIndex {
    start: usize,
    end: usize,
    step: usize,
}

impl RangeIndex {
    /// Create a new range index
    pub fn new(start: usize, end: usize, step: usize) -> Self {
        Self { start, end, step }
    }

    /// Create a range index from 0 to n
    pub fn from_len(n: usize) -> Self {
        Self {
            start: 0,
            end: n,
            step: 1,
        }
    }
}

impl IndexTrait for RangeIndex {
    fn len(&self) -> usize {
        ((self.end - self.start) as f64 / self.step as f64).ceil() as usize
    }

    fn get(&self, idx: usize) -> Option<IndexValue> {
        let pos = self.start + idx * self.step;
        if pos < self.end {
            Some(IndexValue::Integer(pos))
        } else {
            None
        }
    }

    fn slice(&self, range: std::ops::Range<usize>) -> Box<dyn IndexTrait> {
        let start = self.start + range.start * self.step;
        let end = (self.start + range.end * self.step).min(self.end);
        Box::new(RangeIndex::new(start, end, self.step))
    }
}

/// Generic index that can hold any type
#[derive(Debug, Clone)]
pub struct GenericIndex {
    values: Vec<IndexValue>,
}

impl GenericIndex {
    /// Create a new generic index from values
    pub fn new(values: Vec<IndexValue>) -> Self {
        Self { values }
    }

    /// Create from strings
    pub fn from_strings(strings: Vec<String>) -> Self {
        Self {
            values: strings.into_iter().map(IndexValue::String).collect(),
        }
    }

    /// Create from integers
    pub fn from_integers(ints: Vec<usize>) -> Self {
        Self {
            values: ints.into_iter().map(IndexValue::Integer).collect(),
        }
    }

    /// Create from floats
    pub fn from_floats(floats: Vec<f64>) -> Self {
        Self {
            values: floats.into_iter().map(IndexValue::Float).collect(),
        }
    }
}

impl IndexTrait for GenericIndex {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn get(&self, idx: usize) -> Option<IndexValue> {
        self.values.get(idx).cloned()
    }

    fn slice(&self, range: std::ops::Range<usize>) -> Box<dyn IndexTrait> {
        let values = self.values[range].to_vec();
        Box::new(GenericIndex::new(values))
    }
}

/// Enum for different index types
#[derive(Debug, Clone)]
pub enum Index {
    Range(RangeIndex),
    Generic(GenericIndex),
}

impl Index {
    /// Create a range index
    pub fn range(start: usize, end: usize, step: usize) -> Self {
        Self::Range(RangeIndex::new(start, end, step))
    }

    /// Create a generic index
    pub fn generic(values: Vec<IndexValue>) -> Self {
        Self::Generic(GenericIndex::new(values))
    }

    /// Get the length of the index
    pub fn len(&self) -> usize {
        match self {
            Index::Range(idx) => idx.len(),
            Index::Generic(idx) => idx.len(),
        }
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get value at index
    pub fn get(&self, idx: usize) -> Option<IndexValue> {
        match self {
            Index::Range(rdx) => rdx.get(idx),
            Index::Generic(rdx) => rdx.get(idx),
        }
    }

    /// Slice the index
    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        match self {
            Index::Range(idx) => Index::Range(RangeIndex::new(
                idx.start + range.start * idx.step,
                (idx.start + range.end * idx.step).min(idx.end),
                idx.step,
            )),
            Index::Generic(idx) => {
                let values = match idx.values.get(range) {
                    Some(slice) => slice.to_vec(),
                    None => Vec::new(),
                };
                Index::Generic(GenericIndex::new(values))
            }
        }
    }
}
