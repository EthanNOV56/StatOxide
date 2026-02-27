//! Time series data structure with datetime indexing

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use so_core::data::{DataFrame, Series};
use so_core::error::{Error, Result};

/// Frequency of time series data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Frequency {
    /// Annual data
    Yearly,
    /// Quarterly data
    Quarterly,
    /// Monthly data
    Monthly,
    /// Weekly data
    Weekly,
    /// Daily data
    Daily,
    /// Hourly data
    Hourly,
    /// Minutely data
    Minutely,
    /// Secondly data
    Secondly,
    /// Custom frequency (custom steps per year)
    Custom(u32),
    /// Irregular/unspecified frequency
    Irregular,
}

impl Frequency {
    /// Get number of periods per year
    pub fn periods_per_year(&self) -> Option<f64> {
        match self {
            Frequency::Yearly => Some(1.0),
            Frequency::Quarterly => Some(4.0),
            Frequency::Monthly => Some(12.0),
            Frequency::Weekly => Some(52.1775), // Average weeks per year
            Frequency::Daily => Some(365.25),   // Account for leap years
            Frequency::Hourly => Some(365.25 * 24.0),
            Frequency::Minutely => Some(365.25 * 24.0 * 60.0),
            Frequency::Secondly => Some(365.25 * 24.0 * 60.0 * 60.0),
            Frequency::Custom(n) => Some(*n as f64),
            Frequency::Irregular => None,
        }
    }

    /// Get frequency name
    pub fn name(&self) -> &'static str {
        match self {
            Frequency::Yearly => "Yearly",
            Frequency::Quarterly => "Quarterly",
            Frequency::Monthly => "Monthly",
            Frequency::Weekly => "Weekly",
            Frequency::Daily => "Daily",
            Frequency::Hourly => "Hourly",
            Frequency::Minutely => "Minutely",
            Frequency::Secondly => "Secondly",
            Frequency::Custom(_n) => "Custom",
            Frequency::Irregular => "Irregular",
        }
    }
}

/// Time series data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Series name
    pub name: String,
    /// Time index (Unix timestamps in seconds)
    pub timestamps: Vec<i64>,
    /// Values
    pub values: Array1<f64>,
    /// Frequency (if regular)
    pub frequency: Option<Frequency>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl TimeSeries {
    /// Create a new time series from arrays
    pub fn new(
        name: impl Into<String>,
        timestamps: Vec<i64>,
        values: Array1<f64>,
        frequency: Option<Frequency>,
    ) -> Result<Self> {
        if timestamps.len() != values.len() {
            return Err(Error::DimensionMismatch(format!(
                "Timestamps length {} != values length {}",
                timestamps.len(),
                values.len()
            )));
        }

        if values.len() < 2 {
            return Err(Error::DataError(
                "Time series must have at least 2 observations".to_string(),
            ));
        }

        // Check if timestamps are sorted
        for i in 1..timestamps.len() {
            if timestamps[i] <= timestamps[i - 1] {
                return Err(Error::DataError(
                    "Timestamps must be strictly increasing".to_string(),
                ));
            }
        }

        Ok(Self {
            name: name.into(),
            timestamps,
            values,
            frequency,
            metadata: HashMap::new(),
        })
    }

    /// Create from DataFrame with date and value columns
    pub fn from_dataframe(df: &DataFrame, value_col: &str, date_col: &str) -> Result<Self> {
        let value_series = df
            .column(value_col)
            .ok_or_else(|| Error::DataError(format!("Value column '{}' not found", value_col)))?;

        let date_series = df
            .column(date_col)
            .ok_or_else(|| Error::DataError(format!("Date column '{}' not found", date_col)))?;

        // Convert date column to timestamps
        // For now, assume dates are already in timestamp format or can be parsed
        let timestamps: Vec<i64> = date_series.data().iter().map(|&x| x as i64).collect();

        Self::new(
            value_col,
            timestamps,
            value_series.data().to_owned(),
            None, // Auto-detect frequency
        )
    }

    /// Create a regular time series with integer index
    pub fn regular(name: impl Into<String>, values: Array1<f64>, frequency: Frequency) -> Self {
        let n = values.len();
        let timestamps: Vec<i64> = (0..n).map(|i| i as i64).collect();

        Self {
            name: name.into(),
            timestamps,
            values,
            frequency: Some(frequency),
            metadata: HashMap::new(),
        }
    }

    /// Get series length
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if series is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get series name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get values
    pub fn values(&self) -> &Array1<f64> {
        &self.values
    }

    /// Get timestamps
    pub fn timestamps(&self) -> &[i64] {
        &self.timestamps
    }

    /// Get frequency
    pub fn frequency(&self) -> Option<Frequency> {
        self.frequency
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Get start time
    pub fn start_time(&self) -> Option<i64> {
        self.timestamps.first().copied()
    }

    /// Get end time
    pub fn end_time(&self) -> Option<i64> {
        self.timestamps.last().copied()
    }

    /// Compute basic statistics
    pub fn stats(&self) -> TimeSeriesStats {
        let n = self.len() as f64;

        let mean = self.values.mean().unwrap_or(0.0);
        let variance = self.values.var(1.0);
        let std = variance.sqrt();

        // Autocorrelation at lag 1
        let acf1 = if n > 1.0 {
            let mut sum = 0.0;
            for i in 1..self.len() {
                sum += (self.values[i] - mean) * (self.values[i - 1] - mean);
            }
            sum / ((n - 1.0) * variance)
        } else {
            0.0
        };

        TimeSeriesStats {
            n_obs: self.len(),
            mean,
            std,
            variance,
            min: self.values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max: self.values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            acf1,
        }
    }

    /// Differencing: y_t - y_{t-d}
    pub fn diff(&self, lag: usize, order: usize) -> Result<TimeSeries> {
        if lag < 1 {
            return Err(Error::DataError("Lag must be >= 1".to_string()));
        }

        if order < 1 {
            return Ok(self.clone());
        }

        let mut current = self.values.clone();
        let mut timestamps = self.timestamps[lag..].to_vec();

        for _ in 0..order {
            let n = current.len();
            if n <= lag {
                return Err(Error::DataError(
                    "Not enough observations for differencing".to_string(),
                ));
            }

            let diffed: Array1<f64> = (lag..n).map(|i| current[i] - current[i - lag]).collect();

            current = diffed;
            timestamps = timestamps[lag..].to_vec();
        }

        TimeSeries::new(
            format!("{}_diff{}", self.name, order),
            timestamps,
            current,
            self.frequency,
        )
    }

    /// Log transformation (with offset to handle zeros/negatives)
    pub fn log(&self, offset: f64) -> TimeSeries {
        let values = self.values.mapv(|x| (x + offset).ln());

        TimeSeries {
            name: format!("log({})", self.name),
            timestamps: self.timestamps.clone(),
            values,
            frequency: self.frequency,
            metadata: self.metadata.clone(),
        }
    }

    /// Box-Cox transformation
    pub fn boxcox(&self, lambda: f64) -> TimeSeries {
        let values = if lambda == 0.0 {
            self.values.mapv(|x| x.ln())
        } else {
            self.values.mapv(|x| (x.powf(lambda) - 1.0) / lambda)
        };

        TimeSeries {
            name: format!("boxcox({}, λ={})", self.name, lambda),
            timestamps: self.timestamps.clone(),
            values,
            frequency: self.frequency,
            metadata: self.metadata.clone(),
        }
    }

    /// Slice time series
    pub fn slice(&self, start: Option<i64>, end: Option<i64>) -> Result<TimeSeries> {
        let start_idx = match start {
            Some(t) => self.timestamps.iter().position(|&ts| ts >= t).unwrap_or(0),
            None => 0,
        };

        let end_idx = match end {
            Some(t) => self
                .timestamps
                .iter()
                .rposition(|&ts| ts <= t)
                .map(|pos| pos + 1)
                .unwrap_or(self.len()),
            None => self.len(),
        };

        if start_idx >= end_idx {
            return Err(Error::DataError("Invalid slice: start >= end".to_string()));
        }

        TimeSeries::new(
            self.name.clone(),
            self.timestamps[start_idx..end_idx].to_vec(),
            self.values
                .slice(ndarray::s![start_idx..end_idx])
                .to_owned(),
            self.frequency,
        )
    }

    /// Fill missing values (NaN) using specified method
    pub fn fillna(&self, method: FillMethod) -> TimeSeries {
        let values = self.values.to_vec();
        let filled = match method {
            FillMethod::Mean => {
                let mean = self.values.mean().unwrap_or(0.0);
                values
                    .iter()
                    .map(|&x| if x.is_nan() { mean } else { x })
                    .collect()
            }
            FillMethod::Median => {
                let mut sorted: Vec<f64> =
                    values.iter().filter(|&&x| !x.is_nan()).copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if sorted.is_empty() {
                    0.0
                } else if sorted.len() % 2 == 0 {
                    (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                } else {
                    sorted[sorted.len() / 2]
                };
                values
                    .iter()
                    .map(|&x| if x.is_nan() { median } else { x })
                    .collect()
            }
            FillMethod::ForwardFill => {
                let mut last_valid = 0.0;
                values
                    .iter()
                    .map(|&x| {
                        if !x.is_nan() {
                            last_valid = x;
                            x
                        } else {
                            last_valid
                        }
                    })
                    .collect()
            }
            FillMethod::BackwardFill => {
                let mut filled = values.clone();
                let mut last_valid = 0.0;
                for i in (0..filled.len()).rev() {
                    if !filled[i].is_nan() {
                        last_valid = filled[i];
                    } else {
                        filled[i] = last_valid;
                    }
                }
                filled
            }
            FillMethod::Linear => {
                let mut filled = values.clone();
                let mut i = 0;
                while i < filled.len() {
                    if filled[i].is_nan() {
                        let start = i;
                        while i < filled.len() && filled[i].is_nan() {
                            i += 1;
                        }
                        let end = i;

                        if start > 0 && end < filled.len() {
                            let prev_val = filled[start - 1];
                            let next_val = filled[end];
                            let step = (next_val - prev_val) / (end - start + 1) as f64;

                            for j in start..end {
                                filled[j] = prev_val + step * (j - start + 1) as f64;
                            }
                        }
                    }
                    i += 1;
                }
                filled
            }
        };

        TimeSeries {
            name: format!("{}_filled", self.name),
            timestamps: self.timestamps.clone(),
            values: Array1::from_vec(filled),
            frequency: self.frequency,
            metadata: self.metadata.clone(),
        }
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers(&self, threshold: f64) -> Vec<usize> {
        // Manual quantile calculation to avoid ndarray_stats dependency issues
        let mut sorted: Vec<f64> = self.values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let q1_idx = ((n as f64) * 0.25).floor() as usize;
        let q3_idx = ((n as f64) * 0.75).floor() as usize;

        let q1 = sorted.get(q1_idx).copied().unwrap_or(0.0);
        let q3 = sorted.get(q3_idx).copied().unwrap_or(0.0);
        let iqr = q3 - q1;

        let lower_bound = q1 - threshold * iqr;
        let upper_bound = q3 + threshold * iqr;

        self.values
            .iter()
            .enumerate()
            .filter(|&(_, &x)| x < lower_bound || x > upper_bound)
            .map(|(i, _)| i)
            .collect()
    }

    /// Convert to DataFrame
    pub fn to_dataframe(&self) -> DataFrame {
        let mut columns = HashMap::new();

        // Add timestamps
        let timestamps_series = Series::new(
            "timestamp",
            Array1::from_vec(self.timestamps.iter().map(|&t| t as f64).collect()),
        );
        columns.insert("timestamp".to_string(), timestamps_series);

        // Add values
        let values_series = Series::new(&self.name, self.values.clone());
        columns.insert(self.name.clone(), values_series);

        DataFrame::from_series(columns).unwrap_or_default()
    }
}

/// Time series statistics
#[derive(Debug, Clone)]
pub struct TimeSeriesStats {
    pub n_obs: usize,
    pub mean: f64,
    pub std: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub acf1: f64,
}

/// Methods for filling missing values
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FillMethod {
    Mean,
    Median,
    ForwardFill,
    BackwardFill,
    Linear,
}

impl TimeSeriesStats {
    /// Create summary string
    pub fn summary(&self) -> String {
        format!(
            "Observations: {}\nMean: {:.4}\nStd: {:.4}\nMin: {:.4}\nMax: {:.4}\nACF(1): {:.4}",
            self.n_obs, self.mean, self.std, self.min, self.max, self.acf1
        )
    }
}
