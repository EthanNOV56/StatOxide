//! Descriptive statistics functions

use ndarray::{Array1, Array2};
use so_core::error::{Error, Result};

/// Compute mean of an array
pub fn mean(data: &Array1<f64>) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    Some(data.sum() / data.len() as f64)
}

/// Compute variance with given degrees of freedom adjustment
pub fn variance(data: &Array1<f64>, ddof: f64) -> Option<f64> {
    let n = data.len() as f64;
    if n <= ddof {
        return None;
    }
    
    let m = mean(data)?;
    let sum_sq_diff: f64 = data.iter().map(|&x| (x - m).powi(2)).sum();
    Some(sum_sq_diff / (n - ddof))
}

/// Compute standard deviation
pub fn std(data: &Array1<f64>, ddof: f64) -> Option<f64> {
    variance(data, ddof).map(|v| v.sqrt())
}

/// Compute median
pub fn median(data: &Array1<f64>) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted.len();
    if n % 2 == 1 {
        Some(sorted[n / 2])
    } else {
        Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    }
}

/// Compute quantile using linear interpolation
pub fn quantile(data: &Array1<f64>, q: f64) -> Option<f64> {
    if data.is_empty() || !(0.0..=1.0).contains(&q) {
        return None;
    }
    
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted.len() as f64;
    let index = q * (n - 1.0);
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    
    if lower == upper {
        Some(sorted[lower])
    } else {
        let weight = index - lower as f64;
        Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
    }
}

/// Compute interquartile range (IQR)
pub fn iqr(data: &Array1<f64>) -> Option<f64> {
    let q1 = quantile(data, 0.25)?;
    let q3 = quantile(data, 0.75)?;
    Some(q3 - q1)
}

/// Compute skewness (Fisher-Pearson coefficient)
pub fn skewness(data: &Array1<f64>) -> Option<f64> {
    let n = data.len() as f64;
    if n < 3.0 {
        return None;
    }
    
    let m = mean(data)?;
    let s = std(data, 1.0)?;  // sample std
    
    if s == 0.0 {
        return Some(0.0);
    }
    
    let sum_cubes: f64 = data.iter().map(|&x| ((x - m) / s).powi(3)).sum();
    Some(sum_cubes / n)
}

/// Compute kurtosis (Fisher's definition, excess kurtosis)
pub fn kurtosis(data: &Array1<f64>) -> Option<f64> {
    let n = data.len() as f64;
    if n < 4.0 {
        return None;
    }
    
    let m = mean(data)?;
    let s = std(data, 1.0)?;  // sample std
    
    if s == 0.0 {
        return Some(0.0);
    }
    
    let sum_quarts: f64 = data.iter().map(|&x| ((x - m) / s).powi(4)).sum();
    Some(sum_quarts / n - 3.0)
}

/// Compute covariance between two arrays
pub fn covariance(x: &Array1<f64>, y: &Array1<f64>, ddof: f64) -> Option<f64> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    
    let n = x.len() as f64;
    if n <= ddof {
        return None;
    }
    
    let x_mean = mean(x)?;
    let y_mean = mean(y)?;
    
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (x[i] - x_mean) * (y[i] - y_mean);
    }
    
    Some(sum / (n - ddof))
}

/// Compute correlation coefficient (Pearson)
pub fn correlation(x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
    let cov = covariance(x, y, 1.0)?;
    let x_std = std(x, 1.0)?;
    let y_std = std(y, 1.0)?;
    
    if x_std == 0.0 || y_std == 0.0 {
        return Some(0.0);
    }
    
    Some(cov / (x_std * y_std))
}

/// Compute summary statistics for an array
#[derive(Debug, Clone)]
pub struct SummaryStats {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub q25: f64,
    pub median: f64,
    pub q75: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl SummaryStats {
    /// Compute summary statistics from data
    pub fn from_data(data: &Array1<f64>) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        
        let count = data.len();
        let mean = mean(data)?;
        let std = std(data, 1.0)?;
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let q25 = quantile(data, 0.25)?;
        let median = median(data)?;
        let q75 = quantile(data, 0.75)?;
        let skewness = skewness(data).unwrap_or(0.0);
        let kurtosis = kurtosis(data).unwrap_or(0.0);
        
        Some(Self {
            count,
            mean,
            std,
            min,
            q25,
            median,
            q75,
            max,
            skewness,
            kurtosis,
        })
    }
}

/// Compute correlation matrix
pub fn correlation_matrix(data: &Array2<f64>) -> Result<Array2<f64>> {
    let (n_samples, n_features) = data.dim();
    
    if n_samples < 2 {
        return Err(Error::DataError(
            "Need at least 2 samples to compute correlation".to_string()
        ));
    }
    
    let mut corr = Array2::zeros((n_features, n_features));
    
    for i in 0..n_features {
        for j in 0..n_features {
            let x = data.column(i);
            let y = data.column(j);
            
            if let Some(c) = correlation(&x.to_owned(), &y.to_owned()) {
                corr[(i, j)] = c;
            } else {
                corr[(i, j)] = 0.0;
            }
        }
    }
    
    Ok(corr)
}

/// Compute covariance matrix
pub fn covariance_matrix(data: &Array2<f64>, ddof: f64) -> Result<Array2<f64>> {
    let (n_samples, n_features) = data.dim();
    
    if n_samples as f64 <= ddof {
        return Err(Error::DataError(
            format!("Not enough samples for covariance with ddof={}", ddof)
        ));
    }
    
    let mut cov = Array2::zeros((n_features, n_features));
    let means: Vec<f64> = (0..n_features)
        .map(|i| data.column(i).mean().unwrap_or(0.0))
        .collect();
    
    for i in 0..n_features {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n_samples {
                sum += (data[(k, i)] - means[i]) * (data[(k, j)] - means[j]);
            }
            let value = sum / (n_samples as f64 - ddof);
            cov[(i, j)] = value;
            if i != j {
                cov[(j, i)] = value;
            }
        }
    }
    
    Ok(cov)
}