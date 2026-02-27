//! Data validation utilities
//!
//! This module provides functions for checking data quality and validity
//! before statistical analysis.

use ndarray::{Array1, Array2};
use so_core::error::{Error, Result};

/// Check if array contains any NaN values
pub fn has_nan(arr: &Array1<f64>) -> bool {
    arr.iter().any(|&x| x.is_nan())
}

/// Check if array contains any infinite values
pub fn has_inf(arr: &Array1<f64>) -> bool {
    arr.iter().any(|&x| x.is_infinite())
}

/// Check if array contains any NaN or infinite values
pub fn is_finite_array(arr: &Array1<f64>) -> bool {
    arr.iter().all(|&x| x.is_finite())
}

/// Validate data for statistical analysis
pub fn validate_data(arr: &Array1<f64>) -> Result<()> {
    if has_nan(arr) {
        return Err(Error::DataError("Data contains NaN values".to_string()));
    }
    if has_inf(arr) {
        return Err(Error::DataError(
            "Data contains infinite values".to_string(),
        ));
    }
    Ok(())
}

/// Check if 2D array has consistent dimensions
pub fn validate_matrix(mat: &Array2<f64>) -> Result<()> {
    if mat.shape()[0] == 0 || mat.shape()[1] == 0 {
        return Err(Error::DimensionMismatch(
            "Matrix has zero dimensions".to_string(),
        ));
    }
    Ok(())
}

/// Check if array has at least n elements
pub fn check_min_samples(arr: &Array1<f64>, min_samples: usize) -> Result<()> {
    if arr.len() < min_samples {
        return Err(Error::DataError(format!(
            "Insufficient samples: {} < {}",
            arr.len(),
            min_samples
        )));
    }
    Ok(())
}
