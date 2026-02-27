//! Numerical utilities for statistical computing

use ndarray::{Array1, Array2};

/// Check if a value is finite (not NaN or infinite)
pub fn is_finite(x: f64) -> bool {
    x.is_finite()
}

/// Check if all values in an array are finite
pub fn all_finite(arr: &Array1<f64>) -> bool {
    arr.iter().all(|&x| is_finite(x))
}

/// Check if all values in a 2D array are finite
pub fn all_finite_2d(arr: &Array2<f64>) -> bool {
    arr.iter().all(|&x| is_finite(x))
}

/// Replace NaN values with a specified value
pub fn replace_nan(arr: &mut Array1<f64>, value: f64) {
    for x in arr.iter_mut() {
        if x.is_nan() {
            *x = value;
        }
    }
}

/// Replace infinite values with a specified value
pub fn replace_inf(arr: &mut Array1<f64>, value: f64) {
    for x in arr.iter_mut() {
        if x.is_infinite() {
            *x = value;
        }
    }
}

/// Clip values to a specified range
pub fn clip(arr: &mut Array1<f64>, min: f64, max: f64) {
    for x in arr.iter_mut() {
        *x = x.clamp(min, max);
    }
}

/// Standardize array (z-score normalization)
pub fn standardize(arr: &Array1<f64>) -> Option<Array1<f64>> {
    let mean = arr.mean()?;
    let std = arr.std(1.0);

    if std == 0.0 {
        return None;
    }

    Some((arr - mean) / std)
}

/// Min-max normalization to [0, 1] range
pub fn min_max_normalize(arr: &Array1<f64>) -> Option<Array1<f64>> {
    let min = arr.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = arr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if min == max {
        return None;
    }

    Some((arr - min) / (max - min))
}

/// Softmax function for probability distribution
pub fn softmax(arr: &Array1<f64>) -> Array1<f64> {
    let max_val = arr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_arr: Array1<f64> = arr.map(|&x| (x - max_val).exp());
    let sum: f64 = exp_arr.sum();
    exp_arr / sum
}

/// Sigmoid function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Logit function (inverse of sigmoid)
pub fn logit(p: f64) -> Option<f64> {
    if p <= 0.0 || p >= 1.0 {
        return None;
    }
    Some((p / (1.0 - p)).ln())
}

/// Logistic function (alias for sigmoid)
pub fn logistic(x: f64) -> f64 {
    sigmoid(x)
}

/// Compute log-sum-exp for numerical stability
pub fn log_sum_exp(arr: &Array1<f64>) -> f64 {
    let max_val = arr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp: f64 = arr.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// Compute the pairwise correlation matrix (from original tools/utils.rs)
pub fn correlation_matrix(data: &Array2<f64>) -> crate::Result<Array2<f64>> {
    use crate::error::UtilsError;

    let (n_samples, n_features) = data.dim();

    if n_samples < 2 {
        return Err(UtilsError::DataError(
            "Need at least 2 samples to compute correlation".to_string(),
        ));
    }

    let mut corr = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in 0..n_features {
            let x = data.column(i);
            let y = data.column(j);

            let x_mean = x.mean().unwrap_or(0.0);
            let y_mean = y.mean().unwrap_or(0.0);

            let mut numerator = 0.0;
            let mut denom_x = 0.0;
            let mut denom_y = 0.0;

            for k in 0..n_samples {
                let x_diff = x[k] - x_mean;
                let y_diff = y[k] - y_mean;

                numerator += x_diff * y_diff;
                denom_x += x_diff * x_diff;
                denom_y += y_diff * y_diff;
            }

            if denom_x > 0.0 && denom_y > 0.0 {
                corr[(i, j)] = numerator / (denom_x.sqrt() * denom_y.sqrt());
            } else {
                corr[(i, j)] = 0.0;
            }
        }
    }

    Ok(corr)
}

/// Compute the covariance matrix (from original tools/utils.rs)
pub fn covariance_matrix(data: &Array2<f64>, ddof: f64) -> crate::Result<Array2<f64>> {
    use crate::error::UtilsError;

    let (n_samples, n_features) = data.dim();

    if n_samples as f64 <= ddof {
        return Err(UtilsError::DataError(format!(
            "Not enough samples for covariance with ddof={}",
            ddof
        )));
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
