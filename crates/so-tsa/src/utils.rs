//! Utility functions for time series analysis
//!
//! This module provides utility functions for working with time series,
//! including autocorrelation functions, periodogram estimation,
//! and various transformations.

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use so_core::error::{Result, Error};

/// Calculate autocorrelation function (ACF)
pub fn acf(series: &Array1<f64>, max_lag: usize) -> Array1<f64> {
    let n = series.len();
    let mean = series.mean().unwrap_or(0.0);
    let variance = series.var(1.0);
    
    if variance <= 0.0 || n <= 1 {
        return Array1::zeros(max_lag.min(n - 1));
    }
    
    let mut acf_values = Array1::zeros(max_lag.min(n - 1));
    
    for lag in 1..=max_lag.min(n - 1) {
        let mut autocov = 0.0;
        for t in lag..n {
            autocov += (series[t] - mean) * (series[t - lag] - mean);
        }
        acf_values[lag - 1] = autocov / (variance * n as f64);
    }
    
    acf_values
}

/// Calculate partial autocorrelation function (PACF)
pub fn pacf(series: &Array1<f64>, max_lag: usize) -> Result<Array1<f64>> {
    let n = series.len();
    if n <= max_lag {
        return Err(Error::DataError(
            format!("Need more observations than max_lag: n={}, max_lag={}", n, max_lag)
        ));
    }
    
    let mut pacf_values = Array1::zeros(max_lag);
    
    // Use Durbin-Levinson algorithm
    let mut phi = Array1::zeros(max_lag + 1);
    let mut v = Array1::zeros(max_lag + 1);
    
    // Initial values
    phi[0] = 1.0;
    v[0] = series.var(1.0);
    
    // Autocorrelations
    let r = acf(series, max_lag);
    
    for k in 1..=max_lag {
        // Compute phi_kk
        let mut num = r[k - 1];
        for j in 1..k {
            num -= phi[j] * r[(k - j - 1).min(max_lag - 1)];
        }
        
        let phi_kk = num / v[k - 1];
        pacf_values[k - 1] = phi_kk;
        
        // Update phi and v
        phi[k] = phi_kk;
        for j in 1..k {
            phi[j] = phi[j] - phi_kk * phi[k - j];
        }
        
        v[k] = v[k - 1] * (1.0 - phi_kk.powi(2));
    }
    
    Ok(pacf_values)
}

/// Calculate cross-correlation function (CCF)
pub fn ccf(x: &Array1<f64>, y: &Array1<f64>, max_lag: usize) -> Array1<f64> {
    let n = x.len();
    if n != y.len() {
        return Array1::zeros(max_lag * 2 + 1);
    }
    
    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);
    let x_var = x.var(1.0);
    let y_var = y.var(1.0);
    
    if x_var <= 0.0 || y_var <= 0.0 {
        return Array1::zeros(max_lag * 2 + 1);
    }
    
    let mut ccf_values = Array1::zeros(max_lag * 2 + 1);
    
    for lag in -(max_lag as isize)..=max_lag as isize {
        let idx = (lag + max_lag as isize) as usize;
        let mut cross_cov = 0.0;
        
        if lag >= 0 {
            for t in lag as usize..n {
                cross_cov += (x[t] - x_mean) * (y[t - lag as usize] - y_mean);
            }
        } else {
            for t in (-lag) as usize..n {
                cross_cov += (x[t + lag as usize] - x_mean) * (y[t] - y_mean);
            }
        }
        
        ccf_values[idx] = cross_cov / (x_var.sqrt() * y_var.sqrt() * n as f64);
    }
    
    ccf_values
}

/// Calculate periodogram (spectral density estimate)
pub fn periodogram(series: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let n = series.len();
    let n_freq = n / 2 + 1;
    
    let mut frequencies = Array1::zeros(n_freq);
    let mut spectrum = Array1::zeros(n_freq);
    
    // Calculate Fourier frequencies
    for k in 0..n_freq {
        frequencies[k] = k as f64 / n as f64;
        
        // Discrete Fourier Transform (simplified)
        let mut real = 0.0;
        let mut imag = 0.0;
        
        for t in 0..n {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
            real += series[t] * angle.cos();
            imag += series[t] * angle.sin();
        }
        
        // Periodogram: squared magnitude / n
        spectrum[k] = (real.powi(2) + imag.powi(2)) / n as f64;
    }
    
    (frequencies, spectrum)
}

/// Calculate spectral density using smoothed periodogram
pub fn spectrum(series: &Array1<f64>, window: &str, bandwidth: f64) -> (Array1<f64>, Array1<f64>) {
    let (freq, mut periodogram) = periodogram(series);
    let n_freq = freq.len();
    
    // Apply smoothing window
    match window {
        "bartlett" => {
            let m = (bandwidth * n_freq as f64).round() as usize;
            for k in 0..n_freq {
                let mut smoothed = 0.0;
                let mut weight_sum = 0.0;
                
                for j in (k as isize - m as isize)..=(k as isize + m as isize) {
                    if j >= 0 && (j as usize) < n_freq {
                        let weight = 1.0 - (j - k as isize).abs() as f64 / m as f64;
                        smoothed += weight * periodogram[j as usize];
                        weight_sum += weight;
                    }
                }
                
                if weight_sum > 0.0 {
                    periodogram[k] = smoothed / weight_sum;
                }
            }
        },
        "parzen" => {
            let m = (bandwidth * n_freq as f64).round() as usize;
            for k in 0..n_freq {
                let mut smoothed = 0.0;
                let mut weight_sum = 0.0;
                
                for j in (k as isize - m as isize)..=(k as isize + m as isize) {
                    if j >= 0 && (j as usize) < n_freq {
                        let u = (j - k as isize).abs() as f64 / m as f64;
                        let weight = if u <= 0.5 {
                            1.0 - 6.0 * u.powi(2) + 6.0 * u.powi(3)
                        } else {
                            2.0 * (1.0 - u).powi(3)
                        };
                        
                        smoothed += weight * periodogram[j as usize];
                        weight_sum += weight;
                    }
                }
                
                if weight_sum > 0.0 {
                    periodogram[k] = smoothed / weight_sum;
                }
            }
        },
        _ => {}, // No smoothing
    }
    
    (freq, periodogram)
}

/// De-trend a series using polynomial regression
pub fn detrend_poly(series: &Array1<f64>, degree: usize) -> Result<Array1<f64>> {
    let n = series.len();
    if n <= degree {
        return Err(Error::DataError(
            format!("Need more observations than degree: n={}, degree={}", n, degree)
        ));
    }
    
    // Build Vandermonde matrix
    let mut x = Array2::zeros((n, degree + 1));
    for i in 0..n {
        for j in 0..=degree {
            x[(i, j)] = (i as f64).powi(j as i32);
        }
    }
    
    // Solve least squares: (X'X)β = X'y
    use so_linalg;
    let xt = x.t();
    let xtx = xt.dot(&x);
    let xty = xt.dot(series);
    
    let beta = so_linalg::solve(&xtx, &xty)
        .map_err(|e| Error::DataError(format!("Detrend failed: {}", e)))?;
    
    // Calculate fitted trend
    let trend = x.dot(&beta);
    let detrended = series - &trend;
    
    Ok(detrended)
}

/// Apply Box-Cox transformation
pub fn box_cox(series: &Array1<f64>, lambda: f64) -> Array1<f64> {
    if lambda.abs() < 1e-10 {
        // Log transformation (special case)
        series.mapv(|x| (x + 1e-10).ln())
    } else {
        series.mapv(|x| (x.powf(lambda) - 1.0) / lambda)
    }
}

/// Find optimal Box-Cox lambda using maximum likelihood
pub fn box_cox_lambda(series: &Array1<f64>, lambda_range: (f64, f64), steps: usize) -> f64 {
    let (min_lambda, max_lambda) = lambda_range;
    let step = (max_lambda - min_lambda) / steps as f64;
    
    let mut best_lambda = 0.0;
    let mut best_log_lik = f64::NEG_INFINITY;
    
    for i in 0..=steps {
        let lambda = min_lambda + i as f64 * step;
        let transformed = box_cox(series, lambda);
        
        // Log-likelihood assuming normality
        let mean = transformed.mean().unwrap_or(0.0);
        let variance = transformed.var(1.0);
        if variance > 0.0 {
            let log_lik = -0.5 * series.len() as f64 * (2.0 * std::f64::consts::PI * variance).ln()
                - 0.5 / variance * transformed.mapv(|x| (x - mean).powi(2)).sum()
                + (lambda - 1.0) * series.mapv(|x| x.ln()).sum();
            
            if log_lik > best_log_lik {
                best_log_lik = log_lik;
                best_lambda = lambda;
            }
        }
    }
    
    best_lambda
}

/// Calculate rolling statistics
pub fn rolling_statistic(
    series: &Array1<f64>,
    window: usize,
    stat: &str,
    center: bool,
) -> Array1<f64> {
    let n = series.len();
    let mut result = Array1::zeros(n);
    
    for i in 0..n {
        let start = if center {
            i.saturating_sub(window / 2)
        } else {
            i.saturating_sub(window - 1).min(i)
        };
        let end = if center {
            (i + window / 2 + 1).min(n)
        } else {
            (i + 1).min(n)
        };
        
        let window_size = end - start;
        if window_size > 0 {
            let window_data = series.slice(ndarray::s![start..end]);
            
            result[i] = match stat {
                "mean" => window_data.mean().unwrap_or(0.0),
                "median" => {
                    let mut sorted: Vec<f64> = window_data.to_vec();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = window_size / 2;
                    if window_size % 2 == 0 {
                        (sorted[mid - 1] + sorted[mid]) / 2.0
                    } else {
                        sorted[mid]
                    }
                },
                "std" => window_data.std(1.0),
                "var" => window_data.var(1.0),
                "min" => window_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                "max" => window_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                "sum" => window_data.sum(),
                _ => window_data.mean().unwrap_or(0.0),
            };
        }
    }
    
    result
}

/// Calculate exponential weighted moving average (EWMA)
pub fn ewma(series: &Array1<f64>, alpha: f64) -> Array1<f64> {
    let n = series.len();
    let mut ewma_values = Array1::zeros(n);
    
    if n > 0 {
        ewma_values[0] = series[0];
        for i in 1..n {
            ewma_values[i] = alpha * series[i] + (1.0 - alpha) * ewma_values[i - 1];
        }
    }
    
    ewma_values
}

/// Calculate seasonal dummy variables
pub fn seasonal_dummies(n: usize, period: usize, include_all: bool) -> Array2<f64> {
    let n_dummies = if include_all { period } else { period - 1 };
    let mut dummies = Array2::zeros((n, n_dummies));
    
    for i in 0..n {
        let season = i % period;
        if season < n_dummies {
            dummies[(i, season)] = 1.0;
        }
    }
    
    dummies
}

/// Calculate forecast error metrics
pub fn forecast_errors(actual: &Array1<f64>, forecast: &Array1<f64>) -> HashMap<String, f64> {
    use std::collections::HashMap;
    
    let n = actual.len().min(forecast.len());
    let mut errors = HashMap::new();
    
    if n == 0 {
        return errors;
    }
    
    let mut mae = 0.0;
    let mut mse = 0.0;
    let mut mape = 0.0;
    let mut mape_count = 0;
    
    for i in 0..n {
        let error = actual[i] - forecast[i];
        mae += error.abs();
        mse += error.powi(2);
        
        if actual[i] != 0.0 {
            mape += (error.abs() / actual[i].abs()) * 100.0;
            mape_count += 1;
        }
    }
    
    errors.insert("MAE".to_string(), mae / n as f64);
    errors.insert("MSE".to_string(), mse / n as f64);
    errors.insert("RMSE".to_string(), (mse / n as f64).sqrt());
    
    if mape_count > 0 {
        errors.insert("MAPE".to_string(), mape / mape_count as f64);
    }
    
    // Theil's U statistic
    let naive_forecast: f64 = if n > 1 {
        let mut naive_error = 0.0;
        for i in 1..n {
            naive_error += (actual[i] - actual[i - 1]).powi(2);
        }
        (naive_error / (n - 1) as f64).sqrt()
    } else { 0.0 };
    
    let rmse = (mse / n as f64).sqrt();
    if naive_forecast > 0.0 {
        errors.insert("TheilU".to_string(), rmse / naive_forecast);
    }
    
    errors
}

/// Calculate information criteria for model selection
pub fn information_criteria(
    log_likelihood: f64,
    n_obs: usize,
    n_params: usize,
) -> HashMap<String, f64> {
    use std::collections::HashMap;
    
    let mut criteria = HashMap::new();
    
    // Akaike Information Criterion
    let aic = 2.0 * n_params as f64 - 2.0 * log_likelihood;
    criteria.insert("AIC".to_string(), aic);
    
    // Bayesian Information Criterion
    let bic = (n_obs as f64).ln() * n_params as f64 - 2.0 * log_likelihood;
    criteria.insert("BIC".to_string(), bic);
    
    // Corrected AIC for small samples
    let aicc = if n_obs > n_params + 1 {
        aic + 2.0 * n_params as f64 * (n_params as f64 + 1.0) / (n_obs as f64 - n_params as f64 - 1.0)
    } else {
        aic
    };
    criteria.insert("AICc".to_string(), aicc);
    
    criteria
}

/// Calculate Diebold-Mariano test for forecast comparison
pub fn diebold_mariano(
    errors_a: &Array1<f64>,
    errors_b: &Array1<f64>,
    loss_fn: &str,
) -> (f64, f64) {
    let n = errors_a.len().min(errors_b.len());
    if n < 2 {
        return (0.0, 1.0);
    }
    
    // Calculate loss differential
    let mut d = Array1::zeros(n);
    for i in 0..n {
        let loss_a = match loss_fn {
            "squared" => errors_a[i].powi(2),
            "absolute" => errors_a[i].abs(),
            "percentage" => if errors_a[i] != 0.0 { errors_a[i].abs() } else { 0.0 },
            _ => errors_a[i].powi(2),
        };
        
        let loss_b = match loss_fn {
            "squared" => errors_b[i].powi(2),
            "absolute" => errors_b[i].abs(),
            "percentage" => if errors_b[i] != 0.0 { errors_b[i].abs() } else { 0.0 },
            _ => errors_b[i].powi(2),
        };
        
        d[i] = loss_a - loss_b;
    }
    
    // Test statistic
    let mean_d = d.mean().unwrap_or(0.0);
    let var_d = d.var(1.0);
    
    if var_d <= 0.0 {
        return (0.0, 1.0);
    }
    
    let dm_stat = mean_d / (var_d / n as f64).sqrt();
    
    // p-value (two-sided normal test)
    use super::results::chi2_cdf;
    let p_value = 2.0 * (1.0 - chi2_cdf(1, dm_stat.abs()));
    
    (dm_stat, p_value)
}