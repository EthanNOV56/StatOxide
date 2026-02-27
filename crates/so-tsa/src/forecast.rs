//! Forecasting evaluation and prediction intervals
//!
//! This module provides tools for evaluating forecast accuracy
//! and constructing prediction intervals.
//!
//! # Evaluation Metrics
//!
//! - **MAE**: Mean Absolute Error
//! - **MSE**: Mean Squared Error  
//! - **RMSE**: Root Mean Squared Error
//! - **MAPE**: Mean Absolute Percentage Error
//! - **SMAPE**: Symmetric Mean Absolute Percentage Error
//! - **MASE**: Mean Absolute Scaled Error
//! - **Theil's U**: Theil's inequality coefficient
//!
//! # Prediction Intervals
//!
//! Methods for constructing prediction intervals:
//!
//! 1. **Normal approximation**: Assuming normal distribution of errors
//! 2. **Empirical quantiles**: Using empirical distribution of residuals
//! 3. **Bootstrapping**: Resampling residuals to estimate uncertainty
//! 4. **Conformal prediction**: Distribution-free intervals

use ndarray::Array1;
use rand::Rng;
use so_core::error::{Error, Result};
use std::collections::HashMap;

/// Forecast evaluation metrics
#[derive(Debug, Clone)]
pub struct ForecastMetrics {
    /// Mean Absolute Error
    pub mae: f64,
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error (%)
    pub mape: f64,
    /// Symmetric Mean Absolute Percentage Error (%)
    pub smape: f64,
    /// Mean Absolute Scaled Error
    pub mase: f64,
    /// Theil's U statistic
    pub theils_u: f64,
    /// R-squared of forecast
    pub r_squared: f64,
    /// Number of observations
    pub n: usize,
    /// Additional custom metrics
    pub custom: HashMap<String, f64>,
}

impl ForecastMetrics {
    /// Create new forecast metrics from actual and predicted values
    pub fn new(actual: &Array1<f64>, predicted: &Array1<f64>) -> Result<Self> {
        let n = actual.len();
        if n != predicted.len() {
            return Err(Error::DataError(format!(
                "Actual and predicted lengths differ: {} vs {}",
                n,
                predicted.len()
            )));
        }

        if n == 0 {
            return Err(Error::DataError(
                "Empty data for forecast evaluation".to_string(),
            ));
        }

        // Calculate errors
        let mut errors = Array1::zeros(n);
        let mut abs_errors = Array1::zeros(n);
        let mut squared_errors = Array1::zeros(n);
        let mut abs_percentage_errors = Array1::zeros(n);
        let mut symmetric_errors = Array1::zeros(n);

        for i in 0..n {
            let error = actual[i] - predicted[i];
            errors[i] = error;
            abs_errors[i] = error.abs();
            squared_errors[i] = error.powi(2);

            if actual[i] != 0.0 {
                abs_percentage_errors[i] = (error.abs() / actual[i].abs()) * 100.0;
                symmetric_errors[i] =
                    (error.abs() / (actual[i].abs() + predicted[i].abs())) * 200.0;
            }
        }

        // Basic metrics
        let mae = abs_errors.mean().unwrap_or(0.0);
        let mse = squared_errors.mean().unwrap_or(0.0);
        let rmse = mse.sqrt();

        // Percentage errors (handle zeros)
        let mape = if abs_percentage_errors.iter().any(|&x| x.is_finite()) {
            abs_percentage_errors
                .iter()
                .filter(|&&x| x.is_finite())
                .sum::<f64>()
                / abs_percentage_errors
                    .iter()
                    .filter(|&&x| x.is_finite())
                    .count() as f64
        } else {
            0.0
        };

        let smape = if symmetric_errors.iter().any(|&x| x.is_finite()) {
            symmetric_errors
                .iter()
                .filter(|&&x| x.is_finite())
                .sum::<f64>()
                / symmetric_errors.iter().filter(|&&x| x.is_finite()).count() as f64
        } else {
            0.0
        };

        // MASE - need naive forecast errors
        let mase = if n > 1 {
            let mut naive_errors = Array1::zeros(n - 1);
            for i in 1..n {
                naive_errors[i - 1] = (actual[i] - actual[i - 1]).abs();
            }
            let mean_naive_error = naive_errors.mean().unwrap_or(1.0);
            if mean_naive_error > 0.0 {
                mae / mean_naive_error
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Theil's U
        let theils_u = if actual.var(1.0) > 0.0 && predicted.var(1.0) > 0.0 {
            rmse / (actual.var(1.0).sqrt() + predicted.var(1.0).sqrt())
        } else {
            0.0
        };

        // R-squared
        let ss_res = squared_errors.sum();
        let ss_tot = actual.var(1.0) * n as f64;
        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        Ok(Self {
            mae,
            mse,
            rmse,
            mape,
            smape,
            mase,
            theils_u,
            r_squared,
            n,
            custom: HashMap::new(),
        })
    }

    /// Add custom metric
    pub fn with_custom(mut self, name: &str, value: f64) -> Self {
        self.custom.insert(name.to_string(), value);
        self
    }

    /// Create summary string
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Forecast Evaluation Metrics\n");
        summary.push_str("===========================\n");
        summary.push_str(&format!("Observations: {}\n", self.n));
        summary.push_str(&format!("MAE:  {:.4}\n", self.mae));
        summary.push_str(&format!("MSE:  {:.4}\n", self.mse));
        summary.push_str(&format!("RMSE: {:.4}\n", self.rmse));
        summary.push_str(&format!("MAPE: {:.2}%\n", self.mape));
        summary.push_str(&format!("sMAPE: {:.2}%\n", self.smape));
        summary.push_str(&format!("MASE: {:.4}\n", self.mase));
        summary.push_str(&format!("Theil's U: {:.4}\n", self.theils_u));
        summary.push_str(&format!("R²:   {:.4}\n", self.r_squared));

        if !self.custom.is_empty() {
            summary.push_str("\nCustom Metrics:\n");
            for (name, value) in &self.custom {
                summary.push_str(&format!("  {}: {:.4}\n", name, value));
            }
        }

        // Interpretation
        summary.push_str("\nInterpretation:\n");
        if self.mape < 10.0 {
            summary.push_str("  MAPE < 10%: Highly accurate forecast\n");
        } else if self.mape < 20.0 {
            summary.push_str("  MAPE < 20%: Good forecast\n");
        } else if self.mape < 50.0 {
            summary.push_str("  MAPE < 50%: Reasonable forecast\n");
        } else {
            summary.push_str("  MAPE ≥ 50%: Inaccurate forecast\n");
        }

        if self.mase < 1.0 {
            summary.push_str("  MASE < 1: Better than naive forecast\n");
        } else {
            summary.push_str("  MASE ≥ 1: Worse than naive forecast\n");
        }

        summary
    }

    /// Compare two forecast methods
    pub fn compare(&self, other: &Self, name_a: &str, name_b: &str) -> String {
        let mut comparison = String::new();
        comparison.push_str(&format!("Forecast Comparison: {} vs {}\n", name_a, name_b));
        comparison.push_str("===================================\n");

        comparison.push_str(&format!(
            "MAE:  {:.4} vs {:.4} ({:+.2}%)\n",
            self.mae,
            other.mae,
            (other.mae - self.mae) / self.mae.max(1e-10) * 100.0
        ));
        comparison.push_str(&format!(
            "RMSE: {:.4} vs {:.4} ({:+.2}%)\n",
            self.rmse,
            other.rmse,
            (other.rmse - self.rmse) / self.rmse.max(1e-10) * 100.0
        ));
        comparison.push_str(&format!(
            "MAPE: {:.2}% vs {:.2}% ({:+.2}pp)\n",
            self.mape,
            other.mape,
            other.mape - self.mape
        ));
        comparison.push_str(&format!(
            "MASE: {:.4} vs {:.4} ({:+.2}%)\n",
            self.mase,
            other.mase,
            (other.mase - self.mase) / self.mase.max(1e-10) * 100.0
        ));

        comparison
    }
}

/// Prediction interval methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntervalMethod {
    /// Normal approximation: ± z * σ
    Normal,
    /// Empirical quantiles of residuals
    Empirical,
    /// Bootstrapped intervals
    Bootstrap,
    /// Conformal prediction
    Conformal,
}

/// Prediction interval
#[derive(Debug, Clone)]
pub struct PredictionInterval {
    /// Point forecast
    pub point: f64,
    /// Lower bound
    pub lower: f64,
    /// Upper bound
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub level: f64,
    /// Method used
    pub method: IntervalMethod,
}

impl PredictionInterval {
    /// Check if actual value is within interval
    pub fn contains(&self, actual: f64) -> bool {
        actual >= self.lower && actual <= self.upper
    }

    /// Interval width
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Interval as string
    pub fn to_string(&self) -> String {
        format!(
            "{:.4} [{:.4}, {:.4}] ({}%)",
            self.point,
            self.lower,
            self.upper,
            (self.level * 100.0) as i32
        )
    }
}

/// Prediction intervals for multiple forecasts
#[derive(Debug, Clone)]
pub struct PredictionIntervals {
    /// Point forecasts
    pub points: Array1<f64>,
    /// Lower bounds
    pub lower: Array1<f64>,
    /// Upper bounds
    pub upper: Array1<f64>,
    /// Confidence level
    pub level: f64,
    /// Method used
    pub method: IntervalMethod,
}

impl PredictionIntervals {
    /// Create normal approximation intervals
    pub fn normal(points: &Array1<f64>, std_dev: f64, level: f64) -> Self {
        let z = normal_quantile(1.0 - (1.0 - level) / 2.0);
        let margin = z * std_dev;

        let lower = points - margin;
        let upper = points + margin;

        Self {
            points: points.clone(),
            lower,
            upper,
            level,
            method: IntervalMethod::Normal,
        }
    }

    /// Create intervals from empirical residuals
    pub fn empirical(points: &Array1<f64>, residuals: &Array1<f64>, level: f64) -> Self {
        let n = residuals.len();
        let mut sorted_residuals: Vec<f64> = residuals.iter().copied().collect();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx = ((1.0 - level) / 2.0 * n as f64).floor() as usize;
        let upper_idx = ((1.0 + level) / 2.0 * n as f64).floor() as usize;

        let lower_quantile = sorted_residuals[lower_idx.min(n - 1)];
        let upper_quantile = sorted_residuals[upper_idx.min(n - 1)];

        let lower = points + lower_quantile;
        let upper = points + upper_quantile;

        Self {
            points: points.clone(),
            lower,
            upper,
            level,
            method: IntervalMethod::Empirical,
        }
    }

    /// Create bootstrapped intervals
    pub fn bootstrap(
        points: &Array1<f64>,
        residuals: &Array1<f64>,
        level: f64,
        n_bootstrap: usize,
    ) -> Self {
        let n = points.len();
        let r = residuals.len();

        let mut bootstrap_forecasts = Vec::new();

        for _ in 0..n_bootstrap {
            let mut boot_points = Array1::zeros(n);

            for i in 0..n {
                // Sample residual with replacement
                let idx = rand::rng().random_range(0..r);
                let boot_error = residuals[idx];
                boot_points[i] = points[i] + boot_error;
            }

            bootstrap_forecasts.push(boot_points);
        }

        // Calculate quantiles
        let mut lower = Array1::zeros(n);
        let mut upper = Array1::zeros(n);

        for i in 0..n {
            let mut values: Vec<f64> = bootstrap_forecasts.iter().map(|arr| arr[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx = ((1.0 - level) / 2.0 * n_bootstrap as f64).floor() as usize;
            let upper_idx = ((1.0 + level) / 2.0 * n_bootstrap as f64).floor() as usize;

            lower[i] = values[lower_idx.min(n_bootstrap - 1)];
            upper[i] = values[upper_idx.min(n_bootstrap - 1)];
        }

        Self {
            points: points.clone(),
            lower,
            upper,
            level,
            method: IntervalMethod::Bootstrap,
        }
    }

    /// Check coverage (proportion of actual values within intervals)
    pub fn coverage(&self, actual: &Array1<f64>) -> f64 {
        let n = actual.len();
        let mut count = 0;

        for i in 0..n.min(self.points.len()) {
            if actual[i] >= self.lower[i] && actual[i] <= self.upper[i] {
                count += 1;
            }
        }

        count as f64 / n.min(self.points.len()) as f64
    }

    /// Average interval width
    pub fn average_width(&self) -> f64 {
        let n = self.points.len();
        let mut total = 0.0;

        for i in 0..n {
            total += self.upper[i] - self.lower[i];
        }

        total / n as f64
    }
}

/// Cross-validation for time series
pub struct TimeSeriesCV {
    /// Number of folds
    pub n_folds: usize,
    /// Minimum training size
    pub min_train_size: usize,
    /// Step size between folds
    pub step_size: usize,
    /// Whether to use expanding window
    pub expanding: bool,
}

impl Default for TimeSeriesCV {
    fn default() -> Self {
        Self {
            n_folds: 5,
            min_train_size: 20,
            step_size: 1,
            expanding: false,
        }
    }
}

impl TimeSeriesCV {
    /// Create new time series cross-validator
    pub fn new(n_folds: usize) -> Self {
        Self {
            n_folds,
            ..Default::default()
        }
    }

    /// Perform cross-validation
    pub fn cross_validate<F>(
        &self,
        data: &Array1<f64>,
        forecast_fn: F,
    ) -> Result<Vec<ForecastMetrics>>
    where
        F: Fn(&Array1<f64>, usize) -> Result<Array1<f64>>,
    {
        let n = data.len();
        let mut results = Vec::new();

        // Determine fold boundaries
        let test_size = (n - self.min_train_size) / self.n_folds.max(1);
        if test_size == 0 {
            return Err(Error::DataError(
                "Not enough data for cross-validation".to_string(),
            ));
        }

        for fold in 0..self.n_folds {
            let train_end = self.min_train_size + fold * self.step_size;
            if train_end >= n {
                break;
            }

            let test_end = (train_end + test_size).min(n);

            // Split data
            let train_data = data.slice(ndarray::s![..train_end]).to_owned();
            let test_data = data.slice(ndarray::s![train_end..test_end]).to_owned();

            // Generate forecasts
            let horizon = test_data.len();
            let forecasts = forecast_fn(&train_data, horizon)?;

            // Evaluate
            if forecasts.len() == test_data.len() {
                let metrics = ForecastMetrics::new(&test_data, &forecasts)?;
                results.push(metrics);
            }
        }

        Ok(results)
    }

    /// Aggregate cross-validation results
    pub fn aggregate_metrics(&self, metrics: &[ForecastMetrics]) -> ForecastMetrics {
        let n = metrics.len();
        let mut aggregated = ForecastMetrics {
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mape: 0.0,
            smape: 0.0,
            mase: 0.0,
            theils_u: 0.0,
            r_squared: 0.0,
            n: metrics.iter().map(|m| m.n).sum(),
            custom: HashMap::new(),
        };

        for metric in metrics {
            aggregated.mae += metric.mae;
            aggregated.mse += metric.mse;
            aggregated.rmse += metric.rmse;
            aggregated.mape += metric.mape;
            aggregated.smape += metric.smape;
            aggregated.mase += metric.mase;
            aggregated.theils_u += metric.theils_u;
            aggregated.r_squared += metric.r_squared;
        }

        aggregated.mae /= n as f64;
        aggregated.mse /= n as f64;
        aggregated.rmse /= n as f64;
        aggregated.mape /= n as f64;
        aggregated.smape /= n as f64;
        aggregated.mase /= n as f64;
        aggregated.theils_u /= n as f64;
        aggregated.r_squared /= n as f64;

        aggregated
    }
}

/// Normal distribution quantile (simplified)
fn normal_quantile(p: f64) -> f64 {
    // Approximation of inverse normal CDF
    let t = if p <= 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let num = c0 + c1 * t + c2 * t.powi(2);
    let den = 1.0 + d1 * t + d2 * t.powi(2) + d3 * t.powi(3);

    if p <= 0.5 {
        -t + num / den
    } else {
        t - num / den
    }
}
