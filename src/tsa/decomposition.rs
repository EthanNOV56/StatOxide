//! Time series decomposition methods
//!
//! This module provides methods for decomposing time series into
//! trend, seasonal, and residual components.
//!
//! # Methods
//!
//! 1. **Moving Average Decomposition**: Classical decomposition using
//!    moving averages to extract trend and seasonal components.
//! 2. **STL (Seasonal-Trend decomposition using LOESS)**: Robust
//!    decomposition that handles any type of seasonality.
//! 3. **X-12-ARIMA**: Census Bureau's seasonal adjustment procedure.
//! 4. **Hodrick-Prescott Filter**: Separates trend from cyclical components.

use ndarray::{Array1, Array2};
use crate::tsa::timeseries::TimeSeries;
use crate::tools::utils::{Result, StatError};
use crate::tools::linalg;

/// Decomposition method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecompositionMethod {
    /// Additive decomposition: yₜ = Tₜ + Sₜ + Rₜ
    Additive,
    /// Multiplicative decomposition: yₜ = Tₜ × Sₜ × Rₜ
    Multiplicative,
}

/// Decomposition results
#[derive(Debug, Clone)]
pub struct DecompositionResults {
    /// Original time series
    pub original: TimeSeries,
    /// Trend component
    pub trend: TimeSeries,
    /// Seasonal component
    pub seasonal: TimeSeries,
    /// Residual component
    pub residual: TimeSeries,
    /// Decomposition method
    pub method: DecompositionMethod,
    /// Seasonal period
    pub period: usize,
}

impl DecompositionResults {
    /// Create summary of decomposition
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Time Series Decomposition\n"));
        summary.push_str(&format!("=========================\n"));
        summary.push_str(&format!("Method: {:?}\n", self.method));
        summary.push_str(&format!("Seasonal Period: {}\n", self.period));
        summary.push_str(&format!("Observations: {}\n", self.original.len()));
        
        // Variance explained by each component
        let total_var = self.original.values().var(1.0);
        let trend_var = self.trend.values().var(1.0);
        let seasonal_var = self.seasonal.values().var(1.0);
        let residual_var = self.residual.values().var(1.0);
        
        if total_var > 0.0 {
            summary.push_str(&format!("\nVariance Explained:\n"));
            summary.push_str(&format!("  Trend: {:.1}%\n", 100.0 * trend_var / total_var));
            summary.push_str(&format!("  Seasonal: {:.1}%\n", 100.0 * seasonal_var / total_var));
            summary.push_str(&format!("  Residual: {:.1}%\n", 100.0 * residual_var / total_var));
        }
        
        // Residual statistics
        let residuals = self.residual.values();
        let mean = residuals.mean().unwrap_or(0.0);
        let std = residuals.std(1.0);
        let min = residuals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = residuals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        summary.push_str(&format!("\nResidual Statistics:\n"));
        summary.push_str(&format!("  Mean: {:.4}\n", mean));
        summary.push_str(&format!("  Std: {:.4}\n", std));
        summary.push_str(&format!("  Min: {:.4}\n", min));
        summary.push_str(&format!("  Max: {:.4}\n", max));
        
        summary
    }
    
    /// Plot decomposition (would integrate with plotting library)
    pub fn plot(&self) -> String {
        "Decomposition plot would be shown here".to_string()
    }
    
    /// Get seasonally adjusted series (original - seasonal)
    pub fn seasonally_adjusted(&self) -> TimeSeries {
        let values = match self.method {
            DecompositionMethod::Additive => {
                self.original.values() - self.seasonal.values()
            },
            DecompositionMethod::Multiplicative => {
                self.original.values() / self.seasonal.values()
            },
        };
        
        TimeSeries {
            name: format!("{}_sa", self.original.name()),
            timestamps: self.original.timestamps().to_vec(),
            values,
            frequency: self.original.frequency(),
            metadata: self.original.get_metadata("").map(|_| "seasonally_adjusted".to_string())
                .map(|v| vec![("adjustment".to_string(), v)].into_iter().collect())
                .unwrap_or_default(),
        }
    }
}

/// Classical decomposition using moving averages
pub struct MovingAverageDecomposition {
    method: DecompositionMethod,
    period: usize,
    trend_window: Option<usize>,
}

impl MovingAverageDecomposition {
    /// Create new moving average decomposition
    pub fn new(method: DecompositionMethod, period: usize) -> Self {
        Self {
            method,
            period,
            trend_window: None,
        }
    }
    
    /// Set trend window size (if None, uses period)
    pub fn trend_window(mut self, window: Option<usize>) -> Self {
        self.trend_window = window;
        self
    }
    
    /// Perform decomposition
    pub fn decompose(&self, ts: &TimeSeries) -> Result<DecompositionResults> {
        let n = ts.len();
        let period = self.period;
        
        if n < 2 * period {
            return Err(StatError::DataError(
                format!("Need at least {} observations for period {}", 2 * period, period)
            ));
        }
        
        // Determine trend window
        let trend_window = self.trend_window.unwrap_or(if period % 2 == 0 {
            period + 1
        } else {
            period
        });
        
        // Step 1: Estimate trend using centered moving average
        let trend = self.estimate_trend(ts.values(), trend_window)?;
        
        // Step 2: Detrend series
        let detrended = self.detrend(ts.values(), &trend);
        
        // Step 3: Estimate seasonal component
        let seasonal = self.estimate_seasonal(&detrended, period);
        
        // Step 4: Calculate residuals
        let residual = self.calculate_residuals(ts.values(), &trend, &seasonal);
        
        // Create time series for components
        let trend_ts = TimeSeries {
            name: format!("{}_trend", ts.name()),
            timestamps: ts.timestamps().to_vec(),
            values: trend,
            frequency: ts.frequency(),
            metadata: ts.get_metadata("").map(|_| "trend".to_string())
                .map(|v| vec![("component".to_string(), v)].into_iter().collect())
                .unwrap_or_default(),
        };
        
        let seasonal_ts = TimeSeries {
            name: format!("{}_seasonal", ts.name()),
            timestamps: ts.timestamps().to_vec(),
            values: seasonal,
            frequency: ts.frequency(),
            metadata: ts.get_metadata("").map(|_| "seasonal".to_string())
                .map(|v| vec![("component".to_string(), v)].into_iter().collect())
                .unwrap_or_default(),
        };
        
        let residual_ts = TimeSeries {
            name: format!("{}_residual", ts.name()),
            timestamps: ts.timestamps().to_vec(),
            values: residual,
            frequency: ts.frequency(),
            metadata: ts.get_metadata("").map(|_| "residual".to_string())
                .map(|v| vec![("component".to_string(), v)].into_iter().collect())
                .unwrap_or_default(),
        };
        
        Ok(DecompositionResults {
            original: ts.clone(),
            trend: trend_ts,
            seasonal: seasonal_ts,
            residual: residual_ts,
            method: self.method,
            period,
        })
    }
    
    /// Estimate trend using centered moving average
    fn estimate_trend(&self, y: &Array1<f64>, window: usize) -> Result<Array1<f64>> {
        let n = y.len();
        let half_window = window / 2;
        
        let mut trend = Array1::zeros(n);
        
        // Use centered moving average for interior points
        for i in half_window..(n - half_window) {
            let mut sum = 0.0;
            for j in (i - half_window)..=(i + half_window) {
                sum += y[j];
            }
            trend[i] = sum / window as f64;
        }
        
        // Use asymmetric moving averages for edges
        for i in 0..half_window {
            let window_size = half_window + i + 1;
            let mut sum = 0.0;
            for j in 0..window_size {
                sum += y[j];
            }
            trend[i] = sum / window_size as f64;
        }
        
        for i in (n - half_window)..n {
            let window_size = half_window + (n - i);
            let mut sum = 0.0;
            for j in (n - window_size)..n {
                sum += y[j];
            }
            trend[i] = sum / window_size as f64;
        }
        
        Ok(trend)
    }
    
    /// Detrend series
    fn detrend(&self, y: &Array1<f64>, trend: &Array1<f64>) -> Array1<f64> {
        match self.method {
            DecompositionMethod::Additive => y - trend,
            DecompositionMethod::Multiplicative => y / trend,
        }
    }
    
    /// Estimate seasonal component
    fn estimate_seasonal(&self, detrended: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = detrended.len();
        let mut seasonal = Array1::zeros(n);
        
        // Calculate seasonal indices for each period position
        let _n_seasons = (n + period - 1) / period;
        let mut seasonal_sums = vec![0.0; period];
        let mut seasonal_counts = vec![0; period];
        
        for i in 0..n {
            let pos = i % period;
            seasonal_sums[pos] += detrended[i];
            seasonal_counts[pos] += 1;
        }
        
        // Average seasonal indices
        let mut seasonal_indices = vec![0.0; period];
        for i in 0..period {
            if seasonal_counts[i] > 0 {
                seasonal_indices[i] = seasonal_sums[i] / seasonal_counts[i] as f64;
            }
        }
        
        // Center seasonal indices (sum to zero for additive, average to 1 for multiplicative)
        match self.method {
            DecompositionMethod::Additive => {
                let mean: f64 = seasonal_indices.iter().sum::<f64>() / period as f64;
                for i in 0..period {
                    seasonal_indices[i] -= mean;
                }
            },
            DecompositionMethod::Multiplicative => {
                let product: f64 = seasonal_indices.iter().product();
                let geometric_mean = product.powf(1.0 / period as f64);
                for i in 0..period {
                    seasonal_indices[i] /= geometric_mean;
                }
            },
        }
        
        // Apply seasonal pattern
        for i in 0..n {
            seasonal[i] = seasonal_indices[i % period];
        }
        
        seasonal
    }
    
    /// Calculate residuals
    fn calculate_residuals(
        &self,
        y: &Array1<f64>,
        trend: &Array1<f64>,
        seasonal: &Array1<f64>,
    ) -> Array1<f64> {
        match self.method {
            DecompositionMethod::Additive => y - trend - seasonal,
            DecompositionMethod::Multiplicative => y / (trend * seasonal),
        }
    }
}

/// STL (Seasonal-Trend decomposition using LOESS)
pub struct STLDecomposition {
    period: usize,
    seasonal_window: usize,
    trend_window: usize,
    low_pass_window: usize,
    robust: bool,
    n_iter: usize,
}

impl Default for STLDecomposition {
    fn default() -> Self {
        Self {
            period: 12, // Monthly data
            seasonal_window: 7,
            trend_window: 13,
            low_pass_window: 13,
            robust: true,
            n_iter: 2,
        }
    }
}

impl STLDecomposition {
    /// Create new STL decomposition
    pub fn new(period: usize) -> Self {
        Self {
            period,
            ..Default::default()
        }
    }
    
    /// Set seasonal window size
    pub fn seasonal_window(mut self, window: usize) -> Self {
        self.seasonal_window = window;
        self
    }
    
    /// Set trend window size
    pub fn trend_window(mut self, window: usize) -> Self {
        self.trend_window = window;
        self
    }
    
    /// Set low-pass window size
    pub fn low_pass_window(mut self, window: usize) -> Self {
        self.low_pass_window = window;
        self
    }
    
    /// Enable/disable robust decomposition
    pub fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }
    
    /// Set number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }
    
    /// Perform STL decomposition
    pub fn decompose(&self, ts: &TimeSeries) -> Result<DecompositionResults> {
        let n = ts.len();
        let period = self.period;
        
        if n < 2 * period {
            return Err(StatError::DataError(
                format!("Need at least {} observations for STL with period {}", 2 * period, period)
            ));
        }
        
        let y = ts.values();
        let mut trend = Array1::zeros(n);
        let mut seasonal = Array1::zeros(n);
        let mut residual = y.clone();
        
        // Robustness weights (initially all ones)
        let mut weights = Array1::ones(n);
        
        for _ in 0..self.n_iter {
            // Inner loop
            for _ in 0..self.n_iter {
                // Step 1: Detrend
                let detrended = &residual - &trend;
                
                // Step 2: Cycle-subseries smoothing
                let seasonal_smoothed = self.smooth_seasonal(&detrended, &weights);
                
                // Step 3: Low-pass filtering of seasonal component
                let seasonal_lowpass = self.low_pass_filter(&seasonal_smoothed);
                
                // Step 4: Detrended seasonal
                seasonal = &seasonal_smoothed - &seasonal_lowpass;
                
                // Step 5: Deseasonalize
                let deseasonalized = y - &seasonal;
                
                // Step 6: Trend smoothing
                trend = self.smooth_trend(&deseasonalized, &weights);
                
                // Update residuals
                residual = y - &trend - &seasonal;
            }
            
            // Update robustness weights if robust estimation
            if self.robust {
                weights = self.update_weights(&residual);
            }
        }
        
        // Create time series for components
        let trend_ts = TimeSeries {
            name: format!("{}_trend", ts.name()),
            timestamps: ts.timestamps().to_vec(),
            values: trend,
            frequency: ts.frequency(),
            metadata: ts.get_metadata("").map(|_| "trend".to_string())
                .map(|v| vec![("component".to_string(), v)].into_iter().collect())
                .unwrap_or_default(),
        };
        
        let seasonal_ts = TimeSeries {
            name: format!("{}_seasonal", ts.name()),
            timestamps: ts.timestamps().to_vec(),
            values: seasonal,
            frequency: ts.frequency(),
            metadata: ts.get_metadata("").map(|_| "seasonal".to_string())
                .map(|v| vec![("component".to_string(), v)].into_iter().collect())
                .unwrap_or_default(),
        };
        
        let residual_ts = TimeSeries {
            name: format!("{}_residual", ts.name()),
            timestamps: ts.timestamps().to_vec(),
            values: residual,
            frequency: ts.frequency(),
            metadata: ts.get_metadata("").map(|_| "residual".to_string())
                .map(|v| vec![("component".to_string(), v)].into_iter().collect())
                .unwrap_or_default(),
        };
        
        Ok(DecompositionResults {
            original: ts.clone(),
            trend: trend_ts,
            seasonal: seasonal_ts,
            residual: residual_ts,
            method: DecompositionMethod::Additive, // STL is additive by design
            period,
        })
    }
    
    /// Smooth seasonal component using LOESS on cycle-subseries
    fn smooth_seasonal(&self, detrended: &Array1<f64>, weights: &Array1<f64>) -> Array1<f64> {
        let n = detrended.len();
        let period = self.period;
        let mut smoothed = Array1::zeros(n);
        
        // For each position in the seasonal cycle
        for pos in 0..period {
            let mut subseries = Vec::new();
            let mut subweights = Vec::new();
            let mut indices = Vec::new();
            
            // Collect all values at this seasonal position
            for i in (pos..n).step_by(period) {
                subseries.push(detrended[i]);
                subweights.push(weights[i]);
                indices.push(i);
            }
            
            if subseries.len() >= 3 {
                // Apply LOESS smoothing (simplified: use weighted moving average)
                let window = self.seasonal_window.min(subseries.len());
                
                for (idx, &i) in indices.iter().enumerate() {
                    let start = idx.saturating_sub(window / 2);
                    let end = (idx + window / 2 + 1).min(subseries.len());
                    
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;
                    
                    for j in start..end {
                        // Triangular kernel weights
                        let distance = (j as isize - idx as isize).abs() as f64;
                        let kernel_weight = 1.0 - distance / (window as f64 / 2.0);
                        let weight = kernel_weight.max(0.0) * subweights[j];
                        
                        sum += weight * subseries[j];
                        weight_sum += weight;
                    }
                    
                    if weight_sum > 0.0 {
                        smoothed[i] = sum / weight_sum;
                    } else {
                        smoothed[i] = subseries[idx];
                    }
                }
            } else {
                // Not enough points, use original values
                for (idx, &i) in indices.iter().enumerate() {
                    smoothed[i] = subseries[idx];
                }
            }
        }
        
        smoothed
    }
    
    /// Low-pass filter for seasonal component
    fn low_pass_filter(&self, seasonal: &Array1<f64>) -> Array1<f64> {
        let n = seasonal.len();
        let window = self.low_pass_window;
        let mut filtered = Array1::zeros(n);
        
        for i in 0..n {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(n);
            
            let mut sum = 0.0;
            for j in start..end {
                sum += seasonal[j];
            }
            filtered[i] = sum / (end - start) as f64;
        }
        
        // Additional moving average for smoothing
        let mut double_filtered = Array1::zeros(n);
        for i in 0..n {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(n);
            
            let mut sum = 0.0;
            for j in start..end {
                sum += filtered[j];
            }
            double_filtered[i] = sum / (end - start) as f64;
        }
        
        double_filtered
    }
    
    /// Smooth trend using LOESS
    fn smooth_trend(&self, deseasonalized: &Array1<f64>, weights: &Array1<f64>) -> Array1<f64> {
        let n = deseasonalized.len();
        let window = self.trend_window;
        let mut smoothed = Array1::zeros(n);
        
        for i in 0..n {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(n);
            
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            
            for j in start..end {
                // Quadratic kernel weights
                let distance = (j as isize - i as isize).abs() as f64;
                let normalized = distance / (window as f64 / 2.0);
                let kernel_weight = if normalized <= 1.0 {
                    (1.0 - normalized * normalized).powi(2)
                } else {
                    0.0
                };
                
                let weight = kernel_weight * weights[j];
                sum += weight * deseasonalized[j];
                weight_sum += weight;
            }
            
            if weight_sum > 0.0 {
                smoothed[i] = sum / weight_sum;
            } else {
                smoothed[i] = deseasonalized[i];
            }
        }
        
        smoothed
    }
    
    /// Update robustness weights based on residuals
    fn update_weights(&self, residuals: &Array1<f64>) -> Array1<f64> {
        let n = residuals.len();
        let median = self.median_abs(residuals);
        
        if median > 0.0 {
            let mut weights = Array1::ones(n);
            for i in 0..n {
                let u = residuals[i].abs() / (6.0 * median);
                if u < 1.0 {
                    weights[i] = (1.0 - u * u).powi(2);
                } else {
                    weights[i] = 0.0;
                }
            }
            weights
        } else {
            Array1::ones(n)
        }
    }
    
    /// Compute median absolute deviation
    fn median_abs(&self, values: &Array1<f64>) -> f64 {
        let n = values.len();
        let mut abs_values: Vec<f64> = values.iter().map(|x| x.abs()).collect();
        abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        if n % 2 == 0 {
            (abs_values[n/2 - 1] + abs_values[n/2]) / 2.0
        } else {
            abs_values[n/2]
        }
    }
}

/// Hodrick-Prescott filter for separating trend from cycle
pub struct HodrickPrescottFilter {
    /// Smoothing parameter (λ)
    /// Common values: 100 for annual, 1600 for quarterly, 14400 for monthly
    lambda: f64,
}

impl Default for HodrickPrescottFilter {
    fn default() -> Self {
        Self { lambda: 1600.0 } // Quarterly data
    }
}

impl HodrickPrescottFilter {
    /// Create new HP filter
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
    
    /// Apply HP filter to time series
    pub fn filter(&self, ts: &TimeSeries) -> Result<(TimeSeries, TimeSeries)> {
        let y = ts.values();
        let n = y.len();
        
        if n < 3 {
            return Err(StatError::DataError(
                "Need at least 3 observations for HP filter".to_string()
            ));
        }
        
        // Build the second difference matrix D
        let mut d = Array2::zeros((n - 2, n));
        for i in 0..(n - 2) {
            d[(i, i)] = 1.0;
            d[(i, i + 1)] = -2.0;
            d[(i, i + 2)] = 1.0;
        }
        
        // Compute trend: τ = (I + λD'D)^{-1} y
        let identity = Array2::eye(n);
        let d_t = d.t();
        let dtd = d_t.dot(&d);
        let mut a = identity.clone();
        a += &(self.lambda * &dtd);
        
        // Solve linear system Aτ = y
        let trend_values = linalg::solve(&a, y)
            .map_err(|e| StatError::ModelError(format!("HP filter solve failed: {}", e)))?;
        
        // Cyclical component: c = y - τ
        let cycle_values = y - &trend_values;
        
        // Create time series
        let trend_ts = TimeSeries {
            name: format!("{}_trend", ts.name()),
            timestamps: ts.timestamps().to_vec(),
            values: trend_values,
            frequency: ts.frequency(),
            metadata: ts.get_metadata("").map(|_| "trend".to_string())
                .map(|v| vec![("component".to_string(), v)].into_iter().collect())
                .unwrap_or_default(),
        };
        
        let cycle_ts = TimeSeries {
            name: format!("{}_cycle", ts.name()),
            timestamps: ts.timestamps().to_vec(),
            values: cycle_values,
            frequency: ts.frequency(),
            metadata: ts.get_metadata("").map(|_| "cycle".to_string())
                .map(|v| vec![("component".to_string(), v)].into_iter().collect())
                .unwrap_or_default(),
        };
        
        Ok((trend_ts, cycle_ts))
    }
}

/// X-12-ARIMA seasonal adjustment (simplified implementation)
pub struct X12ARIMA {
    period: usize,
}

impl X12ARIMA {
    /// Create new X-12-ARIMA adjuster
    pub fn new(period: usize) -> Self {
        Self { period }
    }
    
    /// Apply seasonal adjustment
    pub fn adjust(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        // Simplified implementation: use moving average decomposition
        let decomposition = MovingAverageDecomposition::new(DecompositionMethod::Additive, self.period)
            .decompose(ts)?;
        
        Ok(decomposition.seasonally_adjusted())
    }
}

/// Extension trait for TimeSeries decomposition methods
pub trait DecompositionExt {
    /// Moving average decomposition
    fn decompose_ma(&self, method: DecompositionMethod, period: usize) -> Result<DecompositionResults>;
    
    /// STL decomposition
    fn decompose_stl(&self, period: usize) -> Result<DecompositionResults>;
    
    /// Hodrick-Prescott filter
    fn hp_filter(&self, lambda: f64) -> Result<(TimeSeries, TimeSeries)>;
    
    /// X-12-ARIMA seasonal adjustment
    fn x12_adjust(&self, period: usize) -> Result<TimeSeries>;
}

impl DecompositionExt for TimeSeries {
    fn decompose_ma(&self, method: DecompositionMethod, period: usize) -> Result<DecompositionResults> {
        MovingAverageDecomposition::new(method, period).decompose(self)
    }
    
    fn decompose_stl(&self, period: usize) -> Result<DecompositionResults> {
        STLDecomposition::new(period).decompose(self)
    }
    
    fn hp_filter(&self, lambda: f64) -> Result<(TimeSeries, TimeSeries)> {
        HodrickPrescottFilter::new(lambda).filter(self)
    }
    
    fn x12_adjust(&self, period: usize) -> Result<TimeSeries> {
        X12ARIMA::new(period).adjust(self)
    }
}