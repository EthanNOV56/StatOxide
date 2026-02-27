//! Time Series Analysis (TSA) module for StatOxide
//!
//! This crate provides comprehensive time series analysis tools, including:
#![allow(missing_docs)]
//!
//! 1. **Core Data Structures**: `TimeSeries` with datetime indexing
//! 2. **Stationarity Tests**: ADF, KPSS, PP tests
//! 3. **ARIMA Models**: AR, MA, ARMA, ARIMA, SARIMA
//! 4. **GARCH Models**: ARCH, GARCH for volatility modeling
//! 5. **State Space Models**: Kalman filter, structural time series
//! 6. **Decomposition**: Seasonal-Trend decomposition (STL), moving averages
//! 7. **Forecasting**: Point forecasts, prediction intervals
//! 8. **Diagnostics**: Residual analysis, model selection criteria
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use so_tsa::{TimeSeries, ARIMA, GARCH, GARCHDistribution};
//! use ndarray::Array1;
//!
//! // Create a simple time series with enough data for ARIMA
//! let values = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
//! let timestamps: Vec<i64> = (0..values.len() as i64).collect();
//! let ts = TimeSeries::new("series", timestamps, values, None).unwrap();
//!
//! // Fit ARIMA(0,0,0) model (white noise with constant)
//! let arima = ARIMA::builder(0, 0, 0)
//!     .with_constant(true)
//!     .max_iter(200)
//!     .tol(1e-4)
//!     .fit(&ts).unwrap();
//!
//! // Fit GARCH(1,1) model  
//! let garch = GARCH::builder(1, 1)
//!     .distribution(GARCHDistribution::Normal)
//!     .max_iter(200)
//!     .tol(1e-4)
//!     .fit(&ts).unwrap();
//! ```
//!
//! # References
//!
//! - Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*.
//! - Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*.
//! - R's `forecast` package and statsmodels' `tsa` module.

#![allow(non_snake_case)]
#![warn(missing_docs)]

pub mod arima;
pub mod decomposition;
pub mod forecast;
pub mod garch;
pub mod results;
pub mod statespace;
pub mod stationarity;
pub mod timeseries;
pub mod utils;

// Re-exports for convenience
pub use arima::{ARIMA, ARIMAResults, SARIMABuilder, SARIMAOrder};
pub use decomposition::{
    DecompositionExt, DecompositionMethod, DecompositionResults, HodrickPrescottFilter,
    MovingAverageDecomposition, STLDecomposition, X12ARIMA,
};
pub use forecast::{
    ForecastMetrics, IntervalMethod, PredictionInterval, PredictionIntervals, TimeSeriesCV,
};
pub use garch::{ARCH, GARCH, GARCHDistribution, GARCHOrder, GARCHResults};
pub use results::{ModelComparison, ResidualDiagnostics, TSAResults};
pub use statespace::{KalmanFilter, StateSpaceModel};
pub use stationarity::{ADFTest, KPSSTest, PPTest, StationarityTest};
pub use timeseries::TimeSeries;
pub use utils::{
    acf, box_cox, box_cox_lambda, ccf, detrend_poly, diebold_mariano, ewma, forecast_errors,
    information_criteria, pacf, periodogram, rolling_statistic, seasonal_dummies, spectrum,
};

// Common prelude for TSA
pub mod prelude {
    pub use super::{
        ADFTest, ARCH, ARIMA, ARIMAResults, DecompositionExt, DecompositionMethod,
        DecompositionResults, ForecastMetrics, GARCH, GARCHDistribution, GARCHOrder, GARCHResults,
        HodrickPrescottFilter, IntervalMethod, KPSSTest, KalmanFilter, ModelComparison,
        MovingAverageDecomposition, PPTest, PredictionInterval, PredictionIntervals,
        ResidualDiagnostics, SARIMABuilder, SARIMAOrder, STLDecomposition, StateSpaceModel,
        StationarityTest, TSAResults, TimeSeries, TimeSeriesCV, X12ARIMA,
    };
}
