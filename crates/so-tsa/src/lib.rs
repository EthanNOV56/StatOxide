//! Time Series Analysis (TSA) module for StatOxide
//!
//! This crate provides comprehensive time series analysis tools, including:
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
//! ```rust
//! use so_tsa::{TimeSeries, ARIMA, GARCH};
//! use so_core::DataFrame;
//!
//! // Load time series data
//! let ts = TimeSeries::from_dataframe(&df, "value", "date")?;
//!
//! // Fit ARIMA(1,1,1) model
//! let arima = ARIMA::new(1, 1, 1)
//!     .seasonal(1, 1, 1, 12)  // seasonal ARIMA
//!     .fit(&ts)?;
//!
//! // Forecast next 10 periods
//! let forecast = arima.forecast(10);
//!
//! // Fit GARCH(1,1) model
//! let garch = GARCH::new(1, 1)
//!     .distribution(Distribution::Normal)
//!     .fit(&ts.residuals())?;
//! ```
//!
//! # References
//!
//! - Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis*.
//! - Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*.
//! - R's `forecast` package and statsmodels' `tsa` module.

#![warn(missing_docs)]

pub mod timeseries;
pub mod stationarity;
pub mod arima;
pub mod garch;
pub mod statespace;
pub mod decomposition;
pub mod forecast;
pub mod results;
pub mod utils;

// Re-exports for convenience
pub use timeseries::TimeSeries;
pub use stationarity::{StationarityTest, ADFTest, KPSSTest, PPTest};
pub use arima::{ARIMA, SARIMABuilder, ARIMAResults, SARIMAOrder};
pub use garch::{GARCH, ARCH, GARCHResults, GARCHOrder, GARCHDistribution};
pub use statespace::{StateSpaceModel, KalmanFilter};
pub use decomposition::{DecompositionMethod, DecompositionResults, MovingAverageDecomposition, STLDecomposition, HodrickPrescottFilter, X12ARIMA, DecompositionExt};
pub use forecast::{ForecastMetrics, IntervalMethod, PredictionInterval, PredictionIntervals, TimeSeriesCV};
pub use results::{TSAResults, ResidualDiagnostics, ModelComparison};
pub use utils::{acf, pacf, ccf, periodogram, spectrum, detrend_poly, box_cox, box_cox_lambda, rolling_statistic, ewma, seasonal_dummies, forecast_errors, information_criteria, diebold_mariano};

// Common prelude for TSA
pub mod prelude {
    pub use super::{
        TimeSeries,
        StationarityTest, ADFTest, KPSSTest, PPTest,
        ARIMA, SARIMABuilder, ARIMAResults, SARIMAOrder,
        GARCH, ARCH, GARCHResults, GARCHOrder, GARCHDistribution,
        StateSpaceModel, KalmanFilter,
        DecompositionMethod, DecompositionResults, MovingAverageDecomposition, STLDecomposition, HodrickPrescottFilter, X12ARIMA, DecompositionExt,
        ForecastMetrics, IntervalMethod, PredictionInterval, PredictionIntervals, TimeSeriesCV,
        TSAResults, ResidualDiagnostics, ModelComparison,
    };
}