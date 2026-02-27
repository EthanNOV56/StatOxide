//! StatOxide: High-performance statistical computing library
//!
//! This is the main entry point for the StatOxide library, providing a unified API
//! for all statistical computing functionality.
//!
//! # Overview
//!
//! StatOxide provides:
//! - **Core data structures**: `Series`, `DataFrame`, `Formula`
//! - **Statistical functions**: Descriptive statistics, distributions, tests
//! - **Statistical models**: Linear regression, GLM, mixed effects, time series
//! - **Time series analysis**: ARIMA, GARCH, forecasting, decomposition
//! - **Linear algebra**: Matrix operations, solvers, decompositions
//! - **Utilities**: Random generation, validation, numerical methods
//!
//! # Quick Start
//!
//! ```rust
//! use std::collections::HashMap;
//! use statoxide::{
//!     DataFrame, Series, Formula,
//!     stats::{mean, std, correlation},
//!     models::{GLM, GLMModelBuilder, Family, Link},
//!     tsa::{TimeSeries, ARIMA},
//!     ndarray::Array1,
//! };
//!
//! // Create a DataFrame
//! let mut columns = HashMap::new();
//! columns.insert("x".to_string(), Series::new("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0])));
//! columns.insert("y".to_string(), Series::new("y", Array1::from_vec(vec![2.0, 4.0, 5.0, 4.0, 5.0])));
//! let df = DataFrame::from_series(columns).unwrap();
//!
//! // Parse a formula
//! let formula = Formula::parse("y ~ x + x^2").unwrap();
//!
//! // Compute statistics
//! let x_series = df.column("x").unwrap();
//! let x_data = x_series.data().to_owned(); // Convert view to owned array
//! let avg = statoxide::stats::mean(&x_data).unwrap();
//! println!("Mean of x: {}", avg);
//! ```
//!
//! # Modules
//!
//! - [`core`] - Core data structures and formula parsing
//! - [`models`] - Statistical models (regression, GLM, mixed effects, etc.)
//! - [`stats`] - Statistical functions and tests
//! - [`tsa`] - Time series analysis
//! - [`linalg`] - Linear algebra utilities
//! - [`utils`] - General utilities
//!

// Core module re-exports (top-level for convenience)
pub use so_core::*;

// Models module
pub mod models {
    // Re-export the entire so_models public API
    pub use so_models::*;
}

// Statistics module
pub mod stats {
    // Re-export the entire so_stats public API
    pub use so_stats::*;
}

// Time series analysis module
pub mod tsa {
    // Re-export the entire so_tsa public API
    pub use so_tsa::*;
}

// Linear algebra module
pub mod linalg {
    // Re-export the entire so_linalg public API
    pub use so_linalg::*;
}

// Utilities module
pub mod utils {
    // Re-export the entire so_utils public API
    pub use so_utils::*;
}

// Re-export ndarray for convenience
pub use ndarray;

// Convenience re-exports of most commonly used types
pub use models::{GLM, GLMModelBuilder, GLMResults, Family, Link};
pub use stats::{mean, std, correlation, variance};
pub use tsa::{TimeSeries, ARIMA, GARCH};
pub use linalg::{solve, inv, matmul};
pub use utils::{random_normal_array, standardize, softmax};

// Predule for easy importing
pub mod prelude {
    pub use super::{
        // Core types
        DataFrame, Series, Formula, Error, Result,
        // Models
        GLM, GLMModelBuilder, GLMResults, Family, Link,
        // Statistics
        mean, std, variance, correlation,
        // Time series
        TimeSeries, ARIMA, GARCH,
        // Linear algebra
        solve, inv, matmul,
        // Utilities
        random_normal_array, standardize, softmax,
    };
}

// Module documentation
pub mod core {
    //! Core data structures and formula parsing
    pub use so_core::*;
}

// Version information
/// Returns the version of StatOxide
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}