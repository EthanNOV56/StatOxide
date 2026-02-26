//! StatOxide: A high-performance statistical modeling library
//!
//! StatOxide provides a comprehensive suite of statistical models and methods
//! with a focus on performance, correctness, and ease of use. The library
//! features a dual API: a Rust-native API with chainable methods and
//! a Python/R-style formula API for data science workflows.
//!
//! # Key Features
//!
//! - **Linear Models**: OLS, Ridge, Lasso, ElasticNet
//! - **Generalized Linear Models**: Logistic, Poisson, Gamma regression
//! - **Time Series**: ARIMA, GARCH, state space models
//! - **Mixed Effects**: Linear and generalized linear mixed models
//! - **Survival Analysis**: Cox proportional hazards, Kaplan-Meier
//! - **Nonparametric**: Kernel regression, local regression
//! - **Distributions**: Comprehensive probability distributions
//! - **Statistical Tests**: t-test, ANOVA, chi-square, etc.
//!
//! # Design Philosophy
//!
//! 1. **Performance**: Leverage Rust's zero-cost abstractions and SIMD
//! 2. **Correctness**: Strong typing and extensive testing
//! 3. **Interoperability**: Seamless integration with Python/R ecosystems
//! 4. **Usability**: Clean API design with sensible defaults

pub mod data;
pub mod distributions;
pub mod formula;
pub mod glm;
pub mod linear;
pub mod mixed;
pub mod nonparametric;
pub mod stats;
pub mod timeseries;
pub mod utils;

// Re-exports for common usage
pub use data::*;
pub use distributions::*;
pub use formula::*;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::data::*;
    pub use crate::distributions::*;
    pub use crate::formula::*;
    pub use crate::linear::*;
    pub use crate::glm::*;
    pub use crate::stats::*;
}

// Python module entry point
#[cfg(feature = "python")]
mod python_bindings;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module definition
/// This is the entry point for the Python extension module.
/// The module will be named `statoxide` in Python.
#[cfg(feature = "python")]
#[pymodule]
pub fn statoxide(_py: Python, m: &PyModule) -> PyResult<()> {
    crate::python_bindings::statoxide(_py, m)
}