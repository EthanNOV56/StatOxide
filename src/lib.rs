//! StatOxide: A high-performance statistical modeling library
//!
//! StatOxide provides a comprehensive suite of statistical models and methods
//! with a focus on performance, correctness, and ease of use. The library
//! features a dual API: a Rust-native API with chainable methods and
//! a Python/R-style formula API for data science workflows.
//!
//! # Architecture
//!
//! The library follows a modular architecture inspired by statsmodels:
//!
//! - `statoxide.base`: Core data structures and interfaces
//! - `statoxide.regression`: Linear regression models (OLS, Ridge, etc.)
//! - `statoxide.glm`: Generalized linear models
//! - `statoxide.tsa`: Time series analysis
//! - `statoxide.distributions`: Probability distributions
//! - `statoxide.nonparametric`: Nonparametric methods
//! - `statoxide.multivariate`: Multivariate analysis
//! - `statoxide.robust`: Robust statistics
//! - `statoxide.tools`: Utility functions and formula parsing
//! - `statoxide.api`: High-level Python/R-style API
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

// ============================================================================
// Core Modules (statsmodels-style architecture)
// ============================================================================

/// Core data structures and base classes
pub mod base {
    pub mod data;
}

/// Linear regression models
pub mod regression {
    pub mod linear;
}

/// Generalized linear models
pub mod glm;

/// Time series analysis
pub mod tsa;

/// Probability distributions
pub mod distributions;

/// Nonparametric methods
pub mod nonparametric;

/// Multivariate analysis
pub mod multivariate;

/// Robust statistics
pub mod robust;

/// Utility functions and tools
pub mod tools {
    pub mod formula;
    pub mod linalg;
    pub mod utils;
}

/// High-level API and Python bindings
pub mod api {
    #[cfg(feature = "python")]
    pub mod python_bindings;
}

/// Datasets for examples and testing
pub mod datasets;

// ============================================================================
// Re-exports for backward compatibility and convenience
// ============================================================================

// Re-export core types under their original names
pub use base::data::{DataFrame, Series};
pub use tools::formula::Formula;
pub use regression::linear::{LinearModelBuilder, LinearRegressionResults, OLS, Ridge};
pub use glm::*;

// Legacy module names for compatibility
pub mod data {
    pub use crate::base::data::*;
}

pub mod formula {
    pub use crate::tools::formula::*;
}

pub mod linear {
    pub use crate::regression::linear::*;
}

pub mod utils {
    pub use crate::tools::utils::*;
}

// Placeholder modules (to be implemented)
pub mod mixed;
pub mod stats {}

// ============================================================================
// Prelude module for convenient imports
// ============================================================================

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::base::data::*;
    pub use crate::tools::formula::*;
    pub use crate::regression::linear::*;
    pub use crate::glm::*;
    
    // To be added as modules are implemented
    // pub use crate::distributions::*;
    // pub use crate::tsa::*;
}

// ============================================================================
// Python module entry point
// ============================================================================

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module definition
/// This is the entry point for the Python extension module.
/// The module will be named `statoxide` in Python.
#[cfg(feature = "python")]
#[pymodule]
pub fn statoxide(_py: Python, m: &PyModule) -> PyResult<()> {
    crate::api::python_bindings::statoxide(_py, m)
}