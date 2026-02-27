//! Statistical models for StatOxide
//!
//! This crate provides a comprehensive suite of statistical models:
#![allow(missing_docs)]
//!
//! - **Linear Models**: OLS, Ridge, Lasso, Elastic Net
//! - **Generalized Linear Models**: Logistic, Poisson, Gamma regression
//! - **Mixed Effects Models**: Linear and generalized linear mixed models
//! - **Robust Statistics**: Robust regression and estimation
//! - **Nonparametric Methods**: Kernel regression, local regression, smoothing splines

pub mod glm;
pub mod mixed;
pub mod nonparametric;
pub mod regression;
pub mod robust;

// Re-exports for convenience
pub use glm::{Family, GLM, GLMModelBuilder, GLMResults, Link};
pub use mixed::{
    EstimationMethod, LMMResults, LinearMixedModelBuilder, RandomCovariance, RandomEffect,
};
pub use nonparametric::{
    BandwidthMethod, Kernel, KernelRegression, KernelRegressionResults, LocalRegression,
    LocalRegressionResults, SmoothingSpline, SmoothingSplineResults,
};
pub use regression::{LinearModelBuilder, LinearRegressionResults, OLS, Ridge};
pub use robust::{
    LeastTrimmedSquares, LossFunction, MEstimator, MMEstimator, RobustRegressionResults,
    SEstimator, ScaleEstimator, TuningParameters,
};
