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

pub mod regression;
pub mod glm;
pub mod mixed;
pub mod robust;
pub mod nonparametric;

// Re-exports for convenience
pub use regression::{LinearModelBuilder, LinearRegressionResults, OLS, Ridge};
pub use glm::{GLM, GLMModelBuilder, GLMResults, Family, Link};
pub use mixed::{LinearMixedModelBuilder, LMMResults, EstimationMethod, RandomEffect, RandomCovariance};
pub use robust::{MEstimator, RobustRegressionResults, LossFunction, ScaleEstimator, TuningParameters, 
                LeastTrimmedSquares, SEstimator, MMEstimator};
pub use nonparametric::{KernelRegression, KernelRegressionResults, Kernel, BandwidthMethod,
                      LocalRegression, LocalRegressionResults, SmoothingSpline, SmoothingSplineResults};