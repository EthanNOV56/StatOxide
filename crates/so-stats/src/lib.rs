//! Statistical functions and distributions for StatOxide
//!
//! This crate provides:
//! - Descriptive statistics (mean, variance, skewness, kurtosis, quantiles)
//! - Probability distributions (Normal, t, Chi-Square, F, etc.)
//! - Statistical tests (t-test, z-test, chi-square test, ANOVA)
//! - Correlation measures (Pearson, Spearman, Kendall)
//! - Statistical distances (KL divergence, Wasserstein distance)

#![allow(missing_docs)]

pub mod descriptive;
pub mod distributions;
pub mod tests;
pub mod correlation;
pub mod distance;

// Re-exports
pub use descriptive::*;
pub use distributions::*;
pub use tests::*;
pub use correlation::*;
pub use distance::*;