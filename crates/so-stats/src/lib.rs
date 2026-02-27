//! Statistical functions and distributions for StatOxide
//!
//! This crate provides:
//! - Descriptive statistics (mean, variance, skewness, kurtosis, quantiles)
//! - Probability distributions (Normal, t, Chi-Square, F, etc.)
//! - Statistical tests (t-test, z-test, chi-square test, ANOVA)
//! - Correlation measures (Pearson, Spearman, Kendall)
//! - Statistical distances (KL divergence, Wasserstein distance)

#![allow(missing_docs)]

pub mod correlation;
pub mod descriptive;
pub mod distance;
pub mod distributions;
pub mod tests;

// Re-exports
pub use correlation::*;
pub use descriptive::*;
pub use distance::*;
pub use distributions::*;
pub use tests::*;
