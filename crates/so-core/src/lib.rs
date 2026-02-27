//! Core data structures and formula parsing for StatOxide
//!
//! This crate provides the foundational types for statistical computing:
//!
//! - **Data structures**: Series, DataFrame for columnar data
//! - **Formula parsing**: R-style formula syntax for model specification
//! - **Error types**: Unified error handling across the ecosystem
#![allow(missing_docs)]

pub mod data;
pub mod formula;
pub mod error;

// Re-exports for convenience
pub use data::{DataFrame, Series};
pub use formula::Formula;
pub use error::{Error, Result};