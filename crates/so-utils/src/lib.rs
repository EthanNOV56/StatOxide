//! Utility functions and types for StatOxide
//!
//! This crate provides general-purpose utilities for statistical computing,
//! including error types, numerical utilities, and helper functions.

#![allow(missing_docs)]

pub mod error;
pub mod numerical;
pub mod random;
pub mod validation;

// Re-exports
pub use error::*;
pub use numerical::*;
pub use random::*;
pub use validation::*;
