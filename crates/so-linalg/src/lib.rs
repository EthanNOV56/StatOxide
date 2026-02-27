//! Linear algebra middleware providing unified interface to multiple backends
//!
//! This crate provides a common interface for linear algebra operations,
//! with support for multiple backends (faer, ndarray-linalg, etc.).
//!
//! # Usage
//!
//! ```rust
//! use so_linalg::{solve, inv, matmul};
//! use ndarray::{arr1, arr2};
//!
//! // Solve linear system
//! let A = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
//! let b = arr1(&[5.0, 6.0]);
//! let x = solve(&A, &b).unwrap();
//!
//! // Matrix inverse
//! let inv_A = inv(&A).unwrap();
//!
//! // Matrix multiplication
//! let B = arr2(&[[2.0, 0.0], [1.0, 2.0]]);
//! let C = matmul(&A, &B).unwrap();
//! ```
//!
//! # Backends
//!
//! The crate supports multiple backends selectable via Cargo features:
//! - `faer` (default): Uses the faer library for high-performance linear algebra
//! - `ndarray-linalg`: Uses ndarray-linalg with OpenBLAS/LAPACK
//! - `pure-rust`: Pure Rust implementation (slower but no external dependencies)
//!
//! # Advanced Usage
//!
//! For more control, you can use backends directly:
//! ```rust
//! use so_linalg::backend::{LinalgBackend, FaerBackend};
//! use ndarray::{arr1, arr2};
//!
//! // Create example matrices
//! let A = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
//! let b = arr1(&[5.0, 6.0]);
//! 
//! let backend = FaerBackend::default();
//! let result = backend.solve(&A, &b);
//! ```

#![warn(missing_docs)]
#![allow(non_snake_case)]  // Allow mathematical notation (A, B, etc.)

pub mod backend;
pub mod error;

// Re-exports for convenience
pub use backend::{LinalgBackend, FaerBackend};
pub use error::{LinalgError, Result};

// ============================================================================
// Global Backend Selection
// ============================================================================

/// Get the default linear algebra backend
///
/// The backend is selected based on enabled Cargo features:
/// - `faer` (default): FaerBackend
/// - `ndarray-linalg`: NdarrayLinalgBackend
/// - `pure-rust`: PureRustBackend
pub fn default_backend() -> impl LinalgBackend {
    // Feature-based backend selection
    #[cfg(feature = "faer")]
    {
        backend::FaerBackend::default()
    }
    
    #[cfg(all(not(feature = "faer"), feature = "ndarray-linalg"))]
    {
        backend::NdarrayLinalgBackend::default()
    }
    
    #[cfg(all(not(feature = "faer"), not(feature = "ndarray-linalg"), feature = "pure-rust"))]
    {
        backend::PureRustBackend::default()
    }
    
    #[cfg(not(any(feature = "faer", feature = "ndarray-linalg", feature = "pure-rust")))]
    {
        // Default to faer if no features specified
        backend::FaerBackend::default()
    }
}

// ============================================================================
// Convenience Functions (using default backend)
// ============================================================================

/// Solve linear system A * x = b using the default backend
pub fn solve(A: &ndarray::Array2<f64>, b: &ndarray::Array1<f64>) -> Result<ndarray::Array1<f64>> {
    default_backend().solve(A, b)
}

/// Compute matrix inverse using the default backend
pub fn inv(A: &ndarray::Array2<f64>) -> Result<ndarray::Array2<f64>> {
    default_backend().inv(A)
}

/// Compute matrix multiplication: C = A * B using the default backend
pub fn matmul(A: &ndarray::Array2<f64>, B: &ndarray::Array2<f64>) -> Result<ndarray::Array2<f64>> {
    default_backend().matmul(A, B)
}

// ============================================================================
// Optional Backend Implementations (feature-gated)
// ============================================================================

// TODO: Implement these backends
// #[cfg(feature = "ndarray-linalg")]
// mod ndarray_linalg_backend;
// 
// #[cfg(feature = "pure-rust")]
// mod pure_rust_backend;