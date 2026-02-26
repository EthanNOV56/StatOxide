//! Linear algebra backend abstraction

#![allow(non_snake_case)]  // Allow mathematical notation (A, B, etc.)

use ndarray::{Array1, Array2};
use super::error::{LinalgError, Result};

/// Trait defining linear algebra operations for different backends
pub trait LinalgBackend: Send + Sync {
    /// Solve linear system A * x = b
    fn solve(&self, A: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>>;
    
    /// Compute matrix inverse
    fn inv(&self, A: &Array2<f64>) -> Result<Array2<f64>>;
    
    /// Compute matrix multiplication: C = A * B
    fn matmul(&self, A: &Array2<f64>, B: &Array2<f64>) -> Result<Array2<f64>>;
    
    /// Compute Cholesky decomposition (optional)
    fn cholesky(&self, _A: &Array2<f64>) -> Result<Array2<f64>> {
        Err(LinalgError::OperationFailed("Cholesky not implemented".to_string()))
    }
    
    /// Compute QR decomposition (optional)
    fn qr(&self, _A: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        Err(LinalgError::OperationFailed("QR not implemented".to_string()))
    }
    
    /// Compute singular value decomposition (optional)
    fn svd(&self, _A: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        Err(LinalgError::OperationFailed("SVD not implemented".to_string()))
    }
}

// ============================================================================
// Faer Backend
// ============================================================================

#[cfg(feature = "faer")]
/// Faer-based linear algebra backend
#[derive(Debug, Clone, Default)]
pub struct FaerBackend;

#[cfg(feature = "faer")]
impl LinalgBackend for FaerBackend {
    fn solve(&self, A: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        use faer::{Mat, prelude::Solve};
        
        let n = A.nrows();
        let m = A.ncols();
        
        if n != m {
            return Err(LinalgError::NotSquare { rows: n, cols: m });
        }
        
        if n != b.len() {
            return Err(LinalgError::DimensionMismatch(
                format!("A is {}x{}, b has length {}", n, m, b.len())
            ));
        }
        
        // Convert ndarray to faer Mat
        let mut A_faer = Mat::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                A_faer[(i, j)] = A[(i, j)];
            }
        }
        
        let mut b_faer = Mat::zeros(n, 1);
        for i in 0..n {
            b_faer[(i, 0)] = b[i];
        }
        
        // Solve using LU decomposition
        let lu = A_faer.partial_piv_lu();
        let x_faer = lu.solve(&b_faer);
        
        // Convert back to ndarray
        let mut x = Array1::zeros(n);
        for i in 0..n {
            x[i] = x_faer[(i, 0)];
        }
        
        Ok(x)
    }
    
    fn inv(&self, A: &Array2<f64>) -> Result<Array2<f64>> {
        use faer::{Mat, linalg::solvers::DenseSolveCore, prelude::Solve};
        
        let n = A.nrows();
        let m = A.ncols();
        
        if n != m {
            return Err(LinalgError::NotSquare { rows: n, cols: m });
        }
        
        // Convert ndarray to faer Mat
        let mut A_faer = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                A_faer[(i, j)] = A[(i, j)];
            }
        }
        
        // Compute inverse using LU decomposition
        let lu = A_faer.partial_piv_lu();
        let inv_faer = lu.inverse();
        
        // Convert back to ndarray
        let mut inv = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[(i, j)] = inv_faer[(i, j)];
            }
        }
        
        Ok(inv)
    }
    
    fn matmul(&self, A: &Array2<f64>, B: &Array2<f64>) -> Result<Array2<f64>> {
        use faer::Mat;
        
        let n = A.nrows();
        let k = A.ncols();
        let m = B.ncols();
        
        if A.ncols() != B.nrows() {
            return Err(LinalgError::DimensionMismatch(
                format!("A is {}x{}, B is {}x{}", n, k, B.nrows(), m)
            ));
        }
        
        // Convert to faer for multiplication
        let mut A_faer = Mat::zeros(n, k);
        for i in 0..n {
            for j in 0..k {
                A_faer[(i, j)] = A[(i, j)];
            }
        }
        
        let mut B_faer = Mat::zeros(k, m);
        for i in 0..k {
            for j in 0..m {
                B_faer[(i, j)] = B[(i, j)];
            }
        }
        
        let C_faer = &A_faer * &B_faer;
        
        // Convert back
        let mut C = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                C[(i, j)] = C_faer[(i, j)];
            }
        }
        
        Ok(C)
    }
}

// ============================================================================
// Ndarray-linalg Backend
// ============================================================================

#[cfg(feature = "ndarray-linalg")]
/// Ndarray-linalg backend using OpenBLAS/LAPACK
#[derive(Debug, Clone, Default)]
pub struct NdarrayLinalgBackend;

#[cfg(feature = "ndarray-linalg")]
impl LinalgBackend for NdarrayLinalgBackend {
    fn solve(&self, A: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        use ndarray_linalg::Solve;
        
        let n = A.nrows();
        let m = A.ncols();
        
        if n != m {
            return Err(LinalgError::NotSquare { rows: n, cols: m });
        }
        
        if n != b.len() {
            return Err(LinalgError::DimensionMismatch(
                format!("A is {}x{}, b has length {}", n, m, b.len())
            ));
        }
        
        A.solve(b).map_err(|e| LinalgError::OperationFailed(format!("Solve failed: {}", e)))
    }
    
    fn inv(&self, A: &Array2<f64>) -> Result<Array2<f64>> {
        use ndarray_linalg::Inverse;
        
        let n = A.nrows();
        let m = A.ncols();
        
        if n != m {
            return Err(LinalgError::NotSquare { rows: n, cols: m });
        }
        
        A.inv().map_err(|e| LinalgError::OperationFailed(format!("Inverse failed: {}", e)))
    }
    
    fn matmul(&self, A: &Array2<f64>, B: &Array2<f64>) -> Result<Array2<f64>> {
        let n = A.nrows();
        let k = A.ncols();
        let m = B.ncols();
        
        if A.ncols() != B.nrows() {
            return Err(LinalgError::DimensionMismatch(
                format!("A is {}x{}, B is {}x{}", n, k, B.nrows(), m)
            ));
        }
        
        Ok(A.dot(B))
    }
}

// ============================================================================
// Pure Rust Backend
// ============================================================================

#[cfg(feature = "pure-rust")]
/// Pure Rust backend (no external dependencies, slower)
#[derive(Debug, Clone, Default)]
pub struct PureRustBackend;

#[cfg(feature = "pure-rust")]
impl LinalgBackend for PureRustBackend {
    fn solve(&self, A: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        // Simple Gaussian elimination for small matrices
        // For production, consider implementing LU decomposition
        let n = A.nrows();
        let m = A.ncols();
        
        if n != m {
            return Err(LinalgError::NotSquare { rows: n, cols: m });
        }
        
        if n != b.len() {
            return Err(LinalgError::DimensionMismatch(
                format!("A is {}x{}, b has length {}", n, m, b.len())
            ));
        }
        
        // TODO: Implement Gaussian elimination or LU decomposition
        Err(LinalgError::OperationFailed("Pure Rust solver not yet implemented".to_string()))
    }
    
    fn inv(&self, A: &Array2<f64>) -> Result<Array2<f64>> {
        let n = A.nrows();
        let m = A.ncols();
        
        if n != m {
            return Err(LinalgError::NotSquare { rows: n, cols: m });
        }
        
        // TODO: Implement matrix inverse using LU decomposition
        Err(LinalgError::OperationFailed("Pure Rust inverse not yet implemented".to_string()))
    }
    
    fn matmul(&self, A: &Array2<f64>, B: &Array2<f64>) -> Result<Array2<f64>> {
        let n = A.nrows();
        let k = A.ncols();
        let m = B.ncols();
        
        if A.ncols() != B.nrows() {
            return Err(LinalgError::DimensionMismatch(
                format!("A is {}x{}, B is {}x{}", n, k, B.nrows(), m)
            ));
        }
        
        Ok(A.dot(B))
    }
}

// ============================================================================
// No Backend Available Error
// ============================================================================

#[cfg(not(any(feature = "faer", feature = "ndarray-linalg", feature = "pure-rust")))]
compile_error!("At least one backend feature must be enabled (faer, ndarray-linalg, or pure-rust)");