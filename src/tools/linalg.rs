//! Linear algebra utilities using faer

use ndarray::{Array1, Array2};
use faer::{Mat, Faer, prelude::{Solver, SolverCore}};

/// Solve linear system A * x = b
pub fn solve(A: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let n = A.nrows();
    let m = A.ncols();
    
    if n != m {
        return Err(format!("Matrix must be square, got {}x{}", n, m));
    }
    
    if n != b.len() {
        return Err(format!("Dimension mismatch: A is {}x{}, b has length {}", n, m, b.len()));
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

/// Compute matrix inverse
pub fn inv(A: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = A.nrows();
    let m = A.ncols();
    
    if n != m {
        return Err(format!("Matrix must be square, got {}x{}", n, m));
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

/// Compute matrix multiplication: C = A * B
pub fn matmul(A: &Array2<f64>, B: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = A.nrows();
    let k = A.ncols();
    let m = B.ncols();
    
    if A.ncols() != B.nrows() {
        return Err(format!("Dimension mismatch: A is {}x{}, B is {}x{}", n, k, B.nrows(), m));
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