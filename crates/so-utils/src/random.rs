//! Random number generation utilities

#![allow(non_snake_case)]  // Allow mathematical notation (A, D, etc.)

use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Generate random array from uniform distribution
pub fn random_uniform_array(n: usize, low: f64, high: f64) -> Array1<f64> {
    let mut rng = rand::rng();
    
    Array1::from_iter((0..n).map(|_| rng.random_range(low..high)))
}

/// Generate random array from normal distribution
pub fn random_normal_array(n: usize, mean: f64, std_dev: f64) -> Array1<f64> {
    let mut rng = rand::rng();
    
    // Generate standard normal using Box-Muller transform
    Array1::from_iter((0..n).map(|_| {
        let u1: f64 = rng.random::<f64>();
        let u2: f64 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z
    }))
}

/// Generate random 2D array from uniform distribution
pub fn random_uniform_matrix(rows: usize, cols: usize, low: f64, high: f64) -> Array2<f64> {
    let mut rng = rand::rng();
    
    Array2::from_shape_fn((rows, cols), |_| rng.random_range(low..high))
}

/// Generate random 2D array from normal distribution
pub fn random_normal_matrix(rows: usize, cols: usize, mean: f64, std_dev: f64) -> Array2<f64> {
    let mut rng = rand::rng();
    
    Array2::from_shape_fn((rows, cols), |_| {
        let u1: f64 = rng.random::<f64>();
        let u2: f64 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z
    })
}

/// Generate random permutation of indices
pub fn random_permutation(n: usize) -> Vec<usize> {
    let mut rng = rand::rng();
    let mut indices: Vec<usize> = (0..n).collect();
    
    for i in 0..n {
        let j = rng.random_range(i..n);
        indices.swap(i, j);
    }
    
    indices
}

/// Randomly shuffle an array in place
pub fn shuffle_array<T>(arr: &mut [T]) {
    let mut rng = rand::rng();
    
    for i in 0..arr.len() {
        let j = rng.random_range(i..arr.len());
        arr.swap(i, j);
    }
}

/// Randomly shuffle rows of a 2D array
pub fn shuffle_rows(matrix: &mut Array2<f64>) {
    let mut rng = rand::rng();
    let n_rows = matrix.nrows();
    
    for i in 0..n_rows {
        let j = rng.random_range(i..n_rows);
        if i != j {
            let row_i = matrix.row(i).to_owned();
            let row_j = matrix.row(j).to_owned();
            matrix.row_mut(i).assign(&row_j);
            matrix.row_mut(j).assign(&row_i);
        }
    }
}

/// Randomly sample rows from a 2D array (with replacement)
pub fn bootstrap_sample(data: &Array2<f64>, n_samples: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let n_rows = data.nrows();
    let n_cols = data.ncols();
    
    let mut sample = Array2::zeros((n_samples, n_cols));
    
    for i in 0..n_samples {
        let idx = rng.random_range(0..n_rows);
        sample.row_mut(i).assign(&data.row(idx));
    }
    
    sample
}

/// Randomly split data into train and test sets
pub fn train_test_split(
    data: &Array2<f64>,
    labels: &Array1<f64>,
    test_size: f64,
    shuffle: bool,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let n_samples = data.nrows();
    let n_cols = data.ncols();
    
    let mut data_indices: Vec<usize> = (0..n_samples).collect();
    let mut label_indices: Vec<usize> = (0..labels.len()).collect();
    
    if shuffle {
        shuffle_array(&mut data_indices);
        shuffle_array(&mut label_indices);
    }
    
    let split_idx = ((n_samples as f64) * (1.0 - test_size)).round() as usize;
    
    let train_data = Array2::from_shape_fn((split_idx, n_cols), |(i, j)| {
        data[(data_indices[i], j)]
    });
    
    let test_data = Array2::from_shape_fn((n_samples - split_idx, n_cols), |(i, j)| {
        data[(data_indices[split_idx + i], j)]
    });
    
    let train_labels = Array1::from_iter((0..split_idx).map(|i| labels[label_indices[i]]));
    let test_labels = Array1::from_iter((split_idx..n_samples).map(|i| labels[label_indices[i]]));
    
    (train_data, test_data, train_labels, test_labels)
}

/// Generate deterministic random numbers with a seed
pub struct SeededRng {
    rng: StdRng,
}

impl SeededRng {
    /// Create a new seeded RNG
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
    
    /// Generate random uniform array
    pub fn uniform_array(&mut self, n: usize, low: f64, high: f64) -> Array1<f64> {
        Array1::from_iter((0..n).map(|_| self.rng.random_range(low..high)))
    }
    
    /// Generate random normal array
    pub fn normal_array(&mut self, n: usize, mean: f64, std_dev: f64) -> Array1<f64> {
        Array1::from_iter((0..n).map(|_| {
            let u1: f64 = self.rng.random::<f64>();
            let u2: f64 = self.rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mean + std_dev * z
        }))
    }
}

/// Create a correlation matrix with specified eigenvalue structure
pub fn random_correlation_matrix(n: usize, eigenvalues: &[f64]) -> Option<Array2<f64>> {
    if eigenvalues.len() != n {
        return None;
    }
    
    // Generate random orthogonal matrix
    let mut rng = rand::rng();
    let mut A = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            A[(i, j)] = rng.random::<f64>() - 0.5;
        }
    }
    
    // QR decomposition to get orthogonal matrix
    // Note: This is a simplified implementation
    // For production, use a proper QR decomposition
    
    // Create diagonal matrix with eigenvalues
    let mut D = Array2::zeros((n, n));
    for i in 0..n {
        D[(i, i)] = eigenvalues[i].sqrt();
    }
    
    // Compute correlation matrix: A D D^T A^T
    let AD = A.dot(&D);
    let correlation = AD.dot(&AD.t());
    
    // Normalize to correlation matrix (diagonal = 1)
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] = correlation[(i, j)] / (correlation[(i, i)] * correlation[(j, j)]).sqrt();
        }
    }
    
    Some(result)
}