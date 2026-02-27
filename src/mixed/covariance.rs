//! Covariance structures for mixed models

use std::collections::HashMap;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Type of covariance structure
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CovarianceType {
    /// Independent random effects (diagonal matrix)
    Diagonal,
    /// Compound symmetry (equal variance, equal covariance)
    CompoundSymmetry,
    /// Autoregressive of order 1
    AR1,
    /// Toeplitz (stationary covariance)
    Toeplitz,
    /// Unstructured (full covariance matrix)
    Unstructured,
    /// Identity matrix (scaled)
    Identity,
    /// Custom user-defined structure
    Custom,
}

/// Covariance structure specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovarianceStructure {
    /// Type of covariance structure
    pub cov_type: CovarianceType,
    /// Parameters for the covariance structure
    pub parameters: Vec<f64>,
    /// Dimension of the covariance matrix
    pub dimension: Option<usize>,
    /// Additional constraints or metadata
    pub constraints: HashMap<String, f64>,
}

impl Default for CovarianceStructure {
    fn default() -> Self {
        Self {
            cov_type: CovarianceType::Unstructured,
            parameters: Vec::new(),
            dimension: None,
            constraints: HashMap::new(),
        }
    }
}

impl CovarianceStructure {
    /// Create a diagonal covariance structure
    pub fn diagonal(dimension: usize) -> Self {
        Self {
            cov_type: CovarianceType::Diagonal,
            parameters: vec![1.0; dimension], // variances
            dimension: Some(dimension),
            constraints: HashMap::new(),
        }
    }
    
    /// Create an identity covariance structure (scaled)
    pub fn identity(dimension: usize, scale: f64) -> Self {
        Self {
            cov_type: CovarianceType::Identity,
            parameters: vec![scale],
            dimension: Some(dimension),
            constraints: HashMap::new(),
        }
    }
    
    /// Create compound symmetry structure
    pub fn compound_symmetry(dimension: usize, variance: f64, covariance: f64) -> Self {
        Self {
            cov_type: CovarianceType::CompoundSymmetry,
            parameters: vec![variance, covariance],
            dimension: Some(dimension),
            constraints: HashMap::new(),
        }
    }
    
    /// Create AR1 structure
    pub fn ar1(dimension: usize, variance: f64, rho: f64) -> Self {
        Self {
            cov_type: CovarianceType::AR1,
            parameters: vec![variance, rho],
            dimension: Some(dimension),
            constraints: HashMap::new(),
        }
    }
    
    /// Create Toeplitz structure
    pub fn toeplitz(dimension: usize, params: Vec<f64>) -> Self {
        Self {
            cov_type: CovarianceType::Toeplitz,
            parameters: params,
            dimension: Some(dimension),
            constraints: HashMap::new(),
        }
    }
    
    /// Create unstructured covariance
    pub fn unstructured(dimension: usize) -> Self {
        let n_params = dimension * (dimension + 1) / 2;
        Self {
            cov_type: CovarianceType::Unstructured,
            parameters: vec![0.0; n_params], // will be estimated
            dimension: Some(dimension),
            constraints: HashMap::new(),
        }
    }
    
    /// Create custom covariance structure
    pub fn custom(params: Vec<f64>, dimension: usize) -> Self {
        Self {
            cov_type: CovarianceType::Custom,
            parameters: params,
            dimension: Some(dimension),
            constraints: HashMap::new(),
        }
    }
    
    /// Build covariance matrix from parameters
    pub fn build_matrix(&self) -> Result<Array2<f64>, String> {
        let dim = self.dimension.ok_or("Covariance dimension not specified".to_string())?;
        
        match self.cov_type {
            CovarianceType::Diagonal => {
                if self.parameters.len() != dim {
                    return Err(format!("Expected {} variance parameters, got {}", 
                        dim, self.parameters.len()));
                }
                let mut matrix = Array2::zeros((dim, dim));
                for i in 0..dim {
                    matrix[(i, i)] = self.parameters[i];
                }
                Ok(matrix)
            }
            
            CovarianceType::Identity => {
                if self.parameters.len() != 1 {
                    return Err(format!("Expected 1 scale parameter, got {}", 
                        self.parameters.len()));
                }
                let scale = self.parameters[0];
                Ok(Array2::eye(dim) * scale)
            }
            
            CovarianceType::CompoundSymmetry => {
                if self.parameters.len() != 2 {
                    return Err(format!("Expected 2 parameters (variance, covariance), got {}", 
                        self.parameters.len()));
                }
                let variance = self.parameters[0];
                let covariance = self.parameters[1];
                let mut matrix = Array2::zeros((dim, dim));
                for i in 0..dim {
                    matrix[(i, i)] = variance;
                    for j in i+1..dim {
                        matrix[(i, j)] = covariance;
                        matrix[(j, i)] = covariance;
                    }
                }
                Ok(matrix)
            }
            
            CovarianceType::AR1 => {
                if self.parameters.len() != 2 {
                    return Err(format!("Expected 2 parameters (variance, rho), got {}", 
                        self.parameters.len()));
                }
                let variance = self.parameters[0];
                let rho = self.parameters[1];
                let mut matrix = Array2::zeros((dim, dim));
                for i in 0..dim {
                    for j in 0..dim {
                        matrix[(i, j)] = variance * rho.powf((i as i32 - j as i32).abs() as f64);
                    }
                }
                Ok(matrix)
            }
            
            CovarianceType::Toeplitz => {
                // Parameters are covariances for lags 0..dim-1
                if self.parameters.len() != dim {
                    return Err(format!("Expected {} parameters for Toeplitz, got {}", 
                        dim, self.parameters.len()));
                }
                let mut matrix = Array2::zeros((dim, dim));
                for i in 0..dim {
                    for j in 0..dim {
                        let lag = (i as i32 - j as i32).abs() as usize;
                        matrix[(i, j)] = self.parameters[lag];
                    }
                }
                Ok(matrix)
            }
            
            CovarianceType::Unstructured => {
                let n_expected = dim * (dim + 1) / 2;
                if self.parameters.len() != n_expected {
                    return Err(format!("Expected {} parameters for unstructured covariance, got {}", 
                        n_expected, self.parameters.len()));
                }
                // Parameters are lower triangle in column-major order
                let mut matrix = Array2::zeros((dim, dim));
                let mut idx = 0;
                for j in 0..dim {
                    for i in j..dim {
                        let val = self.parameters[idx];
                        matrix[(i, j)] = val;
                        matrix[(j, i)] = val;
                        idx += 1;
                    }
                }
                Ok(matrix)
            }
            
            CovarianceType::Custom => {
                // For custom structure, assume parameters fill the matrix in row-major order
                if self.parameters.len() != dim * dim {
                    return Err(format!("Expected {} parameters for custom {}x{} matrix, got {}", 
                        dim*dim, dim, dim, self.parameters.len()));
                }
                let mut matrix = Array2::zeros((dim, dim));
                for i in 0..dim {
                    for j in 0..dim {
                        matrix[(i, j)] = self.parameters[i * dim + j];
                    }
                }
                Ok(matrix)
            }
        }
    }
    
    /// Get number of free parameters
    pub fn n_free_parameters(&self) -> Result<usize, String> {
        let dim = self.dimension.ok_or("Covariance dimension not specified".to_string())?;
        
        match self.cov_type {
            CovarianceType::Diagonal => Ok(dim),
            CovarianceType::Identity => Ok(1),
            CovarianceType::CompoundSymmetry => Ok(2),
            CovarianceType::AR1 => Ok(2),
            CovarianceType::Toeplitz => Ok(dim),
            CovarianceType::Unstructured => Ok(dim * (dim + 1) / 2),
            CovarianceType::Custom => Ok(self.parameters.len()),
        }
    }
    
    /// Check if covariance matrix is positive definite
    pub fn is_positive_definite(&self) -> Result<bool, String> {
        let matrix = self.build_matrix()?;
        
        // Simple check: all eigenvalues > 0
        // For now, just check diagonal is positive
        for i in 0..matrix.nrows() {
            if matrix[(i, i)] <= 0.0 {
                return Ok(false);
            }
        }
        
        // TODO: Implement proper eigenvalue check
        Ok(true)
    }
    
    /// Constrain parameters (e.g., fix certain parameters)
    pub fn with_constraint(mut self, name: &str, value: f64) -> Self {
        self.constraints.insert(name.to_string(), value);
        self
    }
}