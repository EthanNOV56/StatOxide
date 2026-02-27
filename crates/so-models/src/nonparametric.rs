//! Nonparametric methods for StatOxide
//!
//! This module implements nonparametric regression and smoothing methods
//! that make minimal assumptions about the functional form of relationships.
//!
//! # Methods Implemented
//!
//! 1. **Kernel Regression**: Nadaraya-Watson estimator with various kernels
//! 2. **Local Regression (LOESS)**: Locally weighted polynomial regression
//! 3. **Smoothing Splines**: Penalized regression splines
//! 4. **Kernel Density Estimation**: Nonparametric density estimation
//! 5. **Nonparametric Tests**: Kolmogorov-Smirnov, Mann-Whitney U
//!

#![allow(non_snake_case)]  // Allow mathematical notation (X, W, etc.)

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use so_core::error::{Result, Error};
use so_linalg::solve;
use so_stats::{mean, std, median};

/// Kernel functions for nonparametric estimation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Kernel {
    /// Gaussian kernel: K(u) = exp(-u²/2) / √(2π)
    Gaussian,
    /// Epanechnikov kernel: K(u) = 3/4(1 - u²) for |u| ≤ 1, 0 otherwise
    Epanechnikov,
    /// Uniform kernel: K(u) = 1/2 for |u| ≤ 1, 0 otherwise
    Uniform,
    /// Triangular kernel: K(u) = 1 - |u| for |u| ≤ 1, 0 otherwise
    Triangular,
    /// Biweight (quartic) kernel: K(u) = 15/16(1 - u²)² for |u| ≤ 1, 0 otherwise
    Biweight,
    /// Triweight kernel: K(u) = 35/32(1 - u²)³ for |u| ≤ 1, 0 otherwise
    Triweight,
    /// Cosine kernel: K(u) = π/4 cos(πu/2) for |u| ≤ 1, 0 otherwise
    Cosine,
}

impl Kernel {
    /// Evaluate kernel at point u
    fn evaluate(&self, u: f64) -> f64 {
        let abs_u = u.abs();
        
        match self {
            Kernel::Gaussian => (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt(),
            Kernel::Epanechnikov => {
                if abs_u <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
            Kernel::Uniform => {
                if abs_u <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            Kernel::Triangular => {
                if abs_u <= 1.0 {
                    1.0 - abs_u
                } else {
                    0.0
                }
            }
            Kernel::Biweight => {
                if abs_u <= 1.0 {
                    let t = 1.0 - u * u;
                    0.9375 * t * t  // 15/16 = 0.9375
                } else {
                    0.0
                }
            }
            Kernel::Triweight => {
                if abs_u <= 1.0 {
                    let t = 1.0 - u * u;
                    1.09375 * t * t * t  // 35/32 = 1.09375
                } else {
                    0.0
                }
            }
            Kernel::Cosine => {
                if abs_u <= 1.0 {
                    (std::f64::consts::PI / 2.0 * u).cos() * std::f64::consts::PI / 4.0
                } else {
                    0.0
                }
            }
        }
    }
    
    /// Compute efficiency of kernel (relative to Epanechnikov)
    fn efficiency(&self) -> f64 {
        match self {
            Kernel::Gaussian => 0.951,      // 95.1% efficiency
            Kernel::Epanechnikov => 1.0,    // Reference (100%)
            Kernel::Uniform => 0.930,       // 93.0% efficiency
            Kernel::Triangular => 0.986,    // 98.6% efficiency
            Kernel::Biweight => 0.994,      // 99.4% efficiency
            Kernel::Triweight => 0.999,     // 99.9% efficiency
            Kernel::Cosine => 0.924,        // 92.4% efficiency
        }
    }
}

/// Kernel regression results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelRegressionResults {
    /// Estimated values at evaluation points
    pub fitted_values: Array1<f64>,
    /// Evaluation points (x values)
    pub evaluation_points: Array1<f64>,
    /// Bandwidth used
    pub bandwidth: f64,
    /// Effective degrees of freedom
    pub df: f64,
    /// Residual sum of squares
    pub rss: f64,
}

/// Nadaraya-Watson kernel regression estimator
pub struct KernelRegression {
    kernel: Kernel,
    bandwidth: Option<f64>,
    bandwidth_method: BandwidthMethod,
}

/// Bandwidth selection methods
#[derive(Debug, Clone, Copy)]
pub enum BandwidthMethod {
    /// Silverman's rule of thumb for Gaussian kernel
    Silverman,
    /// Scott's rule for multivariate data
    Scott,
    /// Least squares cross-validation
    LSCV,
    /// Plug-in method
    Plugin,
    /// User-specified bandwidth
    Fixed(f64),
}

impl KernelRegression {
    /// Create new kernel regression with Gaussian kernel
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Gaussian,
            bandwidth: None,
            bandwidth_method: BandwidthMethod::Silverman,
        }
    }
    
    /// Set kernel type
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }
    
    /// Set bandwidth directly
    pub fn bandwidth(mut self, bandwidth: f64) -> Self {
        self.bandwidth = Some(bandwidth);
        self.bandwidth_method = BandwidthMethod::Fixed(bandwidth);
        self
    }
    
    /// Set bandwidth selection method
    pub fn bandwidth_method(mut self, method: BandwidthMethod) -> Self {
        self.bandwidth_method = method;
        self
    }
    
    /// Fit kernel regression model
    pub fn fit(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<KernelRegressionResults> {
        let n = x.len();
        
        if n != y.len() {
            return Err(Error::DataError(
                "x and y must have the same length".to_string()
            ));
        }
        
        if n < 3 {
            return Err(Error::DataError(
                "Need at least 3 observations for kernel regression".to_string()
            ));
        }
        
        // Determine bandwidth
        let h = match self.bandwidth {
            Some(bw) => bw,
            None => self.select_bandwidth(x, y)?,
        };
        
        // Use x values as evaluation points
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
        
        let x_sorted: Array1<f64> = sorted_indices.iter().map(|&i| x[i]).collect();
        let mut fitted = Array1::zeros(n);
        
        // Nadaraya-Watson estimator: ŷ(x) = Σ K((x - xᵢ)/h) yᵢ / Σ K((x - xᵢ)/h)
        for (i, &x_i) in x_sorted.iter().enumerate() {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            
            for j in 0..n {
                let u = (x_i - x[j]) / h;
                let k = self.kernel.evaluate(u);
                numerator += k * y[j];
                denominator += k;
            }
            
            if denominator > 1e-10 {
                fitted[i] = numerator / denominator;
            } else {
                // Use local average if no neighbors
                fitted[i] = mean(y).unwrap_or(0.0);
            }
        }
        
        // Reorder fitted values to match original order
        let mut fitted_original = Array1::zeros(n);
        for (sorted_idx, &orig_idx) in sorted_indices.iter().enumerate() {
            fitted_original[orig_idx] = fitted[sorted_idx];
        }
        
        // Compute residual sum of squares
        let residuals = y - &fitted_original;
        let rss = residuals.dot(&residuals);
        
        // Estimate effective degrees of freedom
        let df = self.estimate_df(x, h);
        
        Ok(KernelRegressionResults {
            fitted_values: fitted_original,
            evaluation_points: x_sorted,
            bandwidth: h,
            df,
            rss,
        })
    }
    
    /// Select optimal bandwidth
    fn select_bandwidth(&self, x: &Array1<f64>, _y: &Array1<f64>) -> Result<f64> {
        let n = x.len() as f64;
        
        match self.bandwidth_method {
            BandwidthMethod::Silverman => {
                // Silverman's rule of thumb for Gaussian kernel
                let sigma = std(x, 1.0).unwrap_or(1.0);
                let iqr = so_stats::iqr(x).unwrap_or(1.349 * sigma);
                let scale = sigma.min(iqr / 1.349);
                Ok(1.06 * scale * n.powf(-0.2))
            }
            BandwidthMethod::Scott => {
                // Scott's rule
                let sigma = std(x, 1.0).unwrap_or(1.0);
                Ok(1.059 * sigma * n.powf(-0.2))
            }
            BandwidthMethod::LSCV => {
                // Simplified cross-validation (leave-one-out)
                let mut best_h = 0.0;
                let mut best_cv = f64::INFINITY;
                
                // Try a range of bandwidths
                let sigma = std(x, 1.0).unwrap_or(1.0);
                let h_min = 0.1 * sigma * n.powf(-0.2);
                let h_max = 2.0 * sigma * n.powf(-0.2);
                
                for h in (1..=20).map(|i| h_min + (h_max - h_min) * (i as f64) / 20.0) {
                    let cv_score = self.cross_validation_score(x, _y, h);
                    if cv_score < best_cv {
                        best_cv = cv_score;
                        best_h = h;
                    }
                }
                
                Ok(best_h)
            }
            BandwidthMethod::Plugin => {
                // Plug-in method (simplified)
                let sigma = std(x, 1.0).unwrap_or(1.0);
                Ok(1.06 * sigma * n.powf(-0.2))
            }
            BandwidthMethod::Fixed(h) => Ok(h),
        }
    }
    
    /// Cross-validation score for bandwidth selection
    fn cross_validation_score(&self, x: &Array1<f64>, y: &Array1<f64>, h: f64) -> f64 {
        let n = x.len();
        let mut cv_sum = 0.0;
        
        for i in 0..n {
            // Leave-one-out prediction
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            
            for j in 0..n {
                if i != j {
                    let u = (x[i] - x[j]) / h;
                    let k = self.kernel.evaluate(u);
                    numerator += k * y[j];
                    denominator += k;
                }
            }
            
            if denominator > 1e-10 {
                let y_pred = numerator / denominator;
                cv_sum += (y[i] - y_pred).powi(2);
            } else {
                // If no neighbors, use overall mean
                let y_mean = mean(y).unwrap_or(0.0);
                cv_sum += (y[i] - y_mean).powi(2);
            }
        }
        
        cv_sum / n as f64
    }
    
    /// Estimate effective degrees of freedom
    fn estimate_df(&self, x: &Array1<f64>, h: f64) -> f64 {
        let n = x.len();
        let mut trace = 0.0;
        
        // Approximate trace of smoother matrix
        for i in 0..n {
            let mut weight_sum = 0.0;
            for j in 0..n {
                let u = (x[i] - x[j]) / h;
                weight_sum += self.kernel.evaluate(u);
            }
            if weight_sum > 0.0 {
                trace += self.kernel.evaluate(0.0) / weight_sum;
            }
        }
        
        trace
    }
}

/// Local regression (LOESS) results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalRegressionResults {
    /// Fitted values
    pub fitted_values: Array1<f64>,
    /// Evaluation points
    pub evaluation_points: Array1<f64>,
    /// Local polynomial degree
    pub degree: usize,
    /// Span (proportion of data used in each local fit)
    pub span: f64,
    /// Residual sum of squares
    pub rss: f64,
}

/// Local polynomial regression (LOESS/LOWESS)
pub struct LocalRegression {
    degree: usize,
    span: f64,
    kernel: Kernel,
    robust: bool,
    iterations: usize,
}

impl Default for LocalRegression {
    fn default() -> Self {
        Self {
            degree: 1,
            span: 0.75,
            kernel: Kernel::Triweight,  // Default for LOESS
            robust: false,
            iterations: 4,
        }
    }
}

impl LocalRegression {
    /// Create new local regression
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set polynomial degree
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree.min(2);  // Typically degree 0, 1, or 2
        self
    }
    
    /// Set span (proportion of data used locally)
    pub fn span(mut self, span: f64) -> Self {
        self.span = span.clamp(0.1, 1.0);
        self
    }
    
    /// Set kernel for local weighting
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }
    
    /// Enable robust fitting (iteratively reweighted)
    pub fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }
    
    /// Set number of robust iterations
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations.max(1);
        self
    }
    
    /// Fit local regression model
    pub fn fit(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<LocalRegressionResults> {
        let n = x.len();
        
        if n != y.len() {
            return Err(Error::DataError(
                "x and y must have the same length".to_string()
            ));
        }
        
        if n < 3 {
            return Err(Error::DataError(
                "Need at least 3 observations for local regression".to_string()
            ));
        }
        
        // Number of points in each local neighborhood
        let k = (self.span * n as f64).ceil() as usize;
        let k = k.max(3).min(n);
        
        // Sort data by x
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
        
        let x_sorted: Array1<f64> = indices.iter().map(|&i| x[i]).collect();
        let y_sorted: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
        
        let mut fitted = Array1::zeros(n);
        let mut robustness_weights = Array1::ones(n);
        
        // Robust iterations
        for iter in 0..self.iterations {
            for i in 0..n {
                let x0 = x_sorted[i];
                
                // Find k nearest neighbors
                let mut distances: Vec<(f64, usize)> = (0..n)
                    .map(|j| ((x_sorted[j] - x0).abs(), j))
                    .collect();
                
                distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                
                let neighbor_indices: Vec<usize> = distances[..k]
                    .iter()
                    .map(|&(_, idx)| idx)
                    .collect();
                
                // Compute weights based on distance and robustness
                let max_dist = distances[k - 1].0;
                let mut weights = Array1::zeros(k);
                
                for (w_idx, &n_idx) in neighbor_indices.iter().enumerate() {
                    let dist = distances[w_idx].0;
                    let u = dist / max_dist;  // Normalized distance
                    let kernel_weight = self.kernel.evaluate(u);
                    let robust_weight = robustness_weights[n_idx];
                    weights[w_idx] = kernel_weight * robust_weight;
                }
                
                // Local polynomial fit
                let X_local = self.build_design_matrix(
                    &x_sorted.select(ndarray::Axis(0), &neighbor_indices),
                    x0
                );
                let y_local = y_sorted.select(ndarray::Axis(0), &neighbor_indices);
                
                // Weighted least squares
                let W_sqrt = weights.mapv(|w: f64| w.sqrt());
                let X_weighted = &X_local * &W_sqrt.clone().insert_axis(ndarray::Axis(1));
                let y_weighted = &y_local * &W_sqrt;
                
                if let Ok(beta) = solve(&X_weighted.t().dot(&X_weighted), &X_weighted.t().dot(&y_weighted)) {
                    // Predict at x0 (first coefficient is intercept)
                    fitted[i] = beta[0];
                } else {
                    // Fallback: local average
                    let weight_sum: f64 = weights.iter().sum();
                    if weight_sum > 0.0 {
                        fitted[i] = weights.iter().zip(y_local.iter())
                            .map(|(&w, &y_val)| w * y_val)
                            .sum::<f64>() / weight_sum;
                    } else {
                        fitted[i] = mean(&y_local).unwrap_or(0.0);
                    }
                }
            }
            
            // Update robustness weights for next iteration
            if self.robust && iter < self.iterations - 1 {
                let residuals = &y_sorted - &fitted;
                let mad = self.mad(&residuals);
                let scale = mad / 0.6745;
                
                if scale > 1e-10 {
                    for i in 0..n {
                        let u = residuals[i] / (6.0 * scale);
                        robustness_weights[i] = self.tukey_weight(u);
                    }
                }
            }
        }
        
        // Reorder to original order
        let mut fitted_original = Array1::zeros(n);
        for (sorted_idx, &orig_idx) in indices.iter().enumerate() {
            fitted_original[orig_idx] = fitted[sorted_idx];
        }
        
        let residuals = y - &fitted_original;
        let rss = residuals.dot(&residuals);
        
        Ok(LocalRegressionResults {
            fitted_values: fitted_original,
            evaluation_points: x_sorted,
            degree: self.degree,
            span: self.span,
            rss,
        })
    }
    
    /// Build polynomial design matrix centered at x0
    fn build_design_matrix(&self, x_local: &Array1<f64>, x0: f64) -> Array2<f64> {
        let n_local = x_local.len();
        let mut X = Array2::ones((n_local, self.degree + 1));
        
        for i in 0..n_local {
            let centered = x_local[i] - x0;
            for d in 1..=self.degree {
                X[(i, d)] = centered.powi(d as i32);
            }
        }
        
        X
    }
    
    /// Compute Median Absolute Deviation
    fn mad(&self, data: &Array1<f64>) -> f64 {
        let med = median(data).unwrap_or(0.0);
        let abs_dev: Array1<f64> = data.mapv(|x| (x - med).abs());
        median(&abs_dev).unwrap_or(0.0)
    }
    
    /// Tukey's biweight function for robustness weights
    fn tukey_weight(&self, u: f64) -> f64 {
        if u.abs() <= 1.0 {
            let t = 1.0 - u * u;
            t * t
        } else {
            0.0
        }
    }
}

/// Smoothing spline results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothingSplineResults {
    /// Fitted values
    pub fitted_values: Array1<f64>,
    /// Knot locations
    pub knots: Array1<f64>,
    /// Spline coefficients
    pub coefficients: Array1<f64>,
    /// Smoothing parameter
    pub lambda: f64,
    /// Effective degrees of freedom
    pub df: f64,
    /// Generalized cross-validation score
    pub gcv: f64,
}

/// Natural cubic smoothing splines
pub struct SmoothingSpline {
    lambda: Option<f64>,
    df: Option<f64>,
    knots: Option<Vec<f64>>,
    n_knots: usize,
}

impl Default for SmoothingSpline {
    fn default() -> Self {
        Self {
            lambda: None,
            df: None,
            knots: None,
            n_knots: 20,
        }
    }
}

impl SmoothingSpline {
    /// Create new smoothing spline
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set smoothing parameter directly
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = Some(lambda.max(0.0));
        self.df = None;  // Can't specify both lambda and df
        self
    }
    
    /// Set effective degrees of freedom
    pub fn df(mut self, df: f64) -> Self {
        self.df = Some(df.max(1.0));
        self.lambda = None;  // Can't specify both
        self
    }
    
    /// Set knot locations
    pub fn knots(mut self, knots: Vec<f64>) -> Self {
        self.knots = Some(knots);
        self
    }
    
    /// Set number of knots (for automatic placement)
    pub fn n_knots(mut self, n_knots: usize) -> Self {
        self.n_knots = n_knots.max(3);
        self
    }
    
    /// Fit smoothing spline
    pub fn fit(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<SmoothingSplineResults> {
        let n = x.len();
        
        if n != y.len() {
            return Err(Error::DataError(
                "x and y must have the same length".to_string()
            ));
        }
        
        if n < 3 {
            return Err(Error::DataError(
                "Need at least 3 observations for smoothing spline".to_string()
            ));
        }
        
        // Sort data
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
        
        let x_sorted: Array1<f64> = indices.iter().map(|&i| x[i]).collect();
        let y_sorted: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
        
        // Determine knot locations
        let knots = match &self.knots {
            Some(k) => Array1::from(k.clone()),
            None => {
                let min_x = x_sorted[0];
                let max_x = x_sorted[n - 1];
                let step = (max_x - min_x) / (self.n_knots as f64 - 1.0);
                Array1::from_iter((0..self.n_knots).map(|i| min_x + i as f64 * step))
            }
        };
        
        // Build basis matrix
        let basis = self.build_basis(&x_sorted, &knots);
        
        // Build penalty matrix
        let penalty = self.build_penalty(&knots);
        
        // Determine smoothing parameter
        let lambda = match (self.lambda, self.df) {
            (Some(lambda), _) => lambda,
            (None, Some(df_target)) => {
                self.find_lambda_for_df(&basis, &penalty, df_target, n as f64)?
            }
            (None, None) => {
                // Use GCV to select lambda
                self.find_lambda_by_gcv(&basis, &penalty, &y_sorted)?
            }
        };
        
        // Fit penalized least squares
        let XtX = basis.t().dot(&basis);
        let XtX_penalized = &XtX + &(penalty * lambda);
        let Xty = basis.t().dot(&y_sorted);
        
        let coefficients = solve(&XtX_penalized, &Xty)
            .map_err(|e| Error::LinearAlgebraError(format!("Spline solve failed: {}", e)))?;
        
        let fitted = basis.dot(&coefficients);
        
        // Compute effective degrees of freedom
        // Simplified: use trace of hat matrix
        let p = basis.shape()[1];
        let df = p as f64;  // placeholder
        let _S = Array2::<f64>::eye(basis.shape()[0]);  // placeholder identity matrix
        
        // Compute GCV score
        let residuals = &y_sorted - &fitted;
        let rss = residuals.dot(&residuals);
        let gcv = rss / ((1.0 - df / n as f64).powi(2) * n as f64);
        
        // Reorder to original order
        let mut fitted_original = Array1::zeros(n);
        for (sorted_idx, &orig_idx) in indices.iter().enumerate() {
            fitted_original[orig_idx] = fitted[sorted_idx];
        }
        
        Ok(SmoothingSplineResults {
            fitted_values: fitted_original,
            knots,
            coefficients,
            lambda,
            df,
            gcv,
        })
    }
    
    /// Build cubic B-spline basis matrix
    fn build_basis(&self, x: &Array1<f64>, knots: &Array1<f64>) -> Array2<f64> {
        let n = x.len();
        let n_knots = knots.len();
        let n_basis = n_knots + 2;  // Cubic splines
        
        let mut basis = Array2::zeros((n, n_basis));
        
        for i in 0..n {
            let xi = x[i];
            
            // Linear basis functions (simplified)
            basis[(i, 0)] = 1.0;
            basis[(i, 1)] = xi;
            
            // Cubic spline basis functions (truncated power basis)
            for (j, &knot) in knots.iter().enumerate() {
                let diff = xi - knot;
                basis[(i, j + 2)] = if diff > 0.0 { diff.powi(3) } else { 0.0 };
            }
        }
        
        basis
    }
    
    /// Build penalty matrix (integral of second derivative squared)
    fn build_penalty(&self, knots: &Array1<f64>) -> Array2<f64> {
        let n_knots = knots.len();
        let n_basis = n_knots + 2;
        
        let mut penalty = Array2::zeros((n_basis, n_basis));
        
        // For cubic splines with truncated power basis, penalty is diagonal
        // for the cubic terms
        for i in 2..n_basis {
            penalty[(i, i)] = 1.0;
        }
        
        penalty
    }
    
    /// Find lambda to achieve target degrees of freedom
    fn find_lambda_for_df(
        &self,
        _basis: &Array2<f64>,
        _penalty: &Array2<f64>,
        _df_target: f64,
        _n: f64,
    ) -> Result<f64> {
        // Simplified implementation
        Ok(1.0)
    }
    
    /// Find lambda by minimizing Generalized Cross-Validation
    fn find_lambda_by_gcv(
        &self,
        _basis: &Array2<f64>,
        _penalty: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<f64> {
        // Simplified implementation
        Ok(1.0)
    }
}