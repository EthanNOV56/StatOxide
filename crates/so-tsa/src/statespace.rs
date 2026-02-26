//! State space models and Kalman filter
//!
//! This module implements state space models for time series analysis,
//! including the Kalman filter for estimation and forecasting.
//!
//! # State Space Representation
//!
//! General linear Gaussian state space model:
//!
//! Observation equation: yₜ = Zₜ αₜ + εₜ, εₜ ∼ N(0, Hₜ)
//! State equation:     αₜ = Tₜ αₜ₋₁ + Rₜ ηₜ, ηₜ ∼ N(0, Qₜ)
//!
//! where:
//! - yₜ: observed time series
//! - αₜ: unobserved state vector
//! - Zₜ: observation matrix
//! - Tₜ: transition matrix
//! - Rₜ: selection matrix for state disturbances
//! - Hₜ: observation covariance matrix
//! - Qₜ: state disturbance covariance matrix
//!
//! # Common Models
//!
//! 1. **Local Level Model**: yₜ = μₜ + εₜ, μₜ = μₜ₋₁ + ηₜ
//! 2. **Local Linear Trend**: yₜ = μₜ + εₜ, μₜ = μₜ₋₁ + νₜ₋₁ + ηₜ, νₜ = νₜ₋₁ + ζₜ
//! 3. **Basic Structural Model**: Adds seasonal components
//! 4. **ARMA in State Space**: Any ARMA model can be represented in state space

use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use so_core::error::Result;
use so_linalg;

/// State space model specification
#[derive(Debug, Clone)]
pub struct StateSpaceModel {
    /// Observation matrix Z (n_obs × n_states)
    pub observation_matrix: Array2<f64>,
    /// Transition matrix T (n_states × n_states)
    pub transition_matrix: Array2<f64>,
    /// Selection matrix R (n_states × n_disturbances)
    pub selection_matrix: Array2<f64>,
    /// Observation covariance H (n_obs × n_obs)
    pub observation_cov: Array2<f64>,
    /// State disturbance covariance Q (n_disturbances × n_disturbances)
    pub state_cov: Array2<f64>,
    /// Initial state mean α₀ (n_states)
    pub initial_state_mean: Array1<f64>,
    /// Initial state covariance P₀ (n_states × n_states)
    pub initial_state_cov: Array2<f64>,
}

/// Kalman filter results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilterResults {
    /// Filtered state means (n_timesteps × n_states)
    pub filtered_state_means: Array2<f64>,
    /// Filtered state covariances (n_timesteps × n_states × n_states)
    pub filtered_state_covs: Array3<f64>,
    /// Predicted state means (n_timesteps × n_states)
    pub predicted_state_means: Array2<f64>,
    /// Predicted state covariances (n_timesteps × n_states × n_states)
    pub predicted_state_covs: Array3<f64>,
    /// Innovations (prediction errors)
    pub innovations: Array1<f64>,
    /// Innovation variances
    pub innovation_variances: Array1<f64>,
    /// Kalman gains (n_timesteps × n_states × n_obs)
    pub kalman_gains: Array3<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
}

impl StateSpaceModel {
    /// Create local level model (random walk plus noise)
    pub fn local_level(obs_var: f64, level_var: f64) -> Self {
        // y_t = μ_t + ε_t, ε_t ~ N(0, σ_ε²)
        // μ_t = μ_{t-1} + η_t, η_t ~ N(0, σ_η²)
        
        let observation_matrix = ndarray::array![[1.0]];
        let transition_matrix = ndarray::array![[1.0]];
        let selection_matrix = ndarray::array![[1.0]];
        let observation_cov = ndarray::array![[obs_var]];
        let state_cov = ndarray::array![[level_var]];
        let initial_state_mean = ndarray::array![0.0];
        let initial_state_cov = ndarray::array![[1e6]]; // Diffuse prior
        
        Self {
            observation_matrix,
            transition_matrix,
            selection_matrix,
            observation_cov,
            state_cov,
            initial_state_mean,
            initial_state_cov,
        }
    }
    
    /// Create local linear trend model
    pub fn local_linear_trend(obs_var: f64, level_var: f64, slope_var: f64) -> Self {
        // y_t = μ_t + ε_t
        // μ_t = μ_{t-1} + ν_{t-1} + η_t
        // ν_t = ν_{t-1} + ζ_t
        
        let observation_matrix = ndarray::array![[1.0, 0.0]];
        let transition_matrix = ndarray::array![[1.0, 1.0], [0.0, 1.0]];
        let selection_matrix = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let observation_cov = ndarray::array![[obs_var]];
        let state_cov = ndarray::array![[level_var, 0.0], [0.0, slope_var]];
        let initial_state_mean = ndarray::array![0.0, 0.0];
        let initial_state_cov = ndarray::array![[1e6, 0.0], [0.0, 1e6]];
        
        Self {
            observation_matrix,
            transition_matrix,
            selection_matrix,
            observation_cov,
            state_cov,
            initial_state_mean,
            initial_state_cov,
        }
    }
    
    /// Create ARMA(p, q) model in state space form
    pub fn arma(ar_coef: &[f64], ma_coef: &[f64], sigma2: f64) -> Self {
        let p = ar_coef.len();
        let q = ma_coef.len();
        let r = p.max(q + 1);
        let n_states = r;
        
        // Build transition matrix (companion form)
        let mut transition = Array2::zeros((n_states, n_states));
        
        if p > 0 {
            // First row contains AR coefficients
            for j in 0..p {
                transition[(0, j)] = ar_coef[j];
            }
        }
        
        // Sub-diagonal of ones
        for i in 1..n_states {
            transition[(i, i - 1)] = 1.0;
        }
        
        // Observation matrix
        let mut observation = Array1::zeros(n_states);
        observation[0] = 1.0;
        if q > 0 {
            // Include MA coefficients
            for j in 0..q.min(n_states - 1) {
                observation[j + 1] = ma_coef[j];
            }
        }
        let observation_matrix = observation.insert_axis(ndarray::Axis(0));
        
        // Selection matrix (for state disturbances)
        let mut selection = Array2::zeros((n_states, 1));
        selection[(0, 0)] = 1.0;
        for j in 1..q.min(n_states - 1) {
            selection[(j, 0)] = ma_coef[j - 1];
        }
        
        // Covariance matrices
        let observation_cov = ndarray::array![[0.0]]; // No measurement error in standard ARMA
        let state_cov = ndarray::array![[sigma2]];
        
        // Initial state (diffuse)
        let initial_state_mean = Array1::zeros(n_states);
        let mut initial_state_cov = Array2::zeros((n_states, n_states));
        for i in 0..n_states {
            initial_state_cov[(i, i)] = 1e6;
        }
        
        Self {
            observation_matrix,
            transition_matrix: transition,
            selection_matrix: selection,
            observation_cov,
            state_cov,
            initial_state_mean,
            initial_state_cov,
        }
    }
    
    /// Apply Kalman filter to time series
    pub fn filter(&self, y: &Array1<f64>) -> Result<KalmanFilterResults> {
        let n = y.len();
        let n_states = self.observation_matrix.ncols();
        let n_obs = self.observation_matrix.nrows();
        
        // Initialize arrays
        let mut filtered_means = Array2::zeros((n, n_states));
        let mut filtered_covs = Array3::zeros((n, n_states, n_states));
        let mut predicted_means = Array2::zeros((n, n_states));
        let mut predicted_covs = Array3::zeros((n, n_states, n_states));
        let mut innovations = Array1::zeros(n);
        let mut innovation_variances = Array1::zeros(n);
        let mut kalman_gains = Array3::zeros((n, n_states, n_obs));
        
        let mut log_likelihood = 0.0;
        
        // Initial prediction (t = 0)
        let mut pred_mean = self.initial_state_mean.clone();
        let mut pred_cov = self.initial_state_cov.clone();
        
        for t in 0..n {
            // Store prediction
            predicted_means.row_mut(t).assign(&pred_mean);
            predicted_covs.slice_mut(ndarray::s![t, .., ..]).assign(&pred_cov);
            
            // Innovation (prediction error)
            let obs_pred = self.observation_matrix.dot(&pred_mean);
            let innovation = y[t] - obs_pred[0];
            innovations[t] = innovation;
            
            // Innovation variance
            let innovation_var = self.observation_matrix.dot(&pred_cov.dot(&self.observation_matrix.t())) 
                + &self.observation_cov;
            let innovation_var_scalar = innovation_var[(0, 0)];
            innovation_variances[t] = innovation_var_scalar;
            
            // Log-likelihood contribution (ignoring constant)
            if innovation_var_scalar > 0.0 {
                log_likelihood += -0.5 * innovation_var_scalar.ln() 
                    - 0.5 * innovation.powi(2) / innovation_var_scalar;
            }
            
            // Kalman gain
            let kalman_gain = if innovation_var_scalar > 0.0 {
                pred_cov.dot(&self.observation_matrix.t()) / innovation_var_scalar
            } else {
                Array2::zeros((n_states, n_obs))
            };
            
            kalman_gains.slice_mut(ndarray::s![t, .., ..]).assign(&kalman_gain);
            
            // Filtered state estimate
            let filtered_mean = &pred_mean + kalman_gain.dot(&ndarray::array![innovation]);
            let filtered_cov = &pred_cov - kalman_gain.dot(&self.observation_matrix.dot(&pred_cov));
            
            // Store filtered estimates
            filtered_means.row_mut(t).assign(&filtered_mean);
            filtered_covs.slice_mut(ndarray::s![t, .., ..]).assign(&filtered_cov);
            
            // Predict next state
            if t < n - 1 {
                pred_mean = self.transition_matrix.dot(&filtered_mean);
                pred_cov = self.transition_matrix.dot(&filtered_cov.dot(&self.transition_matrix.t()))
                    + self.selection_matrix.dot(&self.state_cov.dot(&self.selection_matrix.t()));
            }
        }
        
        Ok(KalmanFilterResults {
            filtered_state_means: filtered_means,
            filtered_state_covs: filtered_covs,
            predicted_state_means: predicted_means,
            predicted_state_covs: predicted_covs,
            innovations,
            innovation_variances,
            kalman_gains,
            log_likelihood,
        })
    }
    
    /// Apply Kalman smoother (Rauch-Tung-Striebel smoother)
    pub fn smooth(&self, filter_results: &KalmanFilterResults) -> KalmanFilterResults {
        let n = filter_results.filtered_state_means.nrows();
        let n_states = self.observation_matrix.ncols();
        
        // Initialize smoothed arrays
        let mut smoothed_means = filter_results.filtered_state_means.clone();
        let mut smoothed_covs = filter_results.filtered_state_covs.clone();
        
        // Start from last time point
        let mut smoother_gain = Array2::zeros((n_states, n_states));
        
        for t in (0..n-1).rev() {
            // Smoother gain
            let pred_cov = filter_results.predicted_state_covs.slice(ndarray::s![t+1, .., ..]);
            let filtered_cov = filter_results.filtered_state_covs.slice(ndarray::s![t, .., ..]);
            
            let pred_cov_inv = linalg::inv(&pred_cov.to_owned()).unwrap_or_else(|_| pred_cov.to_owned());
            smoother_gain.assign(&filtered_cov.dot(&self.transition_matrix.t()).dot(&pred_cov_inv));
            
            // Smoothed state
            let filtered_mean = filter_results.filtered_state_means.row(t);
            let next_smoothed_mean = smoothed_means.row(t + 1);
            let next_pred_mean = filter_results.predicted_state_means.row(t + 1);
            
            let mut smoothed_mean = filtered_mean.to_owned();
            smoothed_mean += &smoother_gain.dot(&(&next_smoothed_mean - &next_pred_mean));
            
            // Smoothed covariance
            let next_smoothed_cov = smoothed_covs.slice(ndarray::s![t+1, .., ..]);
            let next_pred_cov = filter_results.predicted_state_covs.slice(ndarray::s![t+1, .., ..]);
            
            let mut smoothed_cov = filtered_cov.to_owned();
            let diff_cov = &next_smoothed_cov - &next_pred_cov;
            smoothed_cov += &smoother_gain.dot(&diff_cov.dot(&smoother_gain.t()));
            
            // Store results
            smoothed_means.row_mut(t).assign(&smoothed_mean);
            smoothed_covs.slice_mut(ndarray::s![t, .., ..]).assign(&smoothed_cov);
        }
        
        KalmanFilterResults {
            filtered_state_means: smoothed_means,
            filtered_state_covs: smoothed_covs,
            ..filter_results.clone()
        }
    }
    
    /// Forecast future states
    pub fn forecast(
        &self,
        filter_results: &KalmanFilterResults,
        steps: usize,
    ) -> (Array2<f64>, Array3<f64>) {
        let n = filter_results.filtered_state_means.nrows();
        let n_states = self.observation_matrix.ncols();
        
        let mut forecast_means = Array2::zeros((steps, n_states));
        let mut forecast_covs = Array3::zeros((steps, n_states, n_states));
        
        // Start from last filtered state
        let mut current_mean = filter_results.filtered_state_means.row(n - 1).to_owned();
        let mut current_cov = filter_results.filtered_state_covs.slice(ndarray::s![n-1, .., ..]).to_owned();
        
        for h in 0..steps {
            // Predict state
            current_mean = self.transition_matrix.dot(&current_mean);
            current_cov = self.transition_matrix.dot(&current_cov.dot(&self.transition_matrix.t()))
                + self.selection_matrix.dot(&self.state_cov.dot(&self.selection_matrix.t()));
            
            // Store forecast
            forecast_means.row_mut(h).assign(&current_mean);
            forecast_covs.slice_mut(ndarray::s![h, .., ..]).assign(&current_cov);
        }
        
        (forecast_means, forecast_covs)
    }
    
    /// Calculate marginal log-likelihood
    pub fn log_likelihood(&self, y: &Array1<f64>) -> Result<f64> {
        let results = self.filter(y)?;
        Ok(results.log_likelihood)
    }
    
    /// Estimate parameters via maximum likelihood
    pub fn estimate(&mut self, _y: &Array1<f64>) -> Result<()> {
        // This would implement MLE using EM algorithm or numerical optimization
        // For now, just return the model as-is
        Ok(())
    }
}

/// Kalman filter implementation
pub struct KalmanFilter;

impl KalmanFilter {
    /// Create new Kalman filter
    pub fn new() -> Self {
        Self
    }
    
    /// Run filter on state space model
    pub fn filter(&self, model: &StateSpaceModel, y: &Array1<f64>) -> Result<KalmanFilterResults> {
        model.filter(y)
    }
    
    /// Run smoother on filtered results
    pub fn smooth(&self, model: &StateSpaceModel, results: &KalmanFilterResults) -> KalmanFilterResults {
        model.smooth(results)
    }
    
    /// Run filter and smoother
    pub fn filter_smooth(&self, model: &StateSpaceModel, y: &Array1<f64>) -> Result<KalmanFilterResults> {
        let filtered = model.filter(y)?;
        Ok(model.smooth(&filtered))
    }
    
    /// Forecast future observations
    pub fn forecast(
        &self,
        model: &StateSpaceModel,
        results: &KalmanFilterResults,
        steps: usize,
    ) -> (Array1<f64>, Array1<f64>) {
        let (state_means, state_covs) = model.forecast(results, steps);
        
        let mut forecast_means = Array1::zeros(steps);
        let mut forecast_variances = Array1::zeros(steps);
        
        for h in 0..steps {
            let state_mean = state_means.row(h);
            let state_cov = state_covs.slice(ndarray::s![h, .., ..]);
            
            // Forecast observation
            let obs_mean = model.observation_matrix.dot(&state_mean);
            forecast_means[h] = obs_mean[0];
            
            // Forecast variance
            let obs_var = model.observation_matrix.dot(&state_cov.dot(&model.observation_matrix.t()))
                + &model.observation_cov;
            forecast_variances[h] = obs_var[(0, 0)];
        }
        
        (forecast_means, forecast_variances)
    }
}