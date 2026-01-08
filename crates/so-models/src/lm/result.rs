//! Linear regression result structure
//!
//! This module defines the result structure for linear regression models,
//! containing all the information from fitting a linear model.

use ndarray_linalg::Inverse;
use serde::{Deserialize, Serialize};
use statrs::distribution::ContinuousCDF;
use std::fmt;

use crate::{
    base::{Coefficient, ModelResultTrait, ModelStatistics},
    lm::ols::{Matrix, Vector},
};

/// OLS linear regression result
#[derive(Debug, Clone)]
pub struct LinearRegressionResult {
    /// Coefficients (β)
    pub coefficients: Vector,
    /// Standard errors of coefficients
    pub standard_errors: Vector,
    /// t-statistics for coefficients
    pub t_statistics: Vector,
    /// p-values for coefficients
    pub p_values: Vector,
    /// Lower bounds of confidence intervals
    pub ci_lower: Vector,
    /// Upper bounds of confidence intervals
    pub ci_upper: Vector,
    /// Fitted values (ŷ)
    pub fitted_values: Vector,
    /// Residuals (y - ŷ)
    pub residuals: Vector,
    /// Hat matrix diagonal (leverage)
    pub hat_diagonal: Vector,
    /// Cook's distances
    pub cooks_distance: Vector,
    /// Design matrix (X)
    pub x: Matrix,
    /// Response vector (y)
    pub y: Vector,
    /// Variable names
    pub variable_names: Vec<String>,
    /// Model statistics
    pub model_statistics: ModelStatistics,
    /// Has intercept
    pub has_intercept: bool,
    /// Scale factors if predictors were scaled
    pub scaling_factors: Option<ScalingFactors>,
}

/// Scaling factors for centering and scaling predictors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingFactors {
    /// Means for each predictor (if centered)
    pub means: Vec<f64>,
    /// Standard deviations for each predictor (if scaled)
    pub stds: Vec<f64>,
}

impl LinearRegressionResult {
    /// Create a new LinearRegressionResult
    pub fn new(
        coefficients: Vector,
        standard_errors: Vector,
        t_statistics: Vector,
        p_values: Vector,
        ci_lower: Vector,
        ci_upper: Vector,
        fitted_values: Vector,
        residuals: Vector,
        hat_diagonal: Vector,
        cooks_distance: Vector,
        x: Matrix,
        y: Vector,
        variable_names: Vec<String>,
        model_statistics: ModelStatistics,
        has_intercept: bool,
        scaling_factors: Option<ScalingFactors>,
    ) -> Self {
        Self {
            coefficients,
            standard_errors,
            t_statistics,
            p_values,
            ci_lower,
            ci_upper,
            fitted_values,
            residuals,
            hat_diagonal,
            cooks_distance,
            x,
            y,
            variable_names,
            model_statistics,
            has_intercept,
            scaling_factors,
        }
    }

    /// Convert coefficients to Coefficient structs
    pub fn to_coefficients(&self) -> Vec<Coefficient> {
        self.coefficients
            .iter()
            .zip(self.standard_errors.iter())
            .zip(self.t_statistics.iter())
            .zip(self.p_values.iter())
            .zip(self.ci_lower.iter())
            .zip(self.ci_upper.iter())
            .enumerate()
            .map(|(i, (((((&coef, &se), &t), &p), &ci_low), &ci_up))| {
                let name = if i < self.variable_names.len() {
                    self.variable_names[i].clone()
                } else {
                    format!("x{}", i)
                };

                let is_intercept = if self.has_intercept { i == 0 } else { false };

                Coefficient::new(name, coef)
                    .with_std_error(se)
                    .with_t_stat(t)
                    .with_p_value(p)
                    .with_ci(ci_low, ci_up)
            })
            .collect()
    }

    /// Get the number of observations
    pub fn n_obs(&self) -> usize {
        self.y.len()
    }

    /// Get the number of predictors (including intercept)
    pub fn n_predictors(&self) -> usize {
        self.coefficients.len()
    }

    /// Get residual sum of squares
    pub fn rss(&self) -> f64 {
        self.residuals.mapv(|r| r * r).sum()
    }

    /// Get total sum of squares
    pub fn tss(&self) -> f64 {
        let y_mean = self.y.mean().unwrap_or(0.0);
        self.y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>()
    }

    /// Get explained sum of squares
    pub fn ess(&self) -> f64 {
        self.tss() - self.rss()
    }

    /// Get the covariance matrix of coefficients
    pub fn cov_matrix(&self) -> Matrix {
        let p = self.n_predictors();
        let mut cov = Matrix::zeros((p, p));

        for i in 0..p {
            for j in 0..p {
                // Cov(β_i, β_j) = se_i * se_j * correlation
                // For now, we'll just compute the diagonal
                if i == j {
                    cov[(i, j)] = self.standard_errors[i].powi(2);
                }
            }
        }

        cov
    }

    /// Get studentized residuals
    pub fn studentized_residuals(&self) -> Vector {
        let n = self.n_obs();
        let p = self.n_predictors();
        let rse = self.model_statistics.residual_std_error.unwrap_or(0.0);

        self.residuals
            .iter()
            .zip(self.hat_diagonal.iter())
            .map(|(&r, &h)| r / (rse * (1.0 - h).sqrt()))
            .collect()
    }

    /// Get standardized residuals
    pub fn standardized_residuals(&self) -> Vector {
        let rse = self.model_statistics.residual_std_error.unwrap_or(0.0);
        self.residuals.mapv(|r| r / rse)
    }

    /// Get DFFITS values
    pub fn dffits(&self) -> Vector {
        let n = self.n_obs();
        let p = self.n_predictors();

        self.studentized_residuals()
            .iter()
            .zip(self.hat_diagonal.iter())
            .map(|(&r_star, &h)| r_star * (h / (1.0 - h)).sqrt())
            .collect()
    }

    /// Get DFBETAS for all coefficients
    pub fn dfbetas(&self) -> Matrix {
        let n = self.n_obs();
        let p = self.n_predictors();

        // Initialize DFBETAS matrix
        let dfbetas = Matrix::zeros((n, p));

        // This is a simplified version
        // In practice, we would need to refit the model n times
        // or use the one-step approximation
        unimplemented!("dfbetas");

        dfbetas
    }

    /// Check for influential points
    pub fn influential_points(&self, threshold: Option<f64>) -> Vec<usize> {
        let threshold = threshold.unwrap_or(4.0 / self.n_obs() as f64);
        self.cooks_distance
            .iter()
            .enumerate()
            .filter(|(_, d)| **d > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Check for high leverage points
    pub fn high_leverage_points(&self) -> Vec<usize> {
        let p = self.n_predictors();
        let threshold = 2.0 * p as f64 / self.n_obs() as f64;

        self.hat_diagonal
            .iter()
            .enumerate()
            .filter(|(_, h)| **h > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get predictions for new data
    pub fn predict(&self, x_new: &Matrix) -> Vector {
        x_new.dot(&self.coefficients)
    }

    /// Get confidence intervals for predictions
    pub fn predict_ci(&self, x_new: &Matrix, alpha: f64) -> (Vector, Vector) {
        let predictions = self.predict(x_new);
        let n = self.n_obs();
        let p = self.n_predictors();

        // Calculate standard errors of predictions
        let se_fit: Vector = x_new
            .rows()
            .into_iter()
            .map(|row| {
                let xtx_inv = self
                    .x
                    .t()
                    .dot(&self.x)
                    .inv()
                    .unwrap_or_else(|_| Matrix::eye(p));
                let h_new = row.dot(&xtx_inv).dot(&row.t());
                self.model_statistics.residual_std_error.unwrap_or(0.0) * (1.0 + h_new).sqrt()
            })
            .collect();

        // t-critical value
        let df = n - p;
        let t_dist = statrs::distribution::StudentsT::new(0.0, 1.0, df as f64)
            .unwrap_or_else(|_| statrs::distribution::StudentsT::new(0.0, 1.0, 1.0).unwrap());
        let t_critical = t_dist.inverse_cdf(1.0 - alpha / 2.0);

        // Calculate confidence intervals
        let ci_lower: Vector = predictions
            .iter()
            .zip(se_fit.iter())
            .map(|(&pred, &se)| pred - t_critical * se)
            .collect();

        let ci_upper: Vector = predictions
            .iter()
            .zip(se_fit.iter())
            .map(|(&pred, &se)| pred + t_critical * se)
            .collect();

        (ci_lower, ci_upper)
    }

    /// Get prediction intervals
    pub fn predict_pi(&self, x_new: &Matrix, alpha: f64) -> (Vector, Vector) {
        let predictions = self.predict(x_new);
        let n = self.n_obs();
        let p = self.n_predictors();

        // Standard error for prediction interval
        let se_pred: Vector = x_new
            .rows()
            .into_iter()
            .map(|row| {
                let xtx_inv = self
                    .x
                    .t()
                    .dot(&self.x)
                    .inv()
                    .unwrap_or_else(|_| Matrix::eye(p));
                let h_new = row.dot(&xtx_inv).dot(&row.t());
                self.model_statistics.residual_std_error.unwrap_or(0.0) * (1.0 + h_new).sqrt()
            })
            .collect();

        // t-critical value
        let df = n - p;
        let t_dist = statrs::distribution::StudentsT::new(0.0, 1.0, df as f64)
            .unwrap_or_else(|_| statrs::distribution::StudentsT::new(0.0, 1.0, 1.0).unwrap());
        let t_critical = t_dist.inverse_cdf(1.0 - alpha / 2.0);

        // Calculate prediction intervals
        let pi_lower: Vector = predictions
            .iter()
            .zip(se_pred.iter())
            .map(|(&pred, &se)| pred - t_critical * se)
            .collect();

        let pi_upper: Vector = predictions
            .iter()
            .zip(se_pred.iter())
            .map(|(&pred, &se)| pred + t_critical * se)
            .collect();

        (pi_lower, pi_upper)
    }
}

impl ModelResultTrait for LinearRegressionResult {
    fn coefficients(&self) -> &[Coefficient] {
        // Note: This requires converting to Coefficient structs
        // We'll handle this differently in the LinearRegression struct
        &[]
    }

    fn fitted_values(&self) -> &Vector {
        &self.fitted_values
    }

    fn residuals(&self) -> &Vector {
        &self.residuals
    }

    fn r_squared(&self) -> Option<f64> {
        self.model_statistics.r_squared
    }

    fn adj_r_squared(&self) -> Option<f64> {
        self.model_statistics.adj_r_squared
    }

    fn aic(&self) -> Option<f64> {
        self.model_statistics.aic
    }

    fn bic(&self) -> Option<f64> {
        self.model_statistics.bic
    }

    fn log_likelihood(&self) -> Option<f64> {
        self.model_statistics.log_likelihood
    }

    fn residual_std_error(&self) -> Option<f64> {
        self.model_statistics.residual_std_error
    }

    fn df_residual(&self) -> Option<usize> {
        self.model_statistics.df_residual
    }

    fn df_model(&self) -> Option<usize> {
        self.model_statistics.df_model
    }
}

impl fmt::Display for LinearRegressionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Linear Regression Results")?;
        writeln!(f, "========================")?;
        writeln!(f, "Observations: {}", self.n_obs())?;
        writeln!(f, "Predictors:   {}", self.n_predictors())?;
        writeln!(f)?;

        if let Some(r2) = self.model_statistics.r_squared {
            writeln!(f, "R-squared:           {:.4}", r2)?;
        }

        if let Some(adj_r2) = self.model_statistics.adj_r_squared {
            writeln!(f, "Adjusted R-squared:  {:.4}", adj_r2)?;
        }

        if let Some(f_stat) = self.model_statistics.f_statistic {
            writeln!(f, "F-statistic:        {:.4}", f_stat)?;
        }

        if let Some(p_val) = self.model_statistics.f_p_value {
            writeln!(f, "F p-value:          {:.4}", p_val)?;
        }

        if let Some(rse) = self.model_statistics.residual_std_error {
            writeln!(f, "Residual Std Error: {:.4}", rse)?;
        }

        if let Some(aic) = self.model_statistics.aic {
            writeln!(f, "AIC:                {:.4}", aic)?;
        }

        if let Some(bic) = self.model_statistics.bic {
            writeln!(f, "BIC:                {:.4}", bic)?;
        }

        Ok(())
    }
}
