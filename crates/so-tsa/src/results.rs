//! Unified result structures for time series analysis
//!
//! This module provides common result structures and diagnostics
//! for time series models, including statistical summaries,
//! residual diagnostics, and model comparison.

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use statrs::function::gamma;
use std::collections::HashMap;

/// Unified time series analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSAResults {
    /// Model name/type
    pub model_type: String,
    /// Model parameters
    pub params: HashMap<String, f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Residuals
    pub residuals: Array1<f64>,
    /// Fitted values
    pub fitted: Array1<f64>,
    /// Model-specific diagnostics
    pub diagnostics: HashMap<String, f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TSAResults {
    /// Create new TSA results
    pub fn new(
        model_type: &str,
        params: HashMap<String, f64>,
        n_obs: usize,
        n_params: usize,
        log_likelihood: f64,
        residuals: Array1<f64>,
        fitted: Array1<f64>,
    ) -> Self {
        // Calculate information criteria
        let aic = 2.0 * n_params as f64 - 2.0 * log_likelihood;
        let bic = (n_obs as f64).ln() * n_params as f64 - 2.0 * log_likelihood;

        Self {
            model_type: model_type.to_string(),
            params,
            n_obs,
            n_params,
            log_likelihood,
            aic,
            bic,
            residuals,
            fitted,
            diagnostics: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add diagnostic statistic
    pub fn add_diagnostic(&mut self, name: &str, value: f64) {
        self.diagnostics.insert(name.to_string(), value);
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Calculate residual statistics
    pub fn residual_stats(&self) -> ResidualDiagnostics {
        let residuals = &self.residuals;
        let n = residuals.len();

        // Basic statistics
        let mean = residuals.mean().unwrap_or(0.0);
        let variance = residuals.var(1.0);
        let std_dev = variance.sqrt();
        let skewness = if variance > 0.0 {
            let mean_dev = residuals - mean;
            let m3 = mean_dev.mapv(|x| x.powi(3)).mean().unwrap_or(0.0);
            m3 / variance.powi(3).sqrt()
        } else {
            0.0
        };

        let kurtosis = if variance > 0.0 {
            let mean_dev = residuals - mean;
            let m4 = mean_dev.mapv(|x| x.powi(4)).mean().unwrap_or(0.0);
            m4 / variance.powi(2)
        } else {
            0.0
        };

        // Jarque-Bera test for normality
        let jb_stat = n as f64 / 6.0 * (skewness.powi(2) + 0.25 * (kurtosis - 3.0).powi(2));
        let jb_p_value = 1.0 - chi2_cdf(2, jb_stat);

        // Ljung-Box test for autocorrelation
        let lb_stat = ljung_box(&residuals, 10);
        let lb_p_value = 1.0 - chi2_cdf(10, lb_stat);

        // Durbin-Watson test for autocorrelation
        let dw_stat = durbin_watson(&residuals);

        ResidualDiagnostics {
            mean,
            variance,
            std_dev,
            skewness,
            kurtosis,
            jarque_bera: (jb_stat, jb_p_value),
            ljung_box: (lb_stat, lb_p_value),
            durbin_watson: dw_stat,
            min: residuals.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max: residuals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        }
    }

    /// Create summary string
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("{} Model Results\n", self.model_type));
        summary.push_str(&format!("{}\n", "=".repeat(self.model_type.len() + 13)));

        summary.push_str(&format!("Observations: {}\n", self.n_obs));
        summary.push_str(&format!("Parameters: {}\n", self.n_params));
        summary.push_str(&format!("Log-Likelihood: {:.4}\n", self.log_likelihood));
        summary.push_str(&format!("AIC: {:.4}\n", self.aic));
        summary.push_str(&format!("BIC: {:.4}\n", self.bic));

        // Parameter estimates
        if !self.params.is_empty() {
            summary.push_str("\nParameter Estimates:\n");
            for (name, value) in &self.params {
                summary.push_str(&format!("  {}: {:.6}\n", name, value));
            }
        }

        // Residual diagnostics
        let diag = self.residual_stats();
        summary.push_str(&format!("\nResidual Diagnostics:\n"));
        summary.push_str(&format!("  Mean: {:.6}\n", diag.mean));
        summary.push_str(&format!("  Std Dev: {:.6}\n", diag.std_dev));
        summary.push_str(&format!("  Skewness: {:.4}\n", diag.skewness));
        summary.push_str(&format!("  Kurtosis: {:.4}\n", diag.kurtosis));
        summary.push_str(&format!(
            "  Jarque-Bera: {:.4} (p={:.4})\n",
            diag.jarque_bera.0, diag.jarque_bera.1
        ));
        summary.push_str(&format!(
            "  Ljung-Box(10): {:.4} (p={:.4})\n",
            diag.ljung_box.0, diag.ljung_box.1
        ));
        summary.push_str(&format!("  Durbin-Watson: {:.4}\n", diag.durbin_watson));

        // Additional diagnostics
        if !self.diagnostics.is_empty() {
            summary.push_str("\nModel Diagnostics:\n");
            for (name, value) in &self.diagnostics {
                summary.push_str(&format!("  {}: {:.6}\n", name, value));
            }
        }

        summary
    }

    /// Compare with another model
    pub fn compare(&self, other: &Self) -> ModelComparison {
        ModelComparison {
            model_a: self.model_type.clone(),
            model_b: other.model_type.clone(),
            aic_diff: other.aic - self.aic,
            bic_diff: other.bic - self.bic,
            ll_diff: other.log_likelihood - self.log_likelihood,
        }
    }
}

/// Residual diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualDiagnostics {
    /// Mean of residuals
    pub mean: f64,
    /// Variance of residuals
    pub variance: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Jarque-Bera test (statistic, p-value)
    pub jarque_bera: (f64, f64),
    /// Ljung-Box test (statistic, p-value)
    pub ljung_box: (f64, f64),
    /// Durbin-Watson statistic
    pub durbin_watson: f64,
    /// Minimum residual
    pub min: f64,
    /// Maximum residual
    pub max: f64,
}

impl ResidualDiagnostics {
    /// Check if residuals are normally distributed (5% level)
    pub fn is_normal(&self) -> bool {
        self.jarque_bera.1 > 0.05
    }

    /// Check if residuals show autocorrelation (5% level)
    pub fn has_autocorrelation(&self) -> bool {
        self.ljung_box.1 < 0.05
    }

    /// Check Durbin-Watson for autocorrelation
    /// DW ≈ 2: no autocorrelation
    /// DW < 1.5: positive autocorrelation  
    /// DW > 2.5: negative autocorrelation
    pub fn durbin_watson_interpretation(&self) -> &'static str {
        if self.durbin_watson < 1.5 {
            "Positive autocorrelation"
        } else if self.durbin_watson > 2.5 {
            "Negative autocorrelation"
        } else {
            "No significant autocorrelation"
        }
    }

    /// Create summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Residual Diagnostics\n");
        summary.push_str("====================\n");

        summary.push_str(&format!(
            "Normality (Jarque-Bera): {:.4} (p={:.4}) {}\n",
            self.jarque_bera.0,
            self.jarque_bera.1,
            if self.is_normal() { "✓" } else { "✗" }
        ));
        summary.push_str(&format!(
            "Autocorrelation (Ljung-Box): {:.4} (p={:.4}) {}\n",
            self.ljung_box.0,
            self.ljung_box.1,
            if !self.has_autocorrelation() {
                "✓"
            } else {
                "✗"
            }
        ));
        summary.push_str(&format!(
            "Durbin-Watson: {:.4} - {}\n",
            self.durbin_watson,
            self.durbin_watson_interpretation()
        ));

        summary.push_str(&format!("\nDistribution:\n"));
        summary.push_str(&format!("  Mean: {:.6}\n", self.mean));
        summary.push_str(&format!("  Std Dev: {:.6}\n", self.std_dev));
        summary.push_str(&format!(
            "  Skewness: {:.4} ({})\n",
            self.skewness,
            if self.skewness.abs() > 0.5 {
                "skewed"
            } else {
                "symmetric"
            }
        ));
        summary.push_str(&format!(
            "  Kurtosis: {:.4} ({})\n",
            self.kurtosis,
            if self.kurtosis > 3.5 {
                "leptokurtic"
            } else if self.kurtosis < 2.5 {
                "platykurtic"
            } else {
                "mesokurtic"
            }
        ));
        summary.push_str(&format!("  Range: [{:.4}, {:.4}]\n", self.min, self.max));

        summary
    }
}

/// Model comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// First model name
    pub model_a: String,
    /// Second model name
    pub model_b: String,
    /// AIC difference (B - A)
    pub aic_diff: f64,
    /// BIC difference (B - A)
    pub bic_diff: f64,
    /// Log-likelihood difference (B - A)
    pub ll_diff: f64,
}

impl ModelComparison {
    /// Determine which model is better based on AIC
    pub fn aic_preferred(&self) -> String {
        if self.aic_diff < 0.0 {
            self.model_b.clone()
        } else {
            self.model_a.clone()
        }
    }

    /// Determine which model is better based on BIC
    pub fn bic_preferred(&self) -> String {
        if self.bic_diff < 0.0 {
            self.model_b.clone()
        } else {
            self.model_a.clone()
        }
    }

    /// Strength of evidence based on AIC difference
    pub fn aic_evidence(&self) -> &'static str {
        let diff = self.aic_diff.abs();
        if diff < 2.0 {
            "No meaningful difference"
        } else if diff < 4.0 {
            "Weak evidence"
        } else if diff < 7.0 {
            "Positive evidence"
        } else if diff < 10.0 {
            "Strong evidence"
        } else {
            "Very strong evidence"
        }
    }

    /// Create comparison summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!(
            "Model Comparison: {} vs {}\n",
            self.model_a, self.model_b
        ));
        summary.push_str(&format!(
            "{}\n",
            "=".repeat(self.model_a.len() + self.model_b.len() + 17)
        ));

        summary.push_str(&format!(
            "AIC difference: {:.4} ({} preferred)\n",
            self.aic_diff,
            self.aic_preferred()
        ));
        summary.push_str(&format!("  Evidence: {}\n", self.aic_evidence()));

        summary.push_str(&format!(
            "BIC difference: {:.4} ({} preferred)\n",
            self.bic_diff,
            self.bic_preferred()
        ));

        summary.push_str(&format!("Log-likelihood difference: {:.4}\n", self.ll_diff));

        if self.ll_diff > 0.0 {
            summary.push_str(&format!("  {} has higher likelihood\n", self.model_b));
        } else {
            summary.push_str(&format!("  {} has higher likelihood\n", self.model_a));
        }

        summary
    }
}

/// Chi-squared cumulative distribution function (simplified)
pub fn chi2_cdf(k: usize, x: f64) -> f64 {
    // Incomplete gamma function approximation
    if x <= 0.0 {
        return 0.0;
    }

    let k_half = k as f64 / 2.0;
    let x_half = x / 2.0;

    // Series expansion for incomplete gamma
    let mut sum = 0.0;
    let mut term = 1.0;
    let mut n = 0;

    while term > 1e-10 && n < 100 {
        term = x_half.powi(n as i32) / (k_half + n as f64);
        sum += term;
        n += 1;
    }

    sum * (-x_half).exp() * x_half.powf(k_half) / gamma::gamma(k_half)
}

/// Ljung-Box test for autocorrelation
fn ljung_box(residuals: &Array1<f64>, lag: usize) -> f64 {
    let n = residuals.len();
    let mean = residuals.mean().unwrap_or(0.0);
    let variance = residuals.var(1.0);

    if variance <= 0.0 {
        return 0.0;
    }

    let mut q = 0.0;

    for k in 1..=lag {
        let mut autocov = 0.0;
        for t in k..n {
            autocov += (residuals[t] - mean) * (residuals[t - k] - mean);
        }
        let rk = autocov / (variance * n as f64);
        q += rk.powi(2) / (n - k) as f64;
    }

    q * n as f64 * (n as f64 + 2.0)
}

/// Durbin-Watson test for autocorrelation
fn durbin_watson(residuals: &Array1<f64>) -> f64 {
    let n = residuals.len();
    if n < 2 {
        return 2.0; // No autocorrelation by definition
    }

    let mut numerator = 0.0;
    for i in 1..n {
        numerator += (residuals[i] - residuals[i - 1]).powi(2);
    }

    let denominator: f64 = residuals.iter().map(|&x| x.powi(2)).sum();

    if denominator > 0.0 {
        numerator / denominator
    } else {
        2.0
    }
}
