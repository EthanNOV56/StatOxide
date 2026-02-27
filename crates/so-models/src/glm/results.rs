//! GLM results structure and diagnostics

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::glm::family::{Family, Link};

/// Results from fitting a Generalized Linear Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLMResults {
    /// Estimated coefficients
    pub coefficients: Array1<f64>,
    /// Standard errors of coefficients
    pub std_errors: Array1<f64>,
    /// z-statistics for coefficients (Wald test)
    pub z_values: Array1<f64>,
    /// p-values for coefficients
    pub p_values: Array1<f64>,
    /// Fitted values (on response scale)
    pub fitted_values: Array1<f64>,
    /// Residuals (response scale)
    pub residuals: Array1<f64>,
    /// Pearson residuals
    pub pearson_residuals: Array1<f64>,
    /// Diagonal of hat matrix (leverage)
    pub hat_matrix_diag: Array1<f64>,
    /// Scale/dispersion parameter
    pub scale: f64,
    /// Residual deviance
    pub deviance: f64,
    /// Null deviance
    pub null_deviance: f64,
    /// Residual degrees of freedom
    pub df_residual: usize,
    /// Null degrees of freedom
    pub df_null: usize,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Whether IRLS converged
    pub converged: bool,
    /// Number of IRLS iterations
    pub iterations: usize,
    /// Distribution family
    pub family: Family,
    /// Link function
    pub link: Link,
    /// Whether intercept was included
    pub intercept: bool,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
}

impl GLMResults {
    /// Create a summary string similar to R's summary.glm()
    pub fn summary(&self, feature_names: &[String]) -> String {
        let n_coef = self.coefficients.len();
        let intercept_included = self.intercept;

        let mut summary = String::new();

        // Header
        summary.push_str("Generalized Linear Model Results\n");
        summary.push_str("================================\n");
        summary.push_str(&format!(
            "Family: {} ({})\n",
            self.family.name(),
            self.link.name()
        ));
        summary.push_str(&format!("Link: {}\n", self.link.name()));
        summary.push_str(&format!("Number of observations: {}\n", self.n_obs));
        summary.push_str(&format!(
            "Degrees of Freedom: {} total, {} residual\n",
            self.n_params, self.df_residual
        ));
        summary.push_str(&format!("Scale (dispersion): {:.4}\n", self.scale));

        // Deviance table
        summary.push_str("\nDeviance Residuals:\n");
        summary.push_str(&format!(
            "    Null deviance: {:.4} on {} df\n",
            self.null_deviance, self.df_null
        ));
        summary.push_str(&format!(
            "Residual deviance: {:.4} on {} df\n",
            self.deviance, self.df_residual
        ));

        if self.null_deviance > 0.0 {
            let pseudo_r2 = 1.0 - self.deviance / self.null_deviance;
            summary.push_str(&format!("Pseudo R-squared: {:.4}\n", pseudo_r2));
        }

        summary.push_str(&format!("AIC: {:.2}, BIC: {:.2}\n", self.aic, self.bic));

        // Convergence info
        summary.push_str(&format!(
            "\nIRLS converged: {} ({} iterations)\n",
            self.converged, self.iterations
        ));

        // Coefficients table
        summary.push_str("\nCoefficients:\n");
        summary.push_str(&format!(
            "{:>20} {:>10} {:>10} {:>10} {:>10}\n",
            " ", "Estimate", "Std. Error", "z value", "Pr(>|z|)"
        ));
        summary.push_str(&format!("{}\n", "-".repeat(60)));

        for i in 0..n_coef {
            let name = if i == 0 && intercept_included {
                "(Intercept)".to_string()
            } else if intercept_included {
                feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("x{}", i))
            } else {
                feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("x{}", i))
            };

            let coef = self.coefficients[i];
            let se = self.std_errors[i];
            let z = self.z_values[i];
            let p = self.p_values[i];

            let significance = if p < 0.001 {
                "***"
            } else if p < 0.01 {
                "**"
            } else if p < 0.05 {
                "*"
            } else if p < 0.1 {
                "."
            } else {
                ""
            };

            summary.push_str(&format!(
                "{:>20} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {}\n",
                name, coef, se, z, p, significance
            ));
        }

        summary.push_str(&format!("{}\n", "-".repeat(60)));
        summary.push_str("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n");

        // Dispersion information
        if (self.family == Family::Poisson || self.family == Family::Binomial)
            && (self.scale - 1.0).abs() > 0.1
        {
            summary.push_str(&format!(
                "\nWarning: Dispersion parameter is {:.3}, not 1.\n",
                self.scale
            ));
            summary.push_str(&format!(
                "Consider using quasi-{} family.\n",
                self.family.name().to_lowercase()
            ));
        }

        summary
    }

    /// Get the coefficients as a vector with names
    pub fn coefficients_with_names(&self, feature_names: &[String]) -> Vec<(String, f64)> {
        let n_coef = self.coefficients.len();
        let mut result = Vec::with_capacity(n_coef);

        for i in 0..n_coef {
            let name = if i == 0 && self.intercept {
                "(Intercept)".to_string()
            } else if self.intercept {
                feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("x{}", i))
            } else {
                feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("x{}", i))
            };

            result.push((name, self.coefficients[i]));
        }

        result
    }

    /// Get confidence intervals for coefficients
    pub fn confidence_intervals(
        &self,
        alpha: f64,
        feature_names: &[String],
    ) -> Vec<(String, f64, f64)> {
        let n_coef = self.coefficients.len();
        let z_critical = 1.0 - alpha / 2.0;
        let z_value = Normal::new(0.0, 1.0).unwrap().inverse_cdf(z_critical);

        let mut intervals = Vec::with_capacity(n_coef);

        for i in 0..n_coef {
            let name = if i == 0 && self.intercept {
                "(Intercept)".to_string()
            } else if self.intercept {
                feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("x{}", i))
            } else {
                feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("x{}", i))
            };

            let coef = self.coefficients[i];
            let se = self.std_errors[i];

            let lower = coef - z_value * se;
            let upper = coef + z_value * se;

            intervals.push((name, lower, upper));
        }

        intervals
    }

    /// Compute deviance residuals
    pub fn deviance_residuals(&self, y: &Array1<f64>) -> Array1<f64> {
        y.iter()
            .zip(self.fitted_values.iter())
            .map(|(&y_val, &mu_val)| {
                let unit_deviance = self.family.unit_deviance(y_val, mu_val);
                let sign = if y_val > mu_val { 1.0 } else { -1.0 };
                sign * unit_deviance.sqrt()
            })
            .collect()
    }

    /// Compute Cook's distances for influence diagnostics
    pub fn cooks_distance(&self) -> Array1<f64> {
        let n = self.n_obs;
        let p = self.n_params;

        let mut cooks = Array1::zeros(n);

        for i in 0..n {
            let hii = self.hat_matrix_diag[i];
            let pearson_resid = self.pearson_residuals[i];

            if hii < 1.0 && self.scale > 0.0 {
                cooks[i] =
                    (pearson_resid.powi(2) / (p as f64 * self.scale)) * (hii / (1.0 - hii).powi(2));
            }
        }

        cooks
    }

    /// Compute standardized Pearson residuals
    pub fn standardized_pearson_residuals(&self) -> Array1<f64> {
        let n = self.n_obs;
        let mut std_residuals = Array1::zeros(n);

        for i in 0..n {
            let hii = self.hat_matrix_diag[i];
            let pearson_resid = self.pearson_residuals[i];

            if self.scale > 0.0 && hii < 1.0 {
                std_residuals[i] = pearson_resid / ((self.scale * (1.0 - hii)).sqrt());
            }
        }

        std_residuals
    }

    /// Test for overdispersion (for Poisson and Binomial)
    pub fn overdispersion_test(&self) -> (f64, f64) {
        let n = self.n_obs;
        let p = self.n_params;

        if self.family == Family::Poisson || self.family == Family::Binomial {
            let pearson_chi2: f64 = self.pearson_residuals.iter().map(|r| r.powi(2)).sum();
            let df = (n - p) as f64;

            let test_stat = pearson_chi2 / df;
            let p_value = 1.0 - statrs::function::gamma::gamma_ur(df / 2.0, df / 2.0 * test_stat);

            (test_stat, p_value)
        } else {
            (f64::NAN, f64::NAN)
        }
    }

    /// Compute log-likelihood (up to a constant)
    pub fn log_likelihood(&self, y: &Array1<f64>) -> f64 {
        let n = self.n_obs;
        let mut ll = 0.0;

        for i in 0..n {
            let y_val = y[i];
            let mu_val = self.fitted_values[i];

            // Log-likelihood contributions (up to constant)
            match self.family {
                Family::Gaussian => {
                    ll += -0.5 * (y_val - mu_val).powi(2) / self.scale;
                }
                Family::Binomial => {
                    if y_val == 0.0 {
                        ll += (1.0 - mu_val).ln().max(-100.0);
                    } else if y_val == 1.0 {
                        ll += mu_val.ln().max(-100.0);
                    } else {
                        ll += y_val * mu_val.ln().max(-100.0)
                            + (1.0 - y_val) * (1.0 - mu_val).ln().max(-100.0);
                    }
                }
                Family::Poisson => {
                    ll += y_val * mu_val.ln().max(-100.0) - mu_val;
                }
                Family::Gamma => {
                    // For Gamma with shape parameter = 1/scale
                    let shape = 1.0 / self.scale;
                    ll += shape * mu_val.ln().max(-100.0)
                        - shape * y_val / mu_val
                        - shape.ln().max(-100.0)
                        - (shape - 1.0) * y_val.ln().max(-100.0);
                }
                Family::InverseGaussian => {
                    // For Inverse Gaussian
                    ll += -0.5 * (y_val - mu_val).powi(2) / (self.scale * mu_val.powi(2) * y_val);
                }
            }
        }

        ll
    }

    /// Compute pseudo R-squared measures
    pub fn pseudo_r_squared(&self) -> (f64, f64, f64) {
        // McFadden's R-squared
        let mcfadden = 1.0 - self.deviance / self.null_deviance;

        // Cox & Snell R-squared
        let cox_snell = 1.0 - (self.deviance / self.null_deviance).powf(2.0 / self.n_obs as f64);

        // Nagelkerke R-squared (adjusted Cox & Snell)
        let nagelkerke = cox_snell / (1.0 - (-2.0 * self.null_deviance / self.n_obs as f64).exp());

        (mcfadden, cox_snell, nagelkerke)
    }

    /// Perform likelihood ratio test against null model
    pub fn likelihood_ratio_test(&self) -> (f64, f64) {
        let lr_stat = self.null_deviance - self.deviance;
        let df = (self.n_params - if self.intercept { 1 } else { 0 }) as f64;

        let p_value = 1.0 - statrs::function::gamma::gamma_ur(df / 2.0, lr_stat / 2.0);

        (lr_stat, p_value)
    }

    /// Get model information as a string
    pub fn model_info(&self) -> String {
        format!(
            "GLM(family={}, link={}, intercept={}, n_obs={}, n_params={})",
            self.family.name(),
            self.link.name(),
            self.intercept,
            self.n_obs,
            self.n_params
        )
    }
}
