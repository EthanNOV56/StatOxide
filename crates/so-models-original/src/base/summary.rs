//! Model summary structures

use super::coefficient::Coefficient;
use super::statistics::{ModelStatistics, ResidualStatistics};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Comprehensive model summary structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    /// Model type
    pub model_type: ModelType,
    /// Model formula
    pub formula: String,
    /// Number of observations
    pub n_obs: usize,
    /// Number of predictors (including intercept)
    pub n_predictors: usize,
    /// Coefficients table
    pub coefficients: Vec<Coefficient>,
    /// Model statistics
    pub model_statistics: ModelStatistics,
    /// Residual statistics
    pub residual_statistics: ResidualStatistics,
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model Summary")?;
        writeln!(f, "=============")?;
        writeln!(f, "Model Type: {}", self.model_type)?;
        writeln!(f, "Formula: {}", self.formula)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Predictors: {}", self.n_predictors)?;
        writeln!(f)?;

        // Coefficients
        writeln!(f, "Coefficients:")?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12} {:>12}",
            "Term", "Estimate", "Std Error", "t-value", "p-value"
        )?;
        writeln!(
            f,
            "{:-<20} {:-<12} {:-<12} {:-<12} {:-<12}",
            "", "", "", "", ""
        )?;

        for coeff in &self.coefficients {
            writeln!(
                f,
                "{:<20} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
                coeff.name,
                coeff.estimate,
                coeff.std_error.unwrap_or(f64::NAN),
                coeff.t_stat.unwrap_or(f64::NAN),
                coeff.p_value.unwrap_or(f64::NAN)
            )?;
        }
        writeln!(f)?;

        // Model statistics
        writeln!(f, "Model Statistics:")?;
        if let Some(r2) = self.model_statistics.r_squared {
            writeln!(f, "  R-squared: {:.4}", r2)?;
        }
        if let Some(adj_r2) = self.model_statistics.adj_r_squared {
            writeln!(f, "  Adjusted R-squared: {:.4}", adj_r2)?;
        }
        if let Some(f_stat) = self.model_statistics.f_statistic {
            writeln!(f, "  F-statistic: {:.4}", f_stat)?;
        }
        if let Some(log_lik) = self.model_statistics.log_likelihood {
            writeln!(f, "  Log-likelihood: {:.4}", log_lik)?;
        }
        if let Some(aic) = self.model_statistics.aic {
            writeln!(f, "  AIC: {:.4}", aic)?;
        }
        if let Some(bic) = self.model_statistics.bic {
            writeln!(f, "  BIC: {:.4}", bic)?;
        }
        if let Some(resid_se) = self.model_statistics.residual_std_error {
            writeln!(f, "  Residual Std. Error: {:.4}", resid_se)?;
        }
        if let Some(df_resid) = self.model_statistics.df_residual {
            writeln!(f, "  Residual DF: {}", df_resid)?;
        }
        if let Some(df_model) = self.model_statistics.df_model {
            writeln!(f, "  Model DF: {}", df_model)?;
        }

        Ok(())
    }
}

/// Model type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Linear regression
    LinearRegression,
    /// Logistic regression
    LogisticRegression,
    /// Poisson regression
    PoissonRegression,
    /// Linear mixed model
    LinearMixedModel,
    /// Generalized linear mixed model
    GeneralizedLinearMixedModel,
    /// ARIMA model
    Arima,
    /// Cox proportional hazards model
    CoxPH,
    /// ANOVA
    Anova,
    /// Other model type
    Other(String),
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelType::LinearRegression => write!(f, "Linear Regression"),
            ModelType::LogisticRegression => write!(f, "Logistic Regression"),
            ModelType::PoissonRegression => write!(f, "Poisson Regression"),
            ModelType::LinearMixedModel => write!(f, "Linear Mixed Model"),
            ModelType::GeneralizedLinearMixedModel => write!(f, "GLMM"),
            ModelType::Arima => write!(f, "ARIMA"),
            ModelType::CoxPH => write!(f, "Cox PH"),
            ModelType::Anova => write!(f, "ANOVA"),
            ModelType::Other(s) => write!(f, "{}", s),
        }
    }
}
