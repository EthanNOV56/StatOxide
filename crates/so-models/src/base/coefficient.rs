//! Coefficient definition

use serde::{Deserialize, Serialize};

/// Coefficient estimate with statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coefficient {
    /// Coefficient name
    pub name: String,
    /// Coefficient estimate
    pub estimate: f64,
    /// Standard error
    pub std_error: Option<f64>,
    /// t-statistic
    pub t_stat: Option<f64>,
    /// p-value
    pub p_value: Option<f64>,
    /// Lower bound of confidence interval
    pub ci_lower: Option<f64>,
    /// Upper bound of confidence interval
    pub ci_upper: Option<f64>,
    /// Degrees of freedom
    pub df: Option<f64>,
    /// Is this the intercept?
    pub is_intercept: bool,
}

impl Coefficient {
    /// Create a new coefficient
    pub fn new(name: impl Into<String>, estimate: f64) -> Self {
        Self {
            name: name.into(),
            estimate,
            std_error: None,
            t_stat: None,
            p_value: None,
            ci_lower: None,
            ci_upper: None,
            df: None,
            is_intercept: false,
        }
    }

    /// Set standard error
    pub fn with_std_error(mut self, se: f64) -> Self {
        self.std_error = Some(se);
        self
    }

    /// Set t-statistic
    pub fn with_t_stat(mut self, t: f64) -> Self {
        self.t_stat = Some(t);
        self
    }

    /// Set p-value
    pub fn with_p_value(mut self, p: f64) -> Self {
        self.p_value = Some(p);
        self
    }

    /// Set confidence interval
    pub fn with_ci(mut self, lower: f64, upper: f64) -> Self {
        self.ci_lower = Some(lower);
        self.ci_upper = Some(upper);
        self
    }

    /// Mark as intercept
    pub fn as_intercept(mut self) -> Self {
        self.is_intercept = true;
        self
    }
}
