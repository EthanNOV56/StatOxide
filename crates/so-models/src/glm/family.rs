//! Distribution families for Generalized Linear Models

#![allow(non_snake_case)]  // Allow mathematical notation

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Normal, Continuous};
use so_core::error::Result;

/// Distribution families for GLM
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Family {
    /// Gaussian (normal) distribution with identity link
    Gaussian,
    /// Binomial distribution for binary/count data
    Binomial,
    /// Poisson distribution for count data
    Poisson,
    /// Gamma distribution for positive continuous data
    Gamma,
    /// Inverse Gaussian distribution
    InverseGaussian,
}

impl Family {
    /// Get the default link function for this family
    pub fn default_link(&self) -> Link {
        match self {
            Family::Gaussian => Link::Identity,
            Family::Binomial => Link::Logit,
            Family::Poisson => Link::Log,
            Family::Gamma => Link::Inverse,
            Family::InverseGaussian => Link::InverseSquare,
        }
    }
    
    /// Compute variance function V(μ) for the mean μ
    pub fn variance(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => mu * (1.0 - mu),
            Family::Poisson => mu,
            Family::Gamma => mu.powi(2),
            Family::InverseGaussian => mu.powi(3),
        }
    }
    
    /// Compute the unit deviance d(y, μ) for a single observation
    pub fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        match self {
            Family::Gaussian => (y - mu).powi(2),
            Family::Binomial => {
                if y == 0.0 {
                    2.0 * (1.0 - mu).ln().max(-100.0)
                } else if y == 1.0 {
                    2.0 * mu.ln().max(-100.0)
                } else {
                    // For proportion data (0 < y < 1)
                    2.0 * (y * (y / mu).ln().max(-100.0) + 
                          (1.0 - y) * ((1.0 - y) / (1.0 - mu)).ln().max(-100.0))
                }
            },
            Family::Poisson => {
                if mu == 0.0 {
                    if y == 0.0 { 0.0 } else { 2.0 * y }
                } else {
                    2.0 * (y * (y / mu).ln().max(-100.0) - (y - mu))
                }
            },
            Family::Gamma => 2.0 * ((y - mu) / mu - (y / mu).ln()),
            Family::InverseGaussian => (y - mu).powi(2) / (mu.powi(2) * y),
        }
    }
    
    /// Compute total deviance for a set of observations
    pub fn deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> f64 {
        y.iter()
            .zip(mu.iter())
            .map(|(&y_val, &mu_val)| self.unit_deviance(y_val, mu_val))
            .sum()
    }
    
    /// Compute initial values for the response variable
    pub fn initialize(&self, y: &Array1<f64>) -> Array1<f64> {
        match self {
            Family::Gaussian => y.clone(),
            Family::Binomial => {
                // For binary data, apply logit transform with clipping
                y.mapv(|y_val| {
                    let clipped = y_val.max(0.0001).min(0.9999);
                    (clipped / (1.0 - clipped)).ln()
                })
            },
            Family::Poisson => {
                // For Poisson, log transform with offset for zeros
                y.mapv(|y_val| (y_val + 0.5).ln())
            },
            Family::Gamma => {
                // For Gamma, log transform
                y.mapv(|y_val| y_val.max(1e-8).ln())
            },
            Family::InverseGaussian => {
                // For Inverse Gaussian, log transform
                y.mapv(|y_val| y_val.max(1e-8).ln())
            },
        }
    }
    
    /// Check if response values are valid for this family
    pub fn validate_response(&self, y: &Array1<f64>) -> Result<()> {
        match self {
            Family::Gaussian => Ok(()), // Any real value
            Family::Binomial => {
                // Check that values are in [0, 1]
                for &val in y {
                    if !(0.0..=1.0).contains(&val) {
                        return Err(so_core::error::Error::DataError(
                            format!("Binomial response must be in [0, 1], got {}", val)
                        ));
                    }
                }
                Ok(())
            },
            Family::Poisson => {
                // Check that values are non-negative integers (or counts)
                for &val in y {
                    if val < 0.0 {
                        return Err(so_core::error::Error::DataError(
                            format!("Poisson response must be non-negative, got {}", val)
                        ));
                    }
                }
                Ok(())
            },
            Family::Gamma | Family::InverseGaussian => {
                // Check that values are positive
                for &val in y {
                    if val <= 0.0 {
                        return Err(so_core::error::Error::DataError(
                            format!("{} response must be positive, got {}", 
                                match self {
                                    Family::Gamma => "Gamma",
                                    Family::InverseGaussian => "Inverse Gaussian",
                                    _ => unreachable!(),
                                },
                                val
                            )
                        ));
                    }
                }
                Ok(())
            },
        }
    }
    
    /// Get the name of the family as a string
    pub fn name(&self) -> &'static str {
        match self {
            Family::Gaussian => "Gaussian",
            Family::Binomial => "Binomial",
            Family::Poisson => "Poisson",
            Family::Gamma => "Gamma",
            Family::InverseGaussian => "Inverse Gaussian",
        }
    }
    
    /// Compute the dispersion parameter (scale) from Pearson residuals
    pub fn estimate_dispersion(&self, y: &Array1<f64>, mu: &Array1<f64>, n: usize, p: usize) -> f64 {
        let pearson_residuals: f64 = y.iter()
            .zip(mu.iter())
            .map(|(&y_val, &mu_val)| {
                let variance = self.variance(mu_val);
                if variance > 0.0 {
                    (y_val - mu_val).powi(2) / variance
                } else {
                    0.0
                }
            })
            .sum();
        
        pearson_residuals / (n - p) as f64
    }
}

/// Link functions for GLM
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Link {
    /// Identity link: η = μ
    Identity,
    /// Logit link: η = log(μ / (1 - μ))
    Logit,
    /// Probit link: η = Φ⁻¹(μ)
    Probit,
    /// Complementary log-log link: η = log(-log(1 - μ))
    Cloglog,
    /// Log link: η = log(μ)
    Log,
    /// Inverse link: η = 1/μ
    Inverse,
    /// Inverse square link: η = 1/μ²
    InverseSquare,
    /// Square root link: η = √μ
    Sqrt,
}

impl Link {
    /// Apply link function: η = g(μ)
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            Link::Identity => mu,
            Link::Logit => (mu / (1.0 - mu)).ln(),
            Link::Probit => {
                // Approximate inverse normal CDF
                if mu <= 0.0 || mu >= 1.0 {
                    f64::NAN
                } else {
                    statrs::function::erf::erf_inv(2.0 * mu - 1.0) * 2.0f64.sqrt()
                }
            },
            Link::Cloglog => (-(1.0 - mu).ln()).ln(),
            Link::Log => mu.ln(),
            Link::Inverse => 1.0 / mu,
            Link::InverseSquare => 1.0 / mu.powi(2),
            Link::Sqrt => mu.sqrt(),
        }
    }
    
    /// Apply inverse link: μ = g⁻¹(η)
    pub fn inverse_link(&self, eta: f64) -> f64 {
        match self {
            Link::Identity => eta,
            Link::Logit => 1.0 / (1.0 + (-eta).exp()),
            Link::Probit => 0.5 * (1.0 + statrs::function::erf::erf(eta / 2.0f64.sqrt())),
            Link::Cloglog => 1.0 - (-eta.exp()).exp(),
            Link::Log => eta.exp(),
            Link::Inverse => 1.0 / eta,
            Link::InverseSquare => 1.0 / eta.sqrt(),
            Link::Sqrt => eta.powi(2),
        }
    }
    
    /// Derivative of inverse link: dμ/dη
    pub fn derivative(&self, eta: f64) -> f64 {
        match self {
            Link::Identity => 1.0,
            Link::Logit => {
                let mu = self.inverse_link(eta);
                mu * (1.0 - mu)
            },
            Link::Probit => {
                // Derivative of inverse normal CDF is normal PDF
                Normal::new(0.0, 1.0).unwrap().pdf(eta)
            },
            Link::Cloglog => {
                let mu = self.inverse_link(eta);
                (1.0 - mu) * (-(1.0 - mu).ln())
            },
            Link::Log => eta.exp(), // Same as inverse link for log
            Link::Inverse => -1.0 / eta.powi(2),
            Link::InverseSquare => -0.5 / eta.powf(-1.5),
            Link::Sqrt => 2.0 * eta,
        }
    }
    
    /// Get the name of the link function as a string
    pub fn name(&self) -> &'static str {
        match self {
            Link::Identity => "identity",
            Link::Logit => "logit",
            Link::Probit => "probit",
            Link::Cloglog => "cloglog",
            Link::Log => "log",
            Link::Inverse => "inverse",
            Link::InverseSquare => "inverse square",
            Link::Sqrt => "sqrt",
        }
    }
}

/// Check if a link-function combination is valid
pub fn is_valid_link(family: Family, link: Link) -> bool {
    match family {
        Family::Gaussian => matches!(link, Link::Identity | Link::Log | Link::Inverse),
        Family::Binomial => matches!(link, Link::Logit | Link::Probit | Link::Cloglog | Link::Log),
        Family::Poisson => matches!(link, Link::Log | Link::Identity | Link::Sqrt),
        Family::Gamma => matches!(link, Link::Inverse | Link::Log | Link::Identity),
        Family::InverseGaussian => matches!(link, Link::InverseSquare | Link::Inverse | Link::Log | Link::Identity),
    }
}