//! Probability distributions for statistical computing
//!
//! Provides common probability distributions with PDF, CDF, quantile functions,
//! and random number generation.

use rand::Rng;
use statrs::distribution::{
    Beta, Cauchy, ChiSquared, ContinuousCDF, DiscreteCDF, Exponential, FisherSnedecor, Gamma,
    Laplace, LogNormal, Normal, StudentsT, Triangular, Uniform, Weibull,
};
use statrs::distribution::{Bernoulli, Binomial, Geometric, Hypergeometric, NegativeBinomial, Poisson};
use thiserror::Error;

/// Errors for distribution operations
#[derive(Error, Debug)]
pub enum DistributionError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    #[error("Distribution not supported: {0}")]
    NotSupported(String),
}

/// Result type for distribution operations
pub type Result<T> = std::result::Result<T, DistributionError>;

/// Enum representing common continuous distributions
#[derive(Debug, Clone)]
pub enum ContinuousDistribution {
    Normal { mean: f64, std_dev: f64 },
    StudentsT { df: f64 },
    ChiSquared { df: f64 },
    FisherSnedecor { d1: f64, d2: f64 },
    Exponential { rate: f64 },
    Gamma { shape: f64, rate: f64 },
    Beta { alpha: f64, beta: f64 },
    LogNormal { mu: f64, sigma: f64 },
    Cauchy { location: f64, scale: f64 },
    Weibull { shape: f64, scale: f64 },
    Uniform { lower: f64, upper: f64 },
}

impl ContinuousDistribution {
    /// Create a standard normal distribution
    pub fn standard_normal() -> Self {
        Self::Normal {
            mean: 0.0,
            std_dev: 1.0,
        }
    }
    
    /// Create a normal distribution with given parameters
    pub fn normal(mean: f64, std_dev: f64) -> Result<Self> {
        if std_dev <= 0.0 {
            return Err(DistributionError::InvalidParameter(
                "Standard deviation must be positive".to_string(),
            ));
        }
        Ok(Self::Normal { mean, std_dev })
    }
    
    /// Create a t-distribution
    pub fn students_t(df: f64) -> Result<Self> {
        if df <= 0.0 {
            return Err(DistributionError::InvalidParameter(
                "Degrees of freedom must be positive".to_string(),
            ));
        }
        Ok(Self::StudentsT { df })
    }
    
    /// Create a chi-squared distribution
    pub fn chi_squared(df: f64) -> Result<Self> {
        if df <= 0.0 {
            return Err(DistributionError::InvalidParameter(
                "Degrees of freedom must be positive".to_string(),
            ));
        }
        Ok(Self::ChiSquared { df })
    }
    
    /// Create an F-distribution
    pub fn fisher_snedecor(d1: f64, d2: f64) -> Result<Self> {
        if d1 <= 0.0 || d2 <= 0.0 {
            return Err(DistributionError::InvalidParameter(
                "Degrees of freedom must be positive".to_string(),
            ));
        }
        Ok(Self::FisherSnedecor { d1, d2 })
    }
    
    /// Probability density function
    pub fn pdf(&self, x: f64) -> f64 {
        match self {
            Self::Normal { mean, std_dev } => {
                let dist = Normal::new(*mean, *std_dev).unwrap();
                dist.pdf(x)
            }
            Self::StudentsT { df } => {
                let dist = StudentsT::new(0.0, 1.0, *df).unwrap();
                dist.pdf(x)
            }
            Self::ChiSquared { df } => {
                let dist = ChiSquared::new(*df).unwrap();
                dist.pdf(x)
            }
            Self::FisherSnedecor { d1, d2 } => {
                let dist = FisherSnedecor::new(*d1, *d2).unwrap();
                dist.pdf(x)
            }
            Self::Exponential { rate } => {
                let dist = Exponential::new(*rate).unwrap();
                dist.pdf(x)
            }
            Self::Gamma { shape, rate } => {
                let dist = Gamma::new(*shape, *rate).unwrap();
                dist.pdf(x)
            }
            Self::Beta { alpha, beta } => {
                let dist = Beta::new(*alpha, *beta).unwrap();
                dist.pdf(x)
            }
            Self::LogNormal { mu, sigma } => {
                let dist = LogNormal::new(*mu, *sigma).unwrap();
                dist.pdf(x)
            }
            Self::Cauchy { location, scale } => {
                let dist = Cauchy::new(*location, *scale).unwrap();
                dist.pdf(x)
            }
            Self::Weibull { shape, scale } => {
                let dist = Weibull::new(*shape, *scale).unwrap();
                dist.pdf(x)
            }
            Self::Uniform { lower, upper } => {
                if x < *lower || x > *upper {
                    0.0
                } else {
                    1.0 / (upper - lower)
                }
            }
        }
    }
    
    /// Cumulative distribution function
    pub fn cdf(&self, x: f64) -> f64 {
        match self {
            Self::Normal { mean, std_dev } => {
                let dist = Normal::new(*mean, *std_dev).unwrap();
                dist.cdf(x)
            }
            Self::StudentsT { df } => {
                let dist = StudentsT::new(0.0, 1.0, *df).unwrap();
                dist.cdf(x)
            }
            Self::ChiSquared { df } => {
                let dist = ChiSquared::new(*df).unwrap();
                dist.cdf(x)
            }
            Self::FisherSnedecor { d1, d2 } => {
                let dist = FisherSnedecor::new(*d1, *d2).unwrap();
                dist.cdf(x)
            }
            Self::Exponential { rate } => {
                let dist = Exponential::new(*rate).unwrap();
                dist.cdf(x)
            }
            Self::Gamma { shape, rate } => {
                let dist = Gamma::new(*shape, *rate).unwrap();
                dist.cdf(x)
            }
            Self::Beta { alpha, beta } => {
                let dist = Beta::new(*alpha, *beta).unwrap();
                dist.cdf(x)
            }
            Self::LogNormal { mu, sigma } => {
                let dist = LogNormal::new(*mu, *sigma).unwrap();
                dist.cdf(x)
            }
            Self::Cauchy { location, scale } => {
                let dist = Cauchy::new(*location, *scale).unwrap();
                dist.cdf(x)
            }
            Self::Weibull { shape, scale } => {
                let dist = Weibull::new(*shape, *scale).unwrap();
                dist.cdf(x)
            }
            Self::Uniform { lower, upper } => {
                if x < *lower {
                    0.0
                } else if x > *upper {
                    1.0
                } else {
                    (x - lower) / (upper - lower)
                }
            }
        }
    }
    
    /// Quantile function (inverse CDF)
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }
        
        match self {
            Self::Normal { mean, std_dev } => {
                let dist = Normal::new(*mean, *std_dev).unwrap();
                Some(dist.inverse_cdf(p))
            }
            Self::StudentsT { df } => {
                let dist = StudentsT::new(0.0, 1.0, *df).unwrap();
                Some(dist.inverse_cdf(p))
            }
            Self::ChiSquared { df } => {
                let dist = ChiSquared::new(*df).unwrap();
                Some(dist.inverse_cdf(p))
            }
            Self::FisherSnedecor { d1, d2 } => {
                let dist = FisherSnedecor::new(*d1, *d2).unwrap();
                Some(dist.inverse_cdf(p))
            }
            Self::Exponential { rate } => {
                let dist = Exponential::new(*rate).unwrap();
                Some(dist.inverse_cdf(p))
            }
            Self::Gamma { shape, rate } => {
                let dist = Gamma::new(*shape, *rate).unwrap();
                Some(dist.inverse_cdf(p))
            }
            Self::Beta { alpha, beta } => {
                let dist = Beta::new(*alpha, *beta).unwrap();
                Some(dist.inverse_cdf(p))
            }
            Self::LogNormal { mu, sigma } => {
                let dist = LogNormal::new(*mu, *sigma).unwrap();
                Some(dist.inverse_cdf(p))
            }
            Self::Cauchy { location, scale } => {
                let dist = Cauchy::new(*location, *scale).unwrap();
                Some(dist.inverse_cdf(p))
            }
            Self::Weibull { shape, scale } => {
                let dist = Weibull::new(*shape, *scale).unwrap();
                Some(dist.inverse_cdf(p))
            }
            Self::Uniform { lower, upper } => {
                Some(lower + p * (upper - lower))
            }
        }
    }
    
    /// Generate random sample from distribution
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            Self::Normal { mean, std_dev } => {
                let dist = Normal::new(*mean, *std_dev).unwrap();
                dist.sample(rng)
            }
            Self::StudentsT { df } => {
                let dist = StudentsT::new(0.0, 1.0, *df).unwrap();
                dist.sample(rng)
            }
            Self::ChiSquared { df } => {
                let dist = ChiSquared::new(*df).unwrap();
                dist.sample(rng)
            }
            Self::FisherSnedecor { d1, d2 } => {
                let dist = FisherSnedecor::new(*d1, *d2).unwrap();
                dist.sample(rng)
            }
            Self::Exponential { rate } => {
                let dist = Exponential::new(*rate).unwrap();
                dist.sample(rng)
            }
            Self::Gamma { shape, rate } => {
                let dist = Gamma::new(*shape, *rate).unwrap();
                dist.sample(rng)
            }
            Self::Beta { alpha, beta } => {
                let dist = Beta::new(*alpha, *beta).unwrap();
                dist.sample(rng)
            }
            Self::LogNormal { mu, sigma } => {
                let dist = LogNormal::new(*mu, *sigma).unwrap();
                dist.sample(rng)
            }
            Self::Cauchy { location, scale } => {
                let dist = Cauchy::new(*location, *scale).unwrap();
                dist.sample(rng)
            }
            Self::Weibull { shape, scale } => {
                let dist = Weibull::new(*shape, *scale).unwrap();
                dist.sample(rng)
            }
            Self::Uniform { lower, upper } => {
                rng.gen_range(*lower..=*upper)
            }
        }
    }
}

/// Enum representing common discrete distributions
#[derive(Debug, Clone)]
pub enum DiscreteDistribution {
    Bernoulli { p: f64 },
    Binomial { n: u64, p: f64 },
    Poisson { lambda: f64 },
    Geometric { p: f64 },
    NegativeBinomial { r: f64, p: f64 },
    Hypergeometric { N: u64, K: u64, n: u64 },
}

impl DiscreteDistribution {
    /// Create a Bernoulli distribution
    pub fn bernoulli(p: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(DistributionError::InvalidParameter(
                "Probability must be between 0 and 1".to_string(),
            ));
        }
        Ok(Self::Bernoulli { p })
    }
    
    /// Create a binomial distribution
    pub fn binomial(n: u64, p: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(DistributionError::InvalidParameter(
                "Probability must be between 0 and 1".to_string(),
            ));
        }
        Ok(Self::Binomial { n, p })
    }
    
    /// Create a Poisson distribution
    pub fn poisson(lambda: f64) -> Result<Self> {
        if lambda <= 0.0 {
            return Err(DistributionError::InvalidParameter(
                "Lambda must be positive".to_string(),
            ));
        }
        Ok(Self::Poisson { lambda })
    }
    
    /// Create a geometric distribution
    pub fn geometric(p: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(DistributionError::InvalidParameter(
                "Probability must be between 0 and 1".to_string(),
            ));
        }
        Ok(Self::Geometric { p })
    }
    
    /// Create a negative binomial distribution
    pub fn negative_binomial(r: f64, p: f64) -> Result<Self> {
        if r <= 0.0 || !(0.0..=1.0).contains(&p) {
            return Err(DistributionError::InvalidParameter(
                "r must be positive and p must be between 0 and 1".to_string(),
            ));
        }
        Ok(Self::NegativeBinomial { r, p })
    }
    
    /// Create a hypergeometric distribution
    pub fn hypergeometric(N: u64, K: u64, n: u64) -> Result<Self> {
        if n > N || K > N {
            return Err(DistributionError::InvalidParameter(
                "Invalid parameters for hypergeometric distribution".to_string(),
            ));
        }
        Ok(Self::Hypergeometric { N, K, n })
    }
    
    /// Probability mass function
    pub fn pmf(&self, k: u64) -> f64 {
        match self {
            Self::Bernoulli { p } => {
                let dist = Bernoulli::new(*p).unwrap();
                dist.pmf(k as u64)
            }
            Self::Binomial { n, p } => {
                let dist = Binomial::new(*p, *n).unwrap();
                dist.pmf(k as u64)
            }
            Self::Poisson { lambda } => {
                let dist = Poisson::new(*lambda).unwrap();
                dist.pmf(k as u64)
            }
            Self::Geometric { p } => {
                let dist = Geometric::new(*p).unwrap();
                dist.pmf(k as u64)
            }
            Self::NegativeBinomial { r, p } => {
                let dist = NegativeBinomial::new(*r, *p).unwrap();
                dist.pmf(k as u64)
            }
            Self::Hypergeometric { N, K, n } => {
                let dist = Hypergeometric::new(*N, *K, *n).unwrap();
                dist.pmf(k as u64)
            }
        }
    }
    
    /// Cumulative distribution function
    pub fn cdf(&self, k: u64) -> f64 {
        match self {
            Self::Bernoulli { p } => {
                let dist = Bernoulli::new(*p).unwrap();
                dist.cdf(k as u64)
            }
            Self::Binomial { n, p } => {
                let dist = Binomial::new(*p, *n).unwrap();
                dist.cdf(k as u64)
            }
            Self::Poisson { lambda } => {
                let dist = Poisson::new(*lambda).unwrap();
                dist.cdf(k as u64)
            }
            Self::Geometric { p } => {
                let dist = Geometric::new(*p).unwrap();
                dist.cdf(k as u64)
            }
            Self::NegativeBinomial { r, p } => {
                let dist = NegativeBinomial::new(*r, *p).unwrap();
                dist.cdf(k as u64)
            }
            Self::Hypergeometric { N, K, n } => {
                let dist = Hypergeometric::new(*N, *K, *n).unwrap();
                dist.cdf(k as u64)
            }
        }
    }
    
    /// Generate random sample from distribution
    pub fn sample<R: Rng>(&self, rng: &mut R) -> u64 {
        match self {
            Self::Bernoulli { p } => {
                let dist = Bernoulli::new(*p).unwrap();
                dist.sample(rng) as u64
            }
            Self::Binomial { n, p } => {
                let dist = Binomial::new(*p, *n).unwrap();
                dist.sample(rng)
            }
            Self::Poisson { lambda } => {
                let dist = Poisson::new(*lambda).unwrap();
                dist.sample(rng)
            }
            Self::Geometric { p } => {
                let dist = Geometric::new(*p).unwrap();
                dist.sample(rng)
            }
            Self::NegativeBinomial { r, p } => {
                let dist = NegativeBinomial::new(*r, *p).unwrap();
                dist.sample(rng)
            }
            Self::Hypergeometric { N, K, n } => {
                let dist = Hypergeometric::new(*N, *K, *n).unwrap();
                dist.sample(rng)
            }
        }
    }
}