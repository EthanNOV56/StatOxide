//! Mixed Effects Models (LMM and GLMM)
//!
//! This module implements mixed effects models for analyzing data with
//! hierarchical or grouped structure, combining fixed and random effects.
//!
//! # Key Components
//!
//! 1. **Linear Mixed Models (LMM)**: For normally distributed responses
//! 2. **Generalized Linear Mixed Models (GLMM)**: For non-normal responses
//! 3. **Random Effects**: Intercept-only, slope, or correlated random effects
//! 4. **Estimation Methods**: Restricted Maximum Likelihood (REML) and Maximum Likelihood (ML)
//! 5. **Model Diagnostics**: Likelihood ratio tests, AIC, BIC, random effect predictions
//!
//! # Model Specification
//!
//! Mixed effects models extend the standard linear model:
//! y = Xβ + Zb + ε
//!
//! where:
//! - X: Fixed effects design matrix
//! - β: Fixed effects coefficients
//! - Z: Random effects design matrix  
//! - b ~ N(0, G): Random effects with covariance matrix G
//! - ε ~ N(0, R): Residual errors with covariance matrix R
//!
//! # Example Usage
//!
//! ```rust
//! use statoxide::mixed::{LMM, RandomEffect, CovarianceStructure};
//! use statoxide::DataFrame;
//!
//! // Create a random intercept model: y ~ x1 + (1 | group)
//! let model = LMM::new()
//!     .fixed("y ~ x1 + x2")
//!     .random(RandomEffect::intercept("group"))
//!     .method(EstimationMethod::REML);
//!
//! // Fit the model
//! let results = model.fit(&data)?;
//! ```
//!
//! # References
//!
//! - Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). *Fitting Linear Mixed-Effects Models Using lme4*.
//! - Pinheiro, J. C., & Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*.
//! - R's `lme4` and `nlme` packages.

pub mod model;
pub mod results;
pub mod random_effects;
pub mod covariance;

// Re-exports for convenience
pub use model::{LMM, GLMM, MixedModelBuilder, EstimationMethod};
pub use results::{MixedModelResults, RandomEffectResults};
pub use random_effects::{RandomEffect, RandomEffectType};
pub use covariance::{CovarianceStructure, CovarianceType};

// Common prelude for mixed effects models
pub mod prelude {
    pub use super::{
        LMM, GLMM, MixedModelBuilder, EstimationMethod,
        MixedModelResults, RandomEffectResults,
        RandomEffect, RandomEffectType,
        CovarianceStructure, CovarianceType,
    };
}