//! Generalized Linear Models (GLM)
//!
//! This module implements Generalized Linear Models, which extend
//! linear regression to response variables with non-normal distributions.
//!
//! # Key Components
//!
//! 1. **Distribution Families**: Gaussian, Binomial, Poisson, Gamma, Inverse Gaussian
//! 2. **Link Functions**: Identity, Logit, Log, Inverse, etc.
//! 3. **IRLS Algorithm**: Iteratively Reweighted Least Squares for estimation
//! 4. **Model Diagnostics**: Deviance, Pearson chi-squared, AIC, BIC
//!
//! # Example Usage
//!
//! ```rust
//! use statoxide::glm::{GLM, Family, Link};
//! use statoxide::DataFrame;
//!
//! // Create a logistic regression model
//! let model = GLM::new()
//!     .family(Family::Binomial)
//!     .link(Link::Logit)
//!     .intercept(true);
//!
//! // Fit the model
//! let results = model.fit("y ~ x1 + x2", &data)?;
//! ```
//!
//! # References
//!
//! - McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*.
//! - R's `glm()` function and statsmodels' GLM implementation.

pub mod family;
pub mod model;
pub mod results;

// Re-exports for convenience
pub use family::{Family, Link, is_valid_link};
pub use model::{GLM, GLMModelBuilder};
pub use results::GLMResults;

// Common prelude for GLM
pub mod prelude {
    pub use super::{Family, Link, GLM, GLMModelBuilder, GLMResults, is_valid_link};
}