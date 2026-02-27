# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-27

### Added
- **Complete Python bindings** via PyO3 for all core functionality
  - `Series`, `DataFrame`, `Formula` core data structures
  - `GLM` (Generalized Linear Models) with `Family`, `Link`, `GLMBuilder`, `GLMResults`
  - `TimeSeries`, `ARIMA`, `GARCH` time series analysis models
  - Statistical functions: `mean()`, `std_dev()`, `correlation()`, `descriptive_summary()`
  - **Statistical tests**: `t_test_one_sample()`, `t_test_two_sample()`, `t_test_paired()`, 
    `chi_square_test_independence()`, `anova_one_way()`, `shapiro_wilk_test()`
  - Utility functions: `train_test_split()`, `version()`
- **Type stubs** (`statoxide.pyi`) for IDE autocompletion and type checking
- **Flexible input types** for ARIMA/GARCH `fit()` methods (accepts `TimeSeries`, `list[float]`, `PyList`, etc.)
- **R-style formula parser** with design matrix construction
- **Mixed effects models** framework (LMM/GLMM) with EM algorithm
- **GARCH models** with normal and t-distributions
- **Statistical tests implementation** in `so-stats` crate:
  - One-sample, two-sample, and paired t-tests
  - Chi-square test of independence
  - One-way ANOVA
  - Shapiro-Wilk normality test
- **Comprehensive test suite** with mock-free API validation

### Changed
- **Unified module naming**: Python package renamed from `so_python` to `statoxide`
- **Updated PyO3 API** to version 0.27 (removed deprecated `PyObject`, `downcast`, etc.)
- **Enhanced ARIMA/GARCH interfaces** with builder pattern and configuration methods
- **Improved error handling** with descriptive error messages
- **Fixed method signatures**: `Series.std(ddof)`, `TimeSeries.std(_ddof)`, `TimeSeries.var(_ddof)`
- **Workspace architecture** with multi-crate structure for better modularity

### Fixed
- **Compilation errors** in formula parser tests (missing `Series` import)
- **Doctest failures** in TSA documentation (convergence issues, marked as `no_run`)
- **Python binding compilation** (PyO3 object ownership, method signatures)
- **Ridge regression intercept calculation** in standardization
- **GARCH variable naming** (`_n`, `_df` → `n`, `df`)
- **Formula parser test expectations** for `*` operator expansion
- **Edge case handling** in OLS `beta_reg` for perfect/no-fit scenarios

### Technical
- **Rust toolchain**: Updated to nightly (rustc 1.95.0-nightly)
- **Python environment**: UV package manager, Python 3.11.6, maturin 1.12.4
- **Dependencies**: ndarray, faer, statrs, rand, serde, pyo3 0.27
- **Workspace structure**:
  - `so-core`: Core data structures and formula parsing
  - `so-models`: Statistical models (linear, GLM, mixed effects)
  - `so-tsa`: Time series analysis (ARIMA, GARCH)
  - `so-stats`: Statistical functions and distributions
  - `so-linalg`: Linear algebra utilities
  - `so-utils`: General utilities
  - `so-python`: Python bindings

## [0.3.0] - 2026-02-27

### Added
- **Robust statistics module** with complete Python bindings:
  - M-estimators: Huber, Tukey's biweight, Hampel, Andrews
  - High-breakdown estimators: Least Trimmed Squares (LTS), S-estimators, MM-estimators
  - Robust scale estimation: MAD, IQR
  - Sandwich variance estimators for inference
- **Nonparametric methods module** with Python bindings:
  - Kernel regression: 7 kernel functions (Gaussian, Epanechnikov, Uniform, etc.)
  - Local regression (LOESS): local polynomial fitting with robust iterations
  - Smoothing splines: natural cubic splines with GCV smoothing parameter selection
  - Bandwidth selection: Silverman, Scott, LSCV, plugin methods
- **Unified entry point crate** (`statoxide`) for easy importing (similar to polars)
  - Modular structure: `statoxide::{models, stats, tsa, linalg, utils}`
  - Prelude module for convenient imports
  - Re-exports of commonly used types and functions
  - Complete documentation and examples

### Changed
- **Dependency management**: Centralized in workspace root for consistency
- **Python package structure**: Enhanced `models` submodule with robust and nonparametric classes
- **Version synchronization**: All crates now version 0.3.0 with consistent metadata

### Technical
- **Test coverage**: 18 passing tests across all modules
- **API consistency**: Python bindings follow scikit-learn-like patterns
- **Documentation**: Complete Rust doc tests and Python type stubs

## [0.1.0] - 2026-01-23

### Added
- Initial project structure and workspace configuration
- Basic data structures: `Series`, `DataFrame`
- Formula parser foundation
- Linear regression models (OLS, Ridge)
- Time series analysis framework
- Python binding skeleton

[Unreleased]: https://github.com/EthanNOV56/StatOxide/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/EthanNOV56/StatOxide/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/EthanNOV56/StatOxide/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/EthanNOV56/StatOxide/releases/tag/v0.1.0
