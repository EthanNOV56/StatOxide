<img src="assets/logo.png" width="330">

# StatOxide: High-performance statistical computing in Rust with Python bindings

**StatOxide** is a modern, high-performance statistical computing library written in Rust, with comprehensive Python bindings via PyO3. Designed for data scientists, statisticians, and researchers who need both performance and productivity.

## 🚀 Features

### 📊 Core Data Structures
- **Series**: Columnar data with metadata (name, dtype, levels)
- **DataFrame**: Tabular data structure with column operations
- **Formula**: R-style formula parsing for model specification

### 📈 Statistical Functions
- **Descriptive Statistics**: Mean, variance, skewness, kurtosis, quantiles
- **Probability Distributions**: 12 continuous + 6 discrete distributions
- **Statistical Tests**: t-test, chi-square, ANOVA, correlation tests
- **Correlation Measures**: Pearson, Spearman, Kendall tau

### 🧮 Statistical Models
- **Linear Models**: OLS, Ridge, Lasso, Elastic Net with proper inference
- **Generalized Linear Models**: Logistic, Poisson, Gamma, Negative Binomial regression
- **Mixed Effects Models**: Linear and GLMMs with EM algorithm estimation
- **Robust Statistics**: M-estimators, S-estimators, MM-estimators
- **Nonparametric Methods**: Kernel regression, local regression, smoothing splines

### 📉 Time Series Analysis
- **Core Structures**: `TimeSeries` with datetime indexing
- **ARIMA Models**: AR, MA, ARMA, ARIMA, SARIMA
- **GARCH Models**: ARCH, GARCH for volatility modeling
- **Decomposition**: STL, moving averages, Hodrick-Prescott filter
- **Forecasting**: Point forecasts, prediction intervals

### 🛠️ Utilities
- **Linear Algebra**: Matrix operations, solvers, decompositions
- **Random Generation**: Distributions, bootstrap, train-test split
- **Data Validation**: Type checking, missing value detection
- **Numerical Methods**: Softmax, standardization, normalization

## 🐍 Python API

StatOxide provides a complete Python interface through PyO3 bindings:

```python
import statoxide
import statoxide.core as soc
import statoxide.stats as sos

# Core data structures
df = soc.DataFrame({
    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [2.0, 4.0, 5.0, 4.0, 5.0]
})

series = df.get_column("x")
print(f"Mean of x: {series.mean():.2f}")
print(f"Std of x: {series.std(1.0):.2f}")

# Statistical functions
print(f"Correlation: {sos.correlation(df.get_column('x').to_list(), 
                                      df.get_column('y').to_list()):.3f}")

summary = sos.descriptive_summary([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Summary: {summary}")

# Formula parsing
formula = soc.Formula("y ~ x + x^2")
print(f"Formula variables: {formula.variables()}")

# Models
import statoxide.models as som
result = som.linear_regression([[1, 1], [1, 2], [1, 3]], [5, 8, 11])
print(f"Regression coefficients: {result['coefficients']}")

# Mixed effects models
mixed_results = som.mixed_effects("y ~ x + (1 | group)", data)
print(f"Random effect variance: {mixed_results.random_variances}")

# Time series
import statoxide.tsa as sot
arima_result = sot.fit_arima([1.0, 2.0, 3.0, 4.0, 5.0], 1, 0, 1)
print(f"ARIMA AIC: {arima_result['aic']}")

# Utilities
import statoxide.utils as sou
train, test = sou.train_test_split([1.0, 2.0, 3.0, 4.0, 5.0], 0.2)
print(f"Train: {train}, Test: {test}")
```

## 🏗️ Architecture

StatOxide is organized as a multi-crate Rust workspace:

```
statoxide/
├── Cargo.toml              # Workspace configuration
├── crates/
│   ├── so-core/           # Core data structures & formula parsing
│   ├── so-linalg/         # Linear algebra abstraction
│   ├── so-stats/          # Statistical functions & distributions
│   ├── so-models/         # Statistical models (regression, GLM, mixed effects, etc.)
│   ├── so-tsa/            # Time series analysis
│   ├── so-utils/          # Utility functions
│   └── so-python/         # Python bindings (PyO3)
├── assets/logo.png        # Project logo
└── LICENSE               # MIT license
```

## 📦 Installation

### Prerequisites

1. **Rust Toolchain**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. **Python Development Files**:
   - Ubuntu/Debian: `sudo apt-get install python3-dev python3.11-dev`
   - macOS: `brew install python@3.11`
3. **Maturin** (recommended): `pip install maturin`

### Building from Source

```bash
# Clone the repository
git clone https://github.com/EthanNOV56/so-dev.git
cd so-dev

# Build Python bindings with maturin
cd crates/so-python
maturin develop  # Editable install for development
# or
maturin build --release  # Build wheel for distribution
```

### Direct Cargo Build

```bash
cd /path/to/statoxide
export PYO3_PYTHON=python3.11
cargo build --release --package so-python
```

The shared library will be at `target/release/libso_python.so`.

## 🧪 Testing

### Rust Tests
```bash
cargo test --all
```

### Python Tests
After installation:
```bash
python -c "import statoxide; print(statoxide.version())"
python crates/so-python/test_api.py  # API demonstration
```

## 📚 Documentation

- **API Reference**: Run `cargo doc --all --no-deps --open` for Rust documentation
- **Python Docstrings**: All Python functions include detailed docstrings
- **Examples**: See `crates/so-python/test_api.py` for usage examples

## 🎯 Design Principles

1. **Performance**: Leverage Rust's zero-cost abstractions and LLVM optimizations
2. **Safety**: Memory safety guarantees without garbage collection
3. **Interoperability**: Seamless Python integration with minimal overhead
4. **Modularity**: Independent crates for clear separation of concerns
5. **API Consistency**: Familiar interfaces inspired by R, pandas, and statsmodels

## 🔧 Development Status

| Module | Status | Notes |
|--------|--------|-------|
| **so-core** | ✅ Complete | Data structures, formula parsing |
| **so-linalg** | ✅ Complete | Linear algebra abstraction |
| **so-stats** | ✅ Complete | Statistical functions & distributions |
| **so-models** | ✅ Complete | Regression, GLM, mixed effects, robust, nonparametric |
| **so-tsa** | ✅ Complete | ARIMA, GARCH, decomposition, forecasting |
| **so-utils** | ✅ Complete | Random generation, validation, numerical methods |
| **so-python** | ✅ **Complete** | **Full Python bindings implemented** |

## 📄 License

MIT OR Apache-2.0 - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **R** and **statsmodels** for statistical API inspiration
- **pandas** for DataFrame design patterns
- **PyO3** team for excellent Rust-Python interop
- **ndarray** and **faer** for numerical computing foundations

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `cargo test --all`
5. Submit a pull request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/EthanNOV56/so-dev/issues)
- **Repository**: [GitHub Repository](https://github.com/EthanNOV56/so-dev)

---

<p align="center">
  <em>High-performance statistics meets Python productivity</em>
</p>
