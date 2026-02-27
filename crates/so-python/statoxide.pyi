"""
StatOxide Python bindings - Type stubs for IDE support

This file provides type hints and autocompletion for the StatOxide library.
StatOxide is a statistical computing library implemented in Rust with Python bindings.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, overload
import numpy as np

# ============================================================================
# Core Data Structures
# ============================================================================

class Series:
    """One-dimensional array-like object with statistical methods."""
    
    def __init__(self, name: str, data: List[float]) -> None:
        """Create a new Series.
        
        Args:
            name: Name of the series
            data: List of floating-point values
        """
        ...
    
    @property
    def name(self) -> str:
        """Get the name of the series."""
        ...
    
    @property
    def len(self) -> int:
        """Get the length of the series."""
        ...
    
    def is_empty(self) -> bool:
        """Check if the series is empty."""
        ...
    
    def mean(self) -> Optional[float]:
        """Compute the mean of the series."""
        ...
    
    def std(self, ddof: float = 1.0) -> Optional[float]:
        """Compute the standard deviation.
        
        Args:
            ddof: Delta degrees of freedom (default: 1.0 for sample std)
        """
        ...
    
    def var(self, ddof: float = 1.0) -> Optional[float]:
        """Compute the variance.
        
        Args:
            ddof: Delta degrees of freedom (default: 1.0 for sample var)
        """
        ...
    
    def min(self) -> Optional[float]:
        """Get the minimum value."""
        ...
    
    def max(self) -> Optional[float]:
        """Get the maximum value."""
        ...
    
    def quantile(self, q: float) -> Optional[float]:
        """Compute the q-th quantile (0 <= q <= 1)."""
        ...
    
    def to_list(self) -> List[float]:
        """Convert series to Python list."""
        ...
    
    def __repr__(self) -> str:
        ...


class DataFrame:
    """Two-dimensional tabular data structure."""
    
    def __init__(self, data: Dict[str, List[float]]) -> None:
        """Create a new DataFrame from a dictionary of columns.
        
        Args:
            data: Dictionary mapping column names to lists of values
        """
        ...
    
    @property
    def n_rows(self) -> int:
        """Get the number of rows."""
        ...
    
    @property
    def n_cols(self) -> int:
        """Get the number of columns."""
        ...
    
    def columns(self) -> List[str]:
        """Get the list of column names."""
        ...
    
    def get_column(self, name: str) -> Series:
        """Get a column as a Series.
        
        Args:
            name: Column name
            
        Raises:
            ValueError: If column not found
        """
        ...
    
    def with_column(self, series: Series) -> 'DataFrame':
        """Add or replace a column.
        
        Args:
            series: Series to add
            
        Returns:
            New DataFrame with the added column
        """
        ...
    
    def __repr__(self) -> str:
        ...


class Formula:
    """R-style formula for statistical models."""
    
    def __init__(self, formula: str) -> None:
        """Create a formula from a string.
        
        Args:
            formula: R-style formula like 'y ~ x1 + x2'
        """
        ...
    
    def variables(self) -> List[str]:
        """Get the variable names in the formula."""
        ...
    
    def __repr__(self) -> str:
        ...


# ============================================================================
# GLM (Generalized Linear Models)
# ============================================================================

class Family:
    """GLM family (distribution)."""
    
    def __init__(self) -> None:
        ...
    
    @staticmethod
    def gaussian() -> 'Family':
        """Gaussian (normal) distribution."""
        ...
    
    @staticmethod
    def binomial() -> 'Family':
        """Binomial distribution."""
        ...
    
    @staticmethod
    def poisson() -> 'Family':
        """Poisson distribution."""
        ...
    
    @staticmethod
    def gamma() -> 'Family':
        """Gamma distribution."""
        ...
    
    @staticmethod
    def inverse_gaussian() -> 'Family':
        """Inverse Gaussian distribution."""
        ...
    
    def name(self) -> str:
        """Get the family name."""
        ...


class Link:
    """GLM link function."""
    
    def __init__(self) -> None:
        ...
    
    @staticmethod
    def identity() -> 'Link':
        """Identity link: η = μ."""
        ...
    
    @staticmethod
    def log() -> 'Link':
        """Log link: η = log(μ)."""
        ...
    
    @staticmethod
    def logit() -> 'Link':
        """Logit link: η = log(μ/(1-μ))."""
        ...
    
    @staticmethod
    def probit() -> 'Link':
        """Probit link: η = Φ⁻¹(μ)."""
        ...
    
    @staticmethod
    def inverse() -> 'Link':
        """Inverse link: η = 1/μ."""
        ...


class GLMBuilder:
    """Builder for GLM configuration."""
    
    def __init__(self) -> None:
        ...
    
    def family(self, family: Family) -> 'GLMBuilder':
        """Set the family (distribution)."""
        ...
    
    def link(self, link: Link) -> 'GLMBuilder':
        """Set the link function."""
        ...
    
    def intercept(self, intercept: bool) -> 'GLMBuilder':
        """Include intercept term."""
        ...
    
    def max_iter(self, max_iter: int) -> 'GLMBuilder':
        """Set maximum IRLS iterations."""
        ...
    
    def scale(self, scale: float) -> 'GLMBuilder':
        """Set dispersion parameter."""
        ...
    
    def tol(self, tol: float) -> 'GLMBuilder':
        """Set convergence tolerance."""
        ...
    
    def build(self) -> 'GLM':
        """Build the GLM instance."""
        ...


class GLM:
    """Generalized Linear Model."""
    
    def __init__(self) -> None:
        ...
    
    @staticmethod
    def new() -> 'GLM':
        """Create a new GLM instance."""
        ...
    
    def fit(self, formula: Formula, data: DataFrame) -> 'GLMResults':
        """Fit model using formula and data."""
        ...
    
    def fit_matrix(self, x: List[List[float]], y: List[float]) -> 'GLMResults':
        """Fit model using design matrix and response vector."""
        ...


class GLMResults:
    """Results from GLM fitting."""
    
    @property
    def coefficients(self) -> List[float]:
        """Estimated coefficients."""
        ...
    
    @property
    def std_errors(self) -> List[float]:
        """Standard errors of coefficients."""
        ...
    
    @property
    def z_values(self) -> List[float]:
        """Z-values for coefficient tests."""
        ...
    
    @property
    def p_values(self) -> List[float]:
        """P-values for coefficient tests."""
        ...
    
    @property
    def fitted_values(self) -> List[float]:
        """Fitted values."""
        ...
    
    @property
    def pearson_residuals(self) -> List[float]:
        """Pearson residuals."""
        ...
    
    @property
    def deviance(self) -> float:
        """Model deviance."""
        ...
    
    @property
    def null_deviance(self) -> float:
        """Null deviance."""
        ...
    
    @property
    def df_residual(self) -> int:
        """Residual degrees of freedom."""
        ...
    
    @property
    def df_null(self) -> int:
        """Null degrees of freedom."""
        ...
    
    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        ...
    
    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        ...
    
    @property
    def scale(self) -> float:
        """Dispersion parameter."""
        ...
    
    @property
    def iterations(self) -> int:
        """Number of IRLS iterations."""
        ...
    
    @property
    def converged(self) -> bool:
        """Whether IRLS converged."""
        ...
    
    def predict(self, x: List[List[float]]) -> List[float]:
        """Predict response for new data."""
        ...
    
    def summary(self) -> str:
        """Get summary statistics as string."""
        ...


# ============================================================================
# Time Series Analysis
# ============================================================================

class TimeSeries:
    """Time series data structure."""
    
    def __init__(self) -> None:
        ...
    
    @staticmethod
    def from_vectors(values: List[float], timestamps: List[str]) -> 'TimeSeries':
        """Create from value and timestamp vectors."""
        ...
    
    @staticmethod
    def from_dataframe(df: DataFrame, value_col: str, timestamp_col: str) -> 'TimeSeries':
        """Create from DataFrame columns."""
        ...
    
    @property
    def len(self) -> int:
        """Length of time series."""
        ...
    
    @property
    def values(self) -> List[float]:
        """Time series values."""
        ...
    
    def is_empty(self) -> bool:
        """Check if empty."""
        ...
    
    def mean(self) -> Optional[float]:
        """Mean of values."""
        ...
    
    def std(self, _ddof: float = 1.0) -> Optional[float]:
        """Standard deviation."""
        ...
    
    def var(self, _ddof: float = 1.0) -> Optional[float]:
        """Variance."""
        ...
    
    def __repr__(self) -> str:
        ...


class ARIMA:
    """ARIMA (AutoRegressive Integrated Moving Average) model."""
    
    def __init__(self, p: int, d: int, q: int) -> None:
        """Create ARIMA(p, d, q) model.
        
        Args:
            p: AR order
            d: Integration order  
            q: MA order
        """
        ...
    
    def with_constant(self, include: bool) -> 'ARIMA':
        """Include constant term."""
        ...
    
    def seasonal(self, _p: int, _d: int, _q: int, _s: int) -> 'ARIMA':
        """Set seasonal orders."""
        ...
    
    def method(self, method: str) -> 'ARIMA':
        """Set estimation method."""
        ...
    
    def max_iter(self, max_iter: int) -> 'ARIMA':
        """Set maximum iterations."""
        ...
    
    def tol(self, tol: float) -> 'ARIMA':
        """Set convergence tolerance."""
        ...
    
    def fit(self, data: Union['TimeSeries', List[float], Any]) -> 'ARIMAResults':
        """Fit model to time series data.
        
        Args:
            data: TimeSeries object, list of floats, or convertible object
        """
        ...


class ARIMAResults:
    """Results from ARIMA fitting."""
    
    @property
    def ar_coef(self) -> List[float]:
        """AR coefficients."""
        ...
    
    @property
    def ma_coef(self) -> List[float]:
        """MA coefficients."""
        ...
    
    @property
    def constant(self) -> Optional[float]:
        """Constant term."""
        ...
    
    @property
    def sigma2(self) -> float:
        """Innovation variance."""
        ...
    
    @property
    def log_likelihood(self) -> float:
        """Log-likelihood."""
        ...
    
    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        ...
    
    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        ...
    
    @property
    def n_obs(self) -> int:
        """Number of observations."""
        ...
    
    @property
    def fitted(self) -> List[float]:
        """Fitted values."""
        ...
    
    @property
    def residuals(self) -> List[float]:
        """Residuals."""
        ...
    
    def forecast(self, steps: int) -> List[float]:
        """Forecast future values.
        
        Args:
            steps: Number of steps to forecast
        """
        ...
    
    def summary(self) -> str:
        """Get summary statistics."""
        ...


class GARCH:
    """GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model."""
    
    def __init__(self, p: int, q: int) -> None:
        """Create GARCH(p, q) model.
        
        Args:
            p: GARCH order (autoregressive conditional heteroskedasticity)
            q: ARCH order (autoregressive conditional heteroskedasticity)
        """
        ...
    
    def arch(self, q: int) -> 'GARCH':
        """Set ARCH order (alias for constructor parameter)."""
        ...
    
    def distribution(self, distribution: str) -> 'GARCH':
        """Set error distribution."""
        ...
    
    def fit(self, data: Union['TimeSeries', List[float], Any]) -> 'GARCHResults':
        """Fit model to residuals or time series.
        
        Args:
            data: TimeSeries object, list of floats, or convertible object
        """
        ...


class GARCHResults:
    """Results from GARCH fitting."""
    
    @property
    def mu(self) -> float:
        """Mean parameter."""
        ...
    
    @property
    def omega(self) -> float:
        """Constant in conditional variance equation."""
        ...
    
    @property
    def arch_coef(self) -> List[float]:
        """ARCH coefficients."""
        ...
    
    @property
    def garch_coef(self) -> List[float]:
        """GARCH coefficients."""
        ...
    
    @property
    def df(self) -> Optional[float]:
        """Degrees of freedom for t-distribution."""
        ...
    
    @property
    def conditional_variances(self) -> List[float]:
        """Conditional variances."""
        ...
    
    @property
    def residuals(self) -> List[float]:
        """Raw residuals."""
        ...
    
    @property
    def standardized_residuals(self) -> List[float]:
        """Standardized residuals."""
        ...
    
    @property
    def log_likelihood(self) -> float:
        """Log-likelihood."""
        ...
    
    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        ...
    
    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        ...
    
    @property
    def n_obs(self) -> int:
        """Number of observations."""
        ...
    
    def summary(self) -> str:
        """Get summary statistics."""
        ...


# ============================================================================
# Top-level Functions
# ============================================================================

def mean(data: List[float]) -> float:
    """Compute mean of a list of numbers."""
    ...


def std_dev(data: List[float]) -> float:
    """Compute standard deviation of a list of numbers."""
    ...


def correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation between two lists.
    
    Args:
        x: First variable
        y: Second variable
        
    Returns:
        Correlation coefficient between -1 and 1
    """
    ...


def descriptive_summary(data: List[float]) -> Dict[str, float]:
    """Compute comprehensive descriptive statistics.
    
    Returns:
        Dictionary with keys: mean, std, min, q25, median, q75, max
    """
    ...


def train_test_split(data: List[float], test_size: float = 0.3) -> Tuple[List[float], List[float]]:
    """Split data into train and test sets.
    
    Args:
        data: List of values to split
        test_size: Proportion of data for test set (default: 0.3)
        
    Returns:
        Tuple of (train, test) lists
    """
    ...


def version() -> str:
    """Get library version."""
    ...


# ============================================================================
# Submodules
# ============================================================================

class _StatsModule:
    """Statistical functions submodule."""
    
    def mean(self, data: List[float]) -> float: ...
    def std_dev(self, data: List[float]) -> float: ...
    def correlation(self, x: List[float], y: List[float]) -> float: ...
    def descriptive_summary(self, data: List[float]) -> Dict[str, float]: ...


class _ModelsModule:
    """Statistical models submodule."""
    
    Family = Family
    Link = Link
    GLMBuilder = GLMBuilder
    GLM = GLM
    GLMResults = GLMResults


class _TSAModule:
    """Time series analysis submodule."""
    
    TimeSeries = TimeSeries
    ARIMA = ARIMA
    ARIMAResults = ARIMAResults
    GARCH = GARCH
    GARCHResults = GARCHResults


class _UtilsModule:
    """Utility functions submodule."""
    
    def train_test_split(self, data: List[float], test_size: float = 0.3) -> Tuple[List[float], List[float]]: ...


# Module exports
stats: _StatsModule
models: _ModelsModule
tsa: _TSAModule
utils: _UtilsModule

__all__: List[str]