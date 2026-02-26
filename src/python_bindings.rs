//! Python bindings for StatOxide
//!
//! This module provides Python interfaces to StatOxide's statistical
//! modeling functionality, with an API inspired by R's formula interface.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, IntoPyArray};
use ndarray::{Array1, Array2, Ix1, Ix2};

use crate::data::{DataFrame, Series};
use crate::formula::Formula;
use crate::linear::{LinearModelBuilder, LinearRegressionResults, OLS, Ridge};

// ============================================================================
// Python Series wrapper
// ============================================================================

#[pyclass(name = "Series")]
pub struct PySeries {
    inner: Series,
}

#[pymethods]
impl PySeries {
    #[new]
    fn new(name: &str, data: Vec<f64>) -> Self {
        PySeries {
            inner: Series::new(name, Array1::from_vec(data)),
        }
    }

    /// Get series name
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get length
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Convert to numpy array
    fn to_numpy<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        let data = self.inner.data().to_vec();
        data.into_pyarray(py)
    }

    /// Convert to Python list
    fn to_list(&self) -> Vec<f64> {
        self.inner.data().to_vec()
    }

    /// Compute mean
    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }

    /// Compute standard deviation
    fn std(&self, ddof: f64) -> Option<f64> {
        self.inner.std(ddof)
    }

    /// Get minimum value
    fn min(&self) -> Option<f64> {
        self.inner.min()
    }

    /// Get maximum value
    fn max(&self) -> Option<f64> {
        self.inner.max()
    }

    /// Get quantile (R's type 7)
    fn quantile(&self, q: f64) -> Option<f64> {
        self.inner.quantile(q)
    }

    /// Standardize (z-score normalization)
    fn standardize(&self) -> PyResult<PySeries> {
        let standardized = self.inner.standardize()
            .ok_or_else(|| PyValueError::new_err("Cannot standardize series"))?;
        Ok(PySeries { inner: standardized })
    }

    fn __repr__(&self) -> String {
        format!("Series(name='{}', len={})", self.name(), self.__len__())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python DataFrame wrapper
// ============================================================================

#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    inner: DataFrame,
}

#[pymethods]
impl PyDataFrame {
    #[new]
    fn new(columns: Option<&PyDict>) -> PyResult<Self> {
        if let Some(cols) = columns {
            let mut inner_columns = std::collections::HashMap::new();
            
            for (key, value) in cols.iter() {
                let col_name = key.extract::<String>()?;
                
                if let Ok(py_series) = value.extract::<PySeries>() {
                    inner_columns.insert(col_name, py_series.inner);
                } else if let Ok(list) = value.extract::<Vec<f64>>() {
                    inner_columns.insert(
                        col_name.clone(),
                        Series::new(col_name, Array1::from_vec(list)),
                    );
                } else {
                    return Err(PyValueError::new_err(
                        format!("Column '{}' must be a Series or list of floats", col_name)
                    ));
                }
            }
            
            let inner = DataFrame::from_series(inner_columns)
                .map_err(|e| PyValueError::new_err(e))?;
            
            Ok(PyDataFrame { inner })
        } else {
            Ok(PyDataFrame { inner: DataFrame::new() })
        }
    }

    /// Create DataFrame from dictionary of lists
    #[staticmethod]
    fn from_dict(py: Python, data: &PyDict) -> PyResult<Self> {
        let mut columns = std::collections::HashMap::new();
        
        for (key, value) in data.iter() {
            let col_name = key.extract::<String>()?;
            
            if let Ok(list) = value.extract::<Vec<f64>>() {
                columns.insert(
                    col_name.clone(),
                    Series::new(col_name, Array1::from_vec(list)),
                );
            } else {
                return Err(PyValueError::new_err(
                    format!("Column '{}' must be a list of floats", col_name)
                ));
            }
        }
        
        let inner = DataFrame::from_series(columns)
            .map_err(|e| PyValueError::new_err(e))?;
        
        Ok(PyDataFrame { inner })
    }

    /// Get number of rows
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.n_rows(), self.inner.n_cols())
    }

    /// Get column names
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.column_names()
    }

    /// Get a column as Series
    fn __getitem__(&self, name: &str) -> PyResult<PySeries> {
        let series = self.inner.column(name)
            .ok_or_else(|| PyValueError::new_err(format!("Column '{}' not found", name)))?;
        
        Ok(PySeries { inner: series.clone() })
    }

    /// Add a column
    fn add_column(&mut self, series: PySeries) -> PyResult<()> {
        let df = std::mem::replace(&mut self.inner, DataFrame::new());
        self.inner = df.with_column(series.inner)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(())
    }

    /// Get head (first n rows) as dictionary
    fn head(&self, n: usize) -> PyResult<Py<PyDict>> {
        let n_rows = self.inner.n_rows();
        let n = n.min(n_rows);
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            for col_name in self.inner.column_names() {
                if let Some(series) = self.inner.column(&col_name) {
                    let data: Vec<f64> = series.data().iter().take(n).cloned().collect();
                    dict.set_item(col_name, data)?;
                }
            }
            
            Ok(dict.into())
        })
    }

    fn __repr__(&self) -> String {
        let (rows, cols) = self.shape();
        format!("DataFrame(rows={}, cols={})", rows, cols)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python Formula wrapper
// ============================================================================

#[pyclass(name = "Formula")]
pub struct PyFormula {
    inner: Formula,
}

#[pymethods]
impl PyFormula {
    #[new]
    fn new(formula: &str) -> PyResult<Self> {
        let inner = Formula::parse(formula)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(PyFormula { inner })
    }

    /// Create formula without intercept
    fn no_intercept(&mut self) {
        self.inner = self.inner.clone().no_intercept();
    }

    /// Get variables in formula
    fn variables(&self) -> Vec<String> {
        self.inner.variables().into_iter().collect()
    }

    fn __repr__(&self) -> String {
        format!("Formula({:?})", self.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python LinearRegressionResults wrapper
// ============================================================================

#[pyclass(name = "LinearRegressionResults")]
pub struct PyLinearRegressionResults {
    inner: LinearRegressionResults,
    feature_names: Vec<String>,
}

#[pymethods]
impl PyLinearRegressionResults {
    /// Get coefficients
    #[getter]
    fn params(&self) -> Vec<f64> {
        self.inner.coefficients.to_vec()
    }

    /// Get standard errors
    #[getter]
    fn bse(&self) -> Option<Vec<f64>> {
        self.inner.std_errors.as_ref().map(|se| se.to_vec())
    }

    /// Get t-values
    #[getter]
    fn tvalues(&self) -> Option<Vec<f64>> {
        self.inner.t_values.as_ref().map(|t| t.to_vec())
    }

    /// Get p-values
    #[getter]
    fn pvalues(&self) -> Option<Vec<f64>> {
        self.inner.p_values.as_ref().map(|p| p.to_vec())
    }

    /// Get R-squared
    #[getter]
    fn rsquared(&self) -> f64 {
        self.inner.r_squared
    }

    /// Get adjusted R-squared
    #[getter]
    fn rsquared_adj(&self) -> f64 {
        self.inner.r_squared_adj
    }

    /// Get AIC
    #[getter]
    fn aic(&self) -> Option<f64> {
        self.inner.aic
    }

    /// Get BIC
    #[getter]
    fn bic(&self) -> Option<f64> {
        self.inner.bic
    }

    /// Get residual standard error
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.sigma
    }

    /// Get degrees of freedom
    #[getter]
    fn df_resid(&self) -> usize {
        self.inner.df_residual
    }

    /// Predict using fitted model
    fn predict(&self, X: &PyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let X_arr = X.readonly();
        let X_view = X_arr.as_array();
        
        let predictions = self.inner.predict(X_view);
        
        Python::with_gil(|py| {
            Ok(predictions.into_pyarray(py).into())
        })
    }

    /// Get summary as string
    fn summary(&self) -> String {
        self.inner.summary(&self.feature_names)
    }

    fn __repr__(&self) -> String {
        format!("LinearRegressionResults(rsquared={:.3})", self.rsquared())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python API Functions
// ============================================================================

/// Fit OLS regression using formula
///
/// Args:
///     formula: Formula string (e.g., "y ~ x1 + x2")
///     data: DataFrame containing the variables
///     intercept: Whether to include intercept (default: True)
///
/// Returns:
///     LinearRegressionResults object
#[pyfunction]
fn ols(formula: &str, data: &PyDataFrame, intercept: bool) -> PyResult<PyLinearRegressionResults> {
    let formula_obj = Formula::parse(formula)
        .map_err(|e| PyValueError::new_err(e))?;
    
    let mut formula_obj = if !intercept {
        formula_obj.no_intercept()
    } else {
        formula_obj
    };
    
    let results = LinearModelBuilder::formula(&formula_obj, &data.inner)
        .ols()
        .map_err(|e| PyValueError::new_err(e))?;
    
    // Extract feature names from formula
    let mut feature_names = Vec::new();
    if intercept {
        feature_names.push("(Intercept)".to_string());
    }
    
    for term in &formula_obj.predictors {
        // Simplified: just use variable names
        // In production, we'd need to properly extract names from terms
        if let crate::formula::Term::Variable(name) = term {
            feature_names.push(name.clone());
        }
    }
    
    Ok(PyLinearRegressionResults {
        inner: results,
        feature_names,
    })
}

/// Fit Ridge regression using formula
///
/// Args:
///     formula: Formula string (e.g., "y ~ x1 + x2")
///     data: DataFrame containing the variables
///     alpha: Regularization strength
///     intercept: Whether to include intercept (default: True)
///     standardize: Whether to standardize features (default: True)
///
/// Returns:
///     LinearRegressionResults object
#[pyfunction]
fn ridge(
    formula: &str,
    data: &PyDataFrame,
    alpha: f64,
    intercept: bool,
    standardize: bool,
) -> PyResult<PyLinearRegressionResults> {
    let formula_obj = Formula::parse(formula)
        .map_err(|e| PyValueError::new_err(e))?;
    
    let mut formula_obj = if !intercept {
        formula_obj.no_intercept()
    } else {
        formula_obj
    };
    
    let mut builder = LinearModelBuilder::formula(&formula_obj, &data.inner);
    if !standardize {
        builder = builder.no_standardize();
    }
    
    let results = builder.ridge(alpha)
        .map_err(|e| PyValueError::new_err(e))?;
    
    // Extract feature names (simplified)
    let mut feature_names = Vec::new();
    if intercept {
        feature_names.push("(Intercept)".to_string());
    }
    
    for term in &formula_obj.predictors {
        if let crate::formula::Term::Variable(name) = term {
            feature_names.push(name.clone());
        }
    }
    
    Ok(PyLinearRegressionResults {
        inner: results,
        feature_names,
    })
}

// ============================================================================
// Python module definition
// ============================================================================

/// StatOxide: High-performance statistical modeling for Python
/// 
/// This module provides a Python interface to StatOxide's statistical
/// modeling functionality, with an API inspired by R's formula interface.
pub fn statoxide(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PyFormula>()?;
    m.add_class::<PyLinearRegressionResults>()?;

    // Add regression functions
    m.add_function(wrap_pyfunction!(ols, m)?)?;
    m.add_function(wrap_pyfunction!(ridge, m)?)?;

    // Add version constant
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}