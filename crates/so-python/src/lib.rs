use ndarray::Array1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use so_core::data::{DataFrame, Series};
use so_core::formula::Formula;

// Import models
use so_models::glm::{Family, GLM as RustGLM, GLMModelBuilder, GLMResults, Link};
use so_models::regression::OLS;

// Import time series
use so_tsa::TimeSeries;
use so_tsa::arima::ARIMAResults;
use so_tsa::garch::GARCHResults;

// Import statistical tests
use so_stats::tests::{
    Alternative, TestResult, anova_one_way as anova_one_way_rs,
    chi_square_test_independence as chi_square_test_independence_rs,
    shapiro_wilk_test as shapiro_wilk_test_rs, t_test_one_sample as t_test_one_sample_rs,
    t_test_paired as t_test_paired_rs, t_test_two_sample as t_test_two_sample_rs,
};

/// Python wrapper for StatOxide Series
#[pyclass(name = "Series")]
struct PySeries {
    inner: Series,
}

#[pymethods]
impl PySeries {
    /// Create a new Series from Python list or array
    #[new]
    fn new(name: String, data: Vec<f64>) -> PyResult<Self> {
        let array = Array1::from_vec(data);
        Ok(PySeries {
            inner: Series::new(name, array),
        })
    }

    /// Get the name of the series
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get the length of the series
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if series is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Compute mean of series
    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }

    /// Compute standard deviation
    fn std(&self, ddof: f64) -> Option<f64> {
        self.inner.std(ddof)
    }

    /// Compute variance
    fn var(&self, ddof: f64) -> Option<f64> {
        self.inner.var(ddof)
    }

    /// Get minimum value
    fn min(&self) -> Option<f64> {
        self.inner.min()
    }

    /// Get maximum value
    fn max(&self) -> Option<f64> {
        self.inner.max()
    }

    /// Compute quantile
    fn quantile(&self, q: f64) -> Option<f64> {
        self.inner.quantile(q)
    }

    /// Convert to Python list
    fn to_list(&self) -> Vec<f64> {
        self.inner.data().to_vec()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Series(name='{}', len={})",
            self.inner.name(),
            self.inner.len()
        )
    }
}

/// Python wrapper for StatOxide DataFrame
#[pyclass(name = "DataFrame")]
struct PyDataFrame {
    inner: DataFrame,
}

#[pymethods]
impl PyDataFrame {
    /// Create a new DataFrame from a dictionary of columns
    #[new]
    fn new(data: HashMap<String, Vec<f64>>) -> PyResult<Self> {
        let mut columns = HashMap::new();

        for (name, values) in data {
            let array = Array1::from_vec(values);
            columns.insert(name.clone(), Series::new(name, array));
        }

        match DataFrame::from_series(columns) {
            Ok(df) => Ok(PyDataFrame { inner: df }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Error creating DataFrame: {:?}",
                e
            ))),
        }
    }

    /// Get number of rows
    #[getter]
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    /// Get number of columns
    #[getter]
    fn n_cols(&self) -> usize {
        self.inner.n_cols()
    }

    /// Get column names
    fn columns(&self) -> Vec<String> {
        self.inner.column_names()
    }

    /// Get a column by name
    fn get_column(&self, name: &str) -> PyResult<PySeries> {
        match self.inner.column(name) {
            Some(series) => Ok(PySeries {
                inner: series.clone(),
            }),
            None => Err(PyValueError::new_err(format!(
                "Column '{}' not found",
                name
            ))),
        }
    }

    /// Add a column to the DataFrame
    fn with_column(&mut self, series: &PySeries) -> PyResult<()> {
        let df = self
            .inner
            .clone()
            .with_column(series.inner.clone())
            .map_err(|e| PyValueError::new_err(format!("Error adding column: {:?}", e)))?;
        self.inner = df;
        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "DataFrame(rows={}, cols={})",
            self.inner.n_rows(),
            self.inner.n_cols()
        )
    }
}

/// Python wrapper for StatOxide Formula
#[pyclass(name = "Formula")]
struct PyFormula {
    inner: Formula,
}

#[pymethods]
impl PyFormula {
    /// Parse a formula string
    #[new]
    fn new(formula: String) -> PyResult<Self> {
        match Formula::parse(&formula) {
            Ok(f) => Ok(PyFormula { inner: f }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Error parsing formula: {:?}",
                e
            ))),
        }
    }

    /// Get all variable names in the formula
    fn variables(&self) -> Vec<String> {
        self.inner.variables().into_iter().collect()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Formula({:?})", self.inner)
    }
}

/// Python wrapper for GLM Family
#[pyclass(name = "Family")]
#[derive(Clone)]
struct PyFamily {
    inner: Family,
}

#[pymethods]
impl PyFamily {
    /// Create Gaussian family
    #[staticmethod]
    fn gaussian() -> Self {
        PyFamily {
            inner: Family::Gaussian,
        }
    }

    /// Create Binomial family
    #[staticmethod]
    fn binomial() -> Self {
        PyFamily {
            inner: Family::Binomial,
        }
    }

    /// Create Poisson family
    #[staticmethod]
    fn poisson() -> Self {
        PyFamily {
            inner: Family::Poisson,
        }
    }

    /// Create Gamma family
    #[staticmethod]
    fn gamma() -> Self {
        PyFamily {
            inner: Family::Gamma,
        }
    }

    /// Create Inverse Gaussian family
    #[staticmethod]
    fn inverse_gaussian() -> Self {
        PyFamily {
            inner: Family::InverseGaussian,
        }
    }

    /// Get family name
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Family({})", self.name())
    }
}

/// Python wrapper for GLM Link function
#[pyclass(name = "Link")]
#[derive(Clone)]
struct PyLink {
    inner: Link,
}

#[pymethods]
impl PyLink {
    /// Identity link: η = μ
    #[staticmethod]
    fn identity() -> Self {
        PyLink {
            inner: Link::Identity,
        }
    }

    /// Logit link: η = log(μ/(1-μ))
    #[staticmethod]
    fn logit() -> Self {
        PyLink { inner: Link::Logit }
    }

    /// Probit link: η = Φ⁻¹(μ)
    #[staticmethod]
    fn probit() -> Self {
        PyLink {
            inner: Link::Probit,
        }
    }

    /// Log link: η = log(μ)
    #[staticmethod]
    fn log() -> Self {
        PyLink { inner: Link::Log }
    }

    /// Inverse link: η = 1/μ
    #[staticmethod]
    fn inverse() -> Self {
        PyLink {
            inner: Link::Inverse,
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        match self.inner {
            Link::Identity => "Link(identity)".to_string(),
            Link::Logit => "Link(logit)".to_string(),
            Link::Probit => "Link(probit)".to_string(),
            Link::Cloglog => "Link(cloglog)".to_string(),
            Link::Log => "Link(log)".to_string(),
            Link::Inverse => "Link(inverse)".to_string(),
            Link::InverseSquare => "Link(inverse-square)".to_string(),
            Link::Sqrt => "Link(sqrt)".to_string(),
        }
    }
}

/// Python wrapper for GLM model builder
#[pyclass(name = "GLMBuilder")]
struct PyGLMBuilder {
    inner: Option<GLMModelBuilder>,
}

#[pymethods]
impl PyGLMBuilder {
    /// Create a new GLM builder
    #[new]
    fn new() -> Self {
        PyGLMBuilder {
            inner: Some(GLMModelBuilder::new()),
        }
    }

    /// Set the distribution family
    fn family(&mut self, family: &PyFamily) -> PyResult<()> {
        if let Some(inner) = self.inner.take() {
            self.inner = Some(inner.family(family.inner));
        }
        Ok(())
    }

    /// Set the link function
    fn link(&mut self, link: &PyLink) -> PyResult<()> {
        if let Some(inner) = self.inner.take() {
            self.inner = Some(inner.link(link.inner));
        }
        Ok(())
    }

    /// Set whether to include intercept
    fn intercept(&mut self, intercept: bool) -> PyResult<()> {
        if let Some(inner) = self.inner.take() {
            self.inner = Some(inner.intercept(intercept));
        }
        Ok(())
    }

    /// Set maximum iterations
    fn max_iter(&mut self, max_iter: usize) -> PyResult<()> {
        if let Some(inner) = self.inner.take() {
            self.inner = Some(inner.max_iter(max_iter));
        }
        Ok(())
    }

    /// Set convergence tolerance
    fn tol(&mut self, tol: f64) -> PyResult<()> {
        if let Some(inner) = self.inner.take() {
            self.inner = Some(inner.tol(tol));
        }
        Ok(())
    }

    /// Set fixed scale parameter
    fn scale(&mut self, scale: f64) -> PyResult<()> {
        if let Some(inner) = self.inner.take() {
            self.inner = Some(inner.scale(scale));
        }
        Ok(())
    }

    /// Build the GLM model
    fn build(&mut self) -> PyResult<PyGLM> {
        if let Some(inner) = self.inner.take() {
            Ok(PyGLM {
                inner: inner.build(),
            })
        } else {
            Err(PyRuntimeError::new_err("GLM builder not available"))
        }
    }
}

/// Python wrapper for GLM model
#[pyclass(name = "GLM")]
struct PyGLM {
    inner: RustGLM,
}

#[pymethods]
impl PyGLM {
    /// Create a new GLM builder
    #[staticmethod]
    fn new() -> PyGLMBuilder {
        PyGLMBuilder::new()
    }

    /// Fit the GLM using formula and DataFrame
    fn fit(&self, formula: &str, data: &PyDataFrame) -> PyResult<PyGLMResults> {
        match self.inner.fit(formula, &data.inner) {
            Ok(results) => Ok(PyGLMResults { inner: results }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "GLM fitting failed: {:?}",
                e
            ))),
        }
    }

    /// Fit the GLM with design matrix X and response y
    fn fit_matrix(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<PyGLMResults> {
        // Convert to DataFrame
        let n_rows = x.len();
        if n_rows == 0 {
            return Err(PyValueError::new_err("X must have at least one row"));
        }
        if n_rows != y.len() {
            return Err(PyValueError::new_err(
                "X and y must have same number of rows",
            ));
        }

        let n_cols = x[0].len();

        // Check all rows have same number of columns
        for (i, row) in x.iter().enumerate() {
            if row.len() != n_cols {
                return Err(PyValueError::new_err(format!(
                    "Row {} has {} columns, expected {}",
                    i,
                    row.len(),
                    n_cols
                )));
            }
        }

        // Create column names
        let mut col_names = Vec::new();
        for i in 0..n_cols {
            col_names.push(format!("x{}", i));
        }
        col_names.push("y".to_string());

        // Create Series for each column
        let mut columns = HashMap::new();

        // Create X columns
        for i in 0..n_cols {
            let mut col_data = Vec::with_capacity(n_rows);
            for row in &x {
                col_data.push(row[i]);
            }
            let series = Series::new(format!("x{}", i), ndarray::Array1::from_vec(col_data));
            columns.insert(format!("x{}", i), series);
        }

        // Create y column
        let y_series = Series::new("y".to_string(), ndarray::Array1::from_vec(y.clone()));
        columns.insert("y".to_string(), y_series);

        // Create DataFrame from Series
        let df = DataFrame::from_series(columns)
            .map_err(|e| PyValueError::new_err(format!("Failed to create DataFrame: {}", e)))?;

        // Create formula: y ~ x0 + x1 + ... + x{n-1}
        let formula_str = if n_cols == 0 {
            "y ~ 1".to_string()
        } else {
            let mut formula = "y ~ ".to_string();
            for i in 0..n_cols {
                formula.push_str(&format!("x{}", i));
                if i < n_cols - 1 {
                    formula.push_str(" + ");
                }
            }
            formula
        };

        // Use the GLM's fit method (takes formula string and DataFrame)
        match self.inner.fit(&formula_str, &df) {
            Ok(results) => Ok(PyGLMResults { inner: results }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "GLM fitting failed: {:?}",
                e
            ))),
        }
    }
}

/// Python wrapper for GLM results
#[pyclass(name = "GLMResults")]
struct PyGLMResults {
    inner: GLMResults,
}

#[pymethods]
impl PyGLMResults {
    /// Get coefficients
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients.to_vec()
    }

    /// Get standard errors
    #[getter]
    fn std_errors(&self) -> Vec<f64> {
        self.inner.std_errors.to_vec()
    }

    /// Get z-values (Wald test statistics)
    #[getter]
    fn z_values(&self) -> Vec<f64> {
        self.inner.z_values.to_vec()
    }

    /// Get p-values
    #[getter]
    fn p_values(&self) -> Vec<f64> {
        self.inner.p_values.to_vec()
    }

    /// Get deviance
    #[getter]
    fn deviance(&self) -> f64 {
        self.inner.deviance
    }

    /// Get null deviance
    #[getter]
    fn null_deviance(&self) -> f64 {
        self.inner.null_deviance
    }

    /// Get AIC
    #[getter]
    fn aic(&self) -> f64 {
        self.inner.aic
    }

    /// Get BIC
    #[getter]
    fn bic(&self) -> f64 {
        self.inner.bic
    }

    /// Get degrees of freedom
    #[getter]
    fn df_residual(&self) -> usize {
        self.inner.df_residual
    }

    /// Get degrees of freedom for null model
    #[getter]
    fn df_null(&self) -> usize {
        self.inner.df_null
    }

    /// Get scale parameter
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Get fitted values
    #[getter]
    fn fitted_values(&self) -> Vec<f64> {
        self.inner.fitted_values.to_vec()
    }

    /// Get Pearson residuals
    #[getter]
    fn pearson_residuals(&self) -> Vec<f64> {
        self.inner.pearson_residuals.to_vec()
    }

    /// Get raw residuals (response scale)
    #[getter]
    fn residuals(&self) -> Vec<f64> {
        self.inner.residuals.to_vec()
    }

    /// Get diagonal of hat matrix (leverage values)
    #[getter]
    fn hat_matrix_diag(&self) -> Vec<f64> {
        self.inner.hat_matrix_diag.to_vec()
    }

    /// Get number of iterations
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Check if model converged
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Predict using the fitted model
    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let n_rows = x.len();
        if n_rows == 0 {
            return Ok(Vec::new());
        }
        let n_cols = x[0].len();

        // Check dimensions
        if n_cols != self.inner.coefficients.len() {
            return Err(PyValueError::new_err(format!(
                "X has {} columns but model has {} coefficients",
                n_cols,
                self.inner.coefficients.len()
            )));
        }

        // Simple linear prediction: y = Xβ
        let mut predictions = Vec::with_capacity(n_rows);
        for row in x {
            if row.len() != n_cols {
                return Err(PyValueError::new_err(
                    "All rows must have same number of columns",
                ));
            }
            let mut pred = 0.0;
            for (i, &xi) in row.iter().enumerate() {
                pred += xi * self.inner.coefficients[i];
            }
            // TODO: Apply inverse link function based on family and link
            predictions.push(pred);
        }

        Ok(predictions)
    }

    /// Get summary string
    fn summary(&self) -> String {
        // Simple summary for now
        format!(
            "GLM Results:\n  Coefficients: {:?}\n  AIC: {:.2}\n  BIC: {:.2}\n  Deviance: {:.2}\n  Scale: {:.2}",
            self.coefficients(),
            self.aic(),
            self.bic(),
            self.deviance(),
            self.scale()
        )
    }
}

/// Python wrapper for TimeSeries
#[pyclass(name = "TimeSeries")]
struct PyTimeSeries {
    inner: TimeSeries,
}

#[pymethods]
impl PyTimeSeries {
    /// Create a TimeSeries from a DataFrame
    #[staticmethod]
    fn from_dataframe(df: &PyDataFrame, value_col: &str, date_col: &str) -> PyResult<Self> {
        match TimeSeries::from_dataframe(&df.inner, value_col, date_col) {
            Ok(ts) => Ok(PyTimeSeries { inner: ts }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to create TimeSeries: {:?}",
                e
            ))),
        }
    }

    /// Create a TimeSeries from vectors
    #[staticmethod]
    fn from_vectors(values: Vec<f64>, _dates: Vec<String>) -> PyResult<Self> {
        // Simple implementation - use index as timestamps
        // In practice, would parse dates string to timestamps
        let timestamps: Vec<i64> = (0..values.len() as i64).collect();
        let values_array = ndarray::Array1::from_vec(values);

        match TimeSeries::new("series", timestamps, values_array, None) {
            Ok(ts) => Ok(PyTimeSeries { inner: ts }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to create TimeSeries: {:?}",
                e
            ))),
        }
    }

    /// Get values
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    /// Get length
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Compute mean
    fn mean(&self) -> Option<f64> {
        Some(self.inner.stats().mean)
    }

    /// Compute standard deviation
    fn std(&self, _ddof: f64) -> Option<f64> {
        // Note: ddof is ignored for now, uses population std
        Some(self.inner.stats().std)
    }

    /// Compute variance
    fn var(&self, _ddof: f64) -> Option<f64> {
        // Note: ddof is ignored for now, uses population variance
        Some(self.inner.stats().variance)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("TimeSeries(len={})", self.len())
    }
}

/// Python wrapper for ARIMA builder
#[pyclass(name = "ARIMA")]
struct PyARIMA {
    builder: Option<so_tsa::arima::ARIMABuilder>,
}

#[pymethods]
impl PyARIMA {
    /// Create a new ARIMA model
    #[new]
    fn new(p: usize, d: usize, q: usize) -> Self {
        use so_tsa::arima::ARIMABuilder;
        PyARIMA {
            builder: Some(ARIMABuilder::new(p, d, q)),
        }
    }

    /// Set seasonal parameters
    fn seasonal(&mut self, _p: usize, _d: usize, _q: usize, _s: usize) -> PyResult<()> {
        // SARIMA not fully implemented in Python bindings yet
        // For now, just do nothing
        // TODO: Implement proper SARIMA support
        Ok(())
    }

    /// Include constant term
    fn with_constant(&mut self, include: bool) -> PyResult<()> {
        if let Some(builder) = self.builder.take() {
            self.builder = Some(builder.with_constant(include));
        }
        Ok(())
    }

    /// Set estimation method
    fn method(&mut self, method: String) -> PyResult<()> {
        use so_tsa::arima::EstimationMethod;
        if let Some(builder) = self.builder.take() {
            let est_method = match method.to_lowercase().as_str() {
                "css" => EstimationMethod::CSS,
                "ml" => EstimationMethod::ML,
                "exactml" => EstimationMethod::ExactML,
                _ => EstimationMethod::CSS,
            };
            self.builder = Some(builder.method(est_method));
        }
        Ok(())
    }

    /// Set maximum iterations
    fn max_iter(&mut self, max_iter: usize) -> PyResult<()> {
        if let Some(builder) = self.builder.take() {
            self.builder = Some(builder.max_iter(max_iter));
        }
        Ok(())
    }

    /// Set convergence tolerance
    fn tol(&mut self, tol: f64) -> PyResult<()> {
        if let Some(builder) = self.builder.take() {
            self.builder = Some(builder.tol(tol));
        }
        Ok(())
    }

    /// Fit the ARIMA model
    ///
    /// Accepts multiple input types:
    /// - PyTimeSeries object
    /// - List of floats (Vec<f64>)
    /// - Any object convertible to a list of floats
    fn fit(&mut self, py: Python, data: Py<PyAny>) -> PyResult<PyARIMAResults> {
        // Get reference to Python object
        let data_ref = data.bind(py);

        // Try to convert input to TimeSeries
        let timeseries = if let Ok(ts) = data_ref.extract::<PyRef<PyTimeSeries>>() {
            // Already a TimeSeries
            ts.inner.clone()
        } else if let Ok(vec) = data_ref.extract::<Vec<f64>>() {
            // Vector of floats - create TimeSeries with index as timestamps
            let timestamps: Vec<i64> = (0..vec.len() as i64).collect();
            let values_array = ndarray::Array1::from_vec(vec);
            TimeSeries::new("series", timestamps, values_array, None).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create TimeSeries: {:?}", e))
            })?
        } else if let Ok(list) = data_ref.cast::<PyList>() {
            // Python list - extract as floats
            let mut vec = Vec::with_capacity(list.len());
            for i in 0..list.len() {
                let item = list.get_item(i)?;
                let val: f64 = item
                    .extract()
                    .map_err(|_| PyValueError::new_err("List must contain only numeric values"))?;
                vec.push(val);
            }
            let timestamps: Vec<i64> = (0..vec.len() as i64).collect();
            let values_array = ndarray::Array1::from_vec(vec);
            TimeSeries::new("series", timestamps, values_array, None).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create TimeSeries: {:?}", e))
            })?
        } else {
            return Err(PyValueError::new_err(
                "Input must be a TimeSeries, list of floats, or convertible to list of floats",
            ));
        };

        if let Some(builder) = self.builder.take() {
            match builder.fit(&timeseries) {
                Ok(results) => Ok(PyARIMAResults { inner: results }),
                Err(e) => Err(PyRuntimeError::new_err(format!(
                    "ARIMA fitting failed: {:?}",
                    e
                ))),
            }
        } else {
            Err(PyRuntimeError::new_err("ARIMA builder not available"))
        }
    }
}

/// Python wrapper for ARIMA results
#[pyclass(name = "ARIMAResults")]
struct PyARIMAResults {
    inner: ARIMAResults,
}

#[pymethods]
impl PyARIMAResults {
    /// Get AR coefficients
    #[getter]
    fn ar_coef(&self) -> Option<Vec<f64>> {
        self.inner.ar_coef.as_ref().map(|coef| coef.to_vec())
    }

    /// Get MA coefficients
    #[getter]
    fn ma_coef(&self) -> Option<Vec<f64>> {
        self.inner.ma_coef.as_ref().map(|coef| coef.to_vec())
    }

    /// Get constant term
    #[getter]
    fn constant(&self) -> Option<f64> {
        self.inner.constant
    }

    /// Get AIC
    #[getter]
    fn aic(&self) -> f64 {
        self.inner.aic
    }

    /// Get BIC
    #[getter]
    fn bic(&self) -> f64 {
        self.inner.bic
    }

    /// Get log-likelihood
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood
    }

    /// Get sigma2 (innovation variance)
    #[getter]
    fn sigma2(&self) -> f64 {
        self.inner.sigma2
    }

    /// Get number of observations
    #[getter]
    fn n_obs(&self) -> usize {
        self.inner.n_obs
    }

    /// Forecast future values
    fn forecast(&self, steps: usize) -> Vec<f64> {
        // Simple forecast - in practice would use proper forecasting method
        // For now, return last value repeated
        let last_value = if self.inner.fitted.len() > 0 {
            self.inner.fitted[self.inner.fitted.len() - 1]
        } else {
            0.0
        };
        vec![last_value; steps]
    }

    /// Get fitted values
    #[getter]
    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted.to_vec()
    }

    /// Get residuals
    #[getter]
    fn residuals(&self) -> Vec<f64> {
        self.inner.residuals.to_vec()
    }

    /// Get summary string
    fn summary(&self) -> String {
        format!(
            "ARIMA Results:\n  AIC: {:.2}\n  BIC: {:.2}\n  Log-Likelihood: {:.2}\n  Sigma2: {:.4}",
            self.aic(),
            self.bic(),
            self.log_likelihood(),
            self.sigma2()
        )
    }
}

/// Python wrapper for GARCH model
#[pyclass(name = "GARCH")]
struct PyGARCH {
    builder: Option<so_tsa::garch::GARCHBuilder>,
}

#[pymethods]
impl PyGARCH {
    /// Create a new GARCH model
    #[new]
    fn new(p: usize, q: usize) -> Self {
        use so_tsa::garch::GARCHBuilder;
        PyGARCH {
            builder: Some(GARCHBuilder::new(p, q)),
        }
    }

    /// Create an ARCH model (GARCH with p=0)
    #[staticmethod]
    fn arch(q: usize) -> Self {
        use so_tsa::garch::GARCHBuilder;
        PyGARCH {
            builder: Some(GARCHBuilder::arch(q)),
        }
    }

    /// Set distribution for innovations
    fn distribution(&mut self, distribution: String) -> PyResult<()> {
        use so_tsa::garch::GARCHDistribution;
        if let Some(builder) = self.builder.take() {
            let dist = match distribution.to_lowercase().as_str() {
                "normal" => GARCHDistribution::Normal,
                "t" | "studentst" => GARCHDistribution::StudentsT(5.0), // Default df=5.0
                "ged" => GARCHDistribution::GED(1.5),                   // Default shape=1.5
                _ => GARCHDistribution::Normal,
            };
            self.builder = Some(builder.distribution(dist));
        }
        Ok(())
    }

    /// Fit the GARCH model to residuals
    ///
    /// Accepts multiple input types:
    /// - List of floats (Vec<f64>) - residuals
    /// - PyTimeSeries object
    /// - Any object convertible to a list of floats
    fn fit(&mut self, py: Python, data: Py<PyAny>) -> PyResult<PyGARCHResults> {
        // Get reference to Python object
        let data_ref = data.bind(py);

        // Try to convert input to TimeSeries
        let timeseries = if let Ok(ts) = data_ref.extract::<PyRef<PyTimeSeries>>() {
            // Already a TimeSeries
            ts.inner.clone()
        } else if let Ok(vec) = data_ref.extract::<Vec<f64>>() {
            // Vector of floats - create TimeSeries with index as timestamps
            let timestamps: Vec<i64> = (0..vec.len() as i64).collect();
            let values_array = ndarray::Array1::from_vec(vec);
            TimeSeries::new("residuals", timestamps, values_array, None).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create TimeSeries: {:?}", e))
            })?
        } else if let Ok(list) = data_ref.cast::<PyList>() {
            // Python list - extract as floats
            let mut vec = Vec::with_capacity(list.len());
            for i in 0..list.len() {
                let item = list.get_item(i)?;
                let val: f64 = item
                    .extract()
                    .map_err(|_| PyValueError::new_err("List must contain only numeric values"))?;
                vec.push(val);
            }
            let timestamps: Vec<i64> = (0..vec.len() as i64).collect();
            let values_array = ndarray::Array1::from_vec(vec);
            TimeSeries::new("residuals", timestamps, values_array, None).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create TimeSeries: {:?}", e))
            })?
        } else {
            return Err(PyValueError::new_err(
                "Input must be a TimeSeries, list of floats, or convertible to list of floats",
            ));
        };

        if let Some(builder) = self.builder.take() {
            match builder.fit(&timeseries) {
                Ok(results) => Ok(PyGARCHResults { inner: results }),
                Err(e) => Err(PyRuntimeError::new_err(format!(
                    "GARCH fitting failed: {:?}",
                    e
                ))),
            }
        } else {
            Err(PyRuntimeError::new_err("GARCH builder not available"))
        }
    }
}

/// Python wrapper for GARCH results
#[pyclass(name = "GARCHResults")]
struct PyGARCHResults {
    inner: GARCHResults,
}

#[pymethods]
impl PyGARCHResults {
    /// Get omega (constant in variance equation)
    #[getter]
    fn omega(&self) -> f64 {
        self.inner.omega
    }

    /// Get ARCH coefficients (α₁, ..., α_q)
    #[getter]
    fn arch_coef(&self) -> Vec<f64> {
        self.inner.arch_coef.to_vec()
    }

    /// Get GARCH coefficients (β₁, ..., β_p)
    #[getter]
    fn garch_coef(&self) -> Vec<f64> {
        self.inner.garch_coef.to_vec()
    }

    /// Get mu (constant in mean equation, if included)
    #[getter]
    fn mu(&self) -> Option<f64> {
        self.inner.mu
    }

    /// Get degrees of freedom (for t/GED distributions)
    #[getter]
    fn df(&self) -> Option<f64> {
        self.inner.df
    }

    /// Get AIC
    #[getter]
    fn aic(&self) -> f64 {
        self.inner.aic
    }

    /// Get BIC
    #[getter]
    fn bic(&self) -> f64 {
        self.inner.bic
    }

    /// Get log-likelihood
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood
    }

    /// Get number of observations
    #[getter]
    fn n_obs(&self) -> usize {
        self.inner.n_obs
    }

    /// Get residuals (εₜ)
    #[getter]
    fn residuals(&self) -> Vec<f64> {
        self.inner.residuals.to_vec()
    }

    /// Get conditional variances (σₜ²)
    #[getter]
    fn conditional_variances(&self) -> Vec<f64> {
        self.inner.conditional_variances.to_vec()
    }

    /// Get standardized residuals (zₜ = εₜ/σₜ)
    #[getter]
    fn standardized_residuals(&self) -> Vec<f64> {
        self.inner.standardized_residuals.to_vec()
    }

    /// Get summary string
    fn summary(&self) -> String {
        format!(
            "GARCH Results:\n  AIC: {:.2}\n  BIC: {:.2}\n  Log-Likelihood: {:.2}",
            self.aic(),
            self.bic(),
            self.log_likelihood()
        )
    }
}

/// StatOxide Python module
#[pymodule]
#[pyo3(name = "statoxide")]
fn statoxide(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register core classes
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PyFormula>()?;

    // Register GLM classes
    m.add_class::<PyFamily>()?;
    m.add_class::<PyLink>()?;
    m.add_class::<PyGLMBuilder>()?;
    m.add_class::<PyGLM>()?;
    m.add_class::<PyGLMResults>()?;

    // Register TSA classes
    m.add_class::<PyTimeSeries>()?;
    m.add_class::<PyARIMA>()?;
    m.add_class::<PyARIMAResults>()?;
    m.add_class::<PyGARCH>()?;
    m.add_class::<PyGARCHResults>()?;

    // Basic functions module
    let stats_module = PyModule::new(m.py(), "stats")?;
    stats_module.add_function(wrap_pyfunction!(mean, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(std_dev, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(correlation, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(descriptive_summary, &stats_module)?)?;
    // Statistical tests
    stats_module.add_function(wrap_pyfunction!(t_test_one_sample, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(t_test_two_sample, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(t_test_paired, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(
        chi_square_test_independence,
        &stats_module
    )?)?;
    stats_module.add_function(wrap_pyfunction!(anova_one_way, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(shapiro_wilk_test, &stats_module)?)?;
    m.add_submodule(&stats_module)?;

    // Models module
    let models_module = PyModule::new(m.py(), "models")?;
    models_module.add_function(wrap_pyfunction!(linear_regression, &models_module)?)?;
    models_module.add_function(wrap_pyfunction!(mixed_effects, &models_module)?)?;
    models_module.add_class::<PyFamily>()?;
    models_module.add_class::<PyLink>()?;
    models_module.add_class::<PyGLMBuilder>()?;
    models_module.add_class::<PyGLM>()?;
    models_module.add_class::<PyGLMResults>()?;
    m.add_submodule(&models_module)?;

    // TSA module
    let tsa_module = PyModule::new(m.py(), "tsa")?;
    tsa_module.add_function(wrap_pyfunction!(fit_arima, &tsa_module)?)?;
    tsa_module.add_class::<PyTimeSeries>()?;
    tsa_module.add_class::<PyARIMA>()?;
    tsa_module.add_class::<PyARIMAResults>()?;
    tsa_module.add_class::<PyGARCH>()?;
    tsa_module.add_class::<PyGARCHResults>()?;
    m.add_submodule(&tsa_module)?;

    // Utilities module
    let utils_module = PyModule::new(m.py(), "utils")?;
    utils_module.add_function(wrap_pyfunction!(train_test_split, &utils_module)?)?;
    m.add_submodule(&utils_module)?;

    // Top-level functions
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // Add commonly used functions to top level for convenience
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(correlation, m)?)?;
    m.add_function(wrap_pyfunction!(descriptive_summary, m)?)?;
    m.add_function(wrap_pyfunction!(train_test_split, m)?)?;

    Ok(())
}

/// Compute mean of data
#[pyfunction]
fn mean(data: Vec<f64>) -> PyResult<f64> {
    if data.is_empty() {
        return Ok(f64::NAN);
    }
    Ok(data.iter().sum::<f64>() / data.len() as f64)
}

/// Compute standard deviation of data
#[pyfunction]
fn std_dev(data: Vec<f64>) -> PyResult<f64> {
    if data.len() < 2 {
        return Ok(f64::NAN);
    }
    let mean_val = mean(data.clone())?;
    let variance =
        data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (data.len() as f64 - 1.0);
    Ok(variance.sqrt())
}

/// Compute correlation between two variables
#[pyfunction]
fn correlation(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return Ok(f64::NAN);
    }

    let x_mean = x.iter().sum::<f64>() / x.len() as f64;
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;

    let covariance = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
        .sum::<f64>()
        / (x.len() as f64 - 1.0);

    let x_std = std_dev(x.clone())?;
    let y_std = std_dev(y.clone())?;

    if x_std == 0.0 || y_std == 0.0 {
        Ok(0.0)
    } else {
        Ok(covariance / (x_std * y_std))
    }
}

/// Compute descriptive statistics summary
#[pyfunction]
fn descriptive_summary(py: Python, data: Vec<f64>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("count", data.len())?;

    if !data.is_empty() {
        let mean_val = data.iter().sum::<f64>() / data.len() as f64;
        dict.set_item("mean", mean_val)?;

        if data.len() >= 2 {
            let variance = data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>()
                / (data.len() as f64 - 1.0);
            dict.set_item("std", variance.sqrt())?;
            dict.set_item("variance", variance)?;

            // Min and max
            if let (Some(min), Some(max)) = (
                data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
                data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
            ) {
                dict.set_item("min", *min)?;
                dict.set_item("max", *max)?;
            }
        }
    }

    Ok(dict.into())
}

/// Convert alternative hypothesis string to Rust enum
fn parse_alternative(alternative: &str) -> PyResult<Alternative> {
    match alternative.to_lowercase().as_str() {
        "two-sided" | "two_sided" | "two.sided" => Ok(Alternative::TwoSided),
        "less" | "smaller" => Ok(Alternative::Less),
        "greater" | "larger" => Ok(Alternative::Greater),
        _ => Err(PyValueError::new_err(
            "alternative must be 'two-sided', 'less', or 'greater'",
        )),
    }
}

/// Convert TestResult to Python dictionary
fn test_result_to_dict(py: Python, result: &TestResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("p_value", result.p_value)?;
    dict.set_item("df", result.df)?;
    dict.set_item(
        "alternative",
        match result.alternative {
            Alternative::TwoSided => "two-sided",
            Alternative::Less => "less",
            Alternative::Greater => "greater",
        },
    )?;
    dict.set_item("null_value", result.null_value)?;
    Ok(dict.into())
}

/// One-sample t-test
#[pyfunction]
fn t_test_one_sample(
    py: Python,
    data: Vec<f64>,
    mu: f64,
    alternative: String,
) -> PyResult<Py<PyDict>> {
    let data_array = ndarray::Array1::from_vec(data);
    let alt = parse_alternative(&alternative)?;
    let result = t_test_one_sample_rs(&data_array, mu, alt)
        .map_err(|e| PyRuntimeError::new_err(format!("t-test failed: {:?}", e)))?;
    test_result_to_dict(py, &result)
}

/// Two-sample t-test (independent samples, equal variance assumed)
#[pyfunction]
fn t_test_two_sample(
    py: Python,
    x: Vec<f64>,
    y: Vec<f64>,
    alternative: String,
) -> PyResult<Py<PyDict>> {
    let x_array = ndarray::Array1::from_vec(x);
    let y_array = ndarray::Array1::from_vec(y);
    let alt = parse_alternative(&alternative)?;
    let result = t_test_two_sample_rs(&x_array, &y_array, alt)
        .map_err(|e| PyRuntimeError::new_err(format!("t-test failed: {:?}", e)))?;
    test_result_to_dict(py, &result)
}

/// Paired t-test
#[pyfunction]
fn t_test_paired(
    py: Python,
    x: Vec<f64>,
    y: Vec<f64>,
    alternative: String,
) -> PyResult<Py<PyDict>> {
    let x_array = ndarray::Array1::from_vec(x);
    let y_array = ndarray::Array1::from_vec(y);
    let alt = parse_alternative(&alternative)?;
    let result = t_test_paired_rs(&x_array, &y_array, alt)
        .map_err(|e| PyRuntimeError::new_err(format!("paired t-test failed: {:?}", e)))?;
    test_result_to_dict(py, &result)
}

/// Chi-square test of independence
#[pyfunction]
fn chi_square_test_independence(py: Python, observed: Vec<Vec<f64>>) -> PyResult<Py<PyDict>> {
    // Convert to ndarray matrix
    let n_rows = observed.len();
    if n_rows == 0 {
        return Err(PyValueError::new_err("observed must have at least one row"));
    }
    let n_cols = observed[0].len();
    let mut flat = Vec::new();
    for row in observed {
        if row.len() != n_cols {
            return Err(PyValueError::new_err("All rows must have same length"));
        }
        flat.extend(row);
    }
    let matrix = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| PyValueError::new_err(format!("Failed to create matrix: {}", e)))?;

    let result = chi_square_test_independence_rs(&matrix)
        .map_err(|e| PyRuntimeError::new_err(format!("chi-square test failed: {:?}", e)))?;
    test_result_to_dict(py, &result)
}

/// One-way ANOVA
#[pyfunction]
fn anova_one_way(py: Python, groups: Vec<Vec<f64>>) -> PyResult<Py<PyDict>> {
    let arrays: Vec<_> = groups
        .into_iter()
        .map(|g| ndarray::Array1::from_vec(g))
        .collect();
    let result = anova_one_way_rs(&arrays)
        .map_err(|e| PyRuntimeError::new_err(format!("ANOVA failed: {:?}", e)))?;
    test_result_to_dict(py, &result)
}

/// Shapiro-Wilk test for normality
#[pyfunction]
fn shapiro_wilk_test(py: Python, data: Vec<f64>) -> PyResult<Py<PyDict>> {
    let data_array = ndarray::Array1::from_vec(data);
    let result = shapiro_wilk_test_rs(&data_array)
        .map_err(|e| PyRuntimeError::new_err(format!("Shapiro-Wilk test failed: {:?}", e)))?;
    test_result_to_dict(py, &result)
}

/// Fit linear regression model
#[pyfunction]
fn linear_regression(py: Python, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<Py<PyDict>> {
    // Convert to ndarray
    let n_rows = x.len();
    if n_rows == 0 {
        return Err(PyValueError::new_err("X must have at least one row"));
    }
    if n_rows != y.len() {
        return Err(PyValueError::new_err(
            "X and y must have same number of rows",
        ));
    }

    let n_cols = x[0].len();
    let x_array =
        ndarray::Array2::from_shape_vec((n_rows, n_cols), x.into_iter().flatten().collect())
            .map_err(|e| PyValueError::new_err(format!("Failed to create X matrix: {}", e)))?;

    let y_array = ndarray::Array1::from_vec(y);

    // Fit OLS model
    let model = OLS::new();
    match model.fit(&x_array, &y_array) {
        Ok(results) => {
            let dict = PyDict::new(py);
            dict.set_item("coefficients", results.coefficients.to_vec())?;
            dict.set_item("r_squared", results.r_squared)?;
            dict.set_item("r_squared_adj", results.r_squared_adj)?;
            dict.set_item("sigma", results.sigma)?;
            dict.set_item("df_residual", results.df_residual)?;
            dict.set_item("df_model", results.df_model)?;

            if let Some(std_errors) = &results.std_errors {
                dict.set_item("std_errors", std_errors.to_vec())?;
            }

            if let Some(t_values) = &results.t_values {
                dict.set_item("t_values", t_values.to_vec())?;
            }

            if let Some(p_values) = &results.p_values {
                dict.set_item("p_values", p_values.to_vec())?;
            }

            if let Some(f_statistic) = &results.f_statistic {
                dict.set_item("f_statistic", f_statistic)?;
            }

            if let Some(f_p_value) = &results.f_p_value {
                dict.set_item("f_p_value", f_p_value)?;
            }

            Ok(dict.into())
        }
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Linear regression failed: {:?}",
            e
        ))),
    }
}

/// Fit mixed effects model (linear mixed model)
#[pyfunction]
fn mixed_effects(py: Python, data: &PyDataFrame, formula: String) -> PyResult<Py<PyDict>> {
    // TODO: Implement actual mixed effects model using so-models
    let dict = PyDict::new(py);

    // For now, return a placeholder result
    dict.set_item("fixed_effects", Vec::<f64>::new())?;
    dict.set_item("random_variances", Vec::<f64>::new())?;
    dict.set_item("residual_variance", 0.0)?;
    dict.set_item("log_likelihood", 0.0)?;
    dict.set_item("aic", 0.0)?;
    dict.set_item("bic", 0.0)?;
    dict.set_item(
        "message",
        "Mixed effects model placeholder - implement using so-models::mixed",
    )?;
    dict.set_item("formula", formula)?;
    dict.set_item("n_obs", data.n_rows())?;

    Ok(dict.into())
}

/// Fit ARIMA model
#[pyfunction]
fn fit_arima(py: Python, data: Vec<f64>, p: usize, d: usize, q: usize) -> PyResult<Py<PyDict>> {
    // Create TimeSeries from data with index as timestamps
    let values = ndarray::Array1::from_vec(data.clone());
    let timestamps: Vec<i64> = (0..data.len() as i64).collect();
    let ts = TimeSeries::new("series", timestamps, values, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create TimeSeries: {:?}", e)))?;

    // Create and fit ARIMA model using builder
    use so_tsa::arima::ARIMABuilder;
    let builder = ARIMABuilder::new(p, d, q);

    match builder.fit(&ts) {
        Ok(results) => {
            let dict = PyDict::new(py);
            dict.set_item("order", (p, d, q))?;

            // Collect coefficients
            let mut coefficients = Vec::new();
            if let Some(ar_coef) = &results.ar_coef {
                coefficients.extend_from_slice(&ar_coef.to_vec());
            }
            if let Some(ma_coef) = &results.ma_coef {
                coefficients.extend_from_slice(&ma_coef.to_vec());
            }
            if let Some(constant) = results.constant {
                coefficients.push(constant);
            }
            dict.set_item("coefficients", coefficients)?;

            dict.set_item("aic", results.aic)?;
            dict.set_item("bic", results.bic)?;
            dict.set_item("log_likelihood", results.log_likelihood)?;
            dict.set_item("sigma2", results.sigma2)?;

            // Add fitted values and residuals
            dict.set_item("fitted", results.fitted.to_vec())?;
            dict.set_item("residuals", results.residuals.to_vec())?;
            dict.set_item("n_obs", results.n_obs)?;

            Ok(dict.into())
        }
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "ARIMA fitting failed: {:?}",
            e
        ))),
    }
}

/// Split data into training and test sets
#[pyfunction]
fn train_test_split(data: Vec<f64>, test_size: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(PyValueError::new_err("test_size must be between 0 and 1"));
    }

    let split_idx = (data.len() as f64 * (1.0 - test_size)) as usize;
    let train = data[..split_idx].to_vec();
    let test = data[split_idx..].to_vec();

    Ok((train, test))
}

/// Get library version
#[pyfunction]
fn version() -> PyResult<String> {
    Ok("0.2.0".to_string())
}
