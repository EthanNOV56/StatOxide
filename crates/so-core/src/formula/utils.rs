//! Utility functions for formula handling
//!
//! This module provides utility functions for common operations in formula
//! parsing and evaluation, such as creating dummy variables, scaling,
//! polynomial features, etc.

use crate::data::{DataFrame, Series, SeriesValue};
use crate::formula::error::{FormulaError, FormulaResult};
use crate::formula::term::Term;
use ndarray::{Array1, Array2, Axis, concatenate};
use std::collections::HashSet;

/// Extract a column from DataFrame as a float array
pub fn extract_float_column(df: &DataFrame, name: &str) -> FormulaResult<Array1<f64>> {
    let series = df
        .get_column(name)
        .ok_or_else(|| FormulaError::variable_not_found(name, &df.column_names()))?;

    match series {
        Series::Float(arr) => Ok(arr.clone()),
        Series::Int(arr) => Ok(arr.mapv(|v| v as f64)),
        Series::Bool(arr) => Ok(arr.mapv(|v| if v { 1.0 } else { 0.0 })),
        _ => Err(FormulaError::TypeMismatch {
            variable: name.to_string(),
            expected_type: "numeric",
            actual_type: series.dtype().to_string(),
        }),
    }
}

/// Extract a column from DataFrame as a 2D matrix
pub fn extract_column_matrix(df: &DataFrame, name: &str) -> FormulaResult<Array2<f64>> {
    let col_data = extract_float_column(df, name)?;
    col_data
        .to_shape((col_data.len(), 1))
        .map(|m| m.into_owned())
        .map_err(|e| FormulaError::EvaluationError {
            message: format!("Failed to reshape column '{}': {}", name, e),
            context: None,
        })
}

/// Check if a column is categorical
pub fn is_categorical(df: &DataFrame, name: &str) -> bool {
    df.get_column(name)
        .map(|s| s.dtype() == "categorical")
        .unwrap_or(false)
}

/// Check if a column is numeric
pub fn is_numeric(df: &DataFrame, name: &str) -> bool {
    df.get_column(name)
        .map(|s| match s {
            Series::Float(_) | Series::Int(_) | Series::Bool(_) => true,
            _ => false,
        })
        .unwrap_or(false)
}

/// Get unique values from a column
pub fn unique_values(df: &DataFrame, name: &str) -> FormulaResult<Vec<String>> {
    let series = df
        .get_column(name)
        .ok_or_else(|| FormulaError::variable_not_found(name, &df.column_names()))?;

    match series {
        Series::Categorical(_, categories) => Ok(categories.clone()),
        Series::String(vec) => {
            let unique: HashSet<_> = vec.iter().collect();
            let mut unique_vec: Vec<String> = unique.into_iter().cloned().collect();
            unique_vec.sort();
            Ok(unique_vec)
        }
        _ => Err(FormulaError::TypeMismatch {
            variable: name.to_string(),
            expected_type: "categorical or string",
            actual_type: series.dtype().to_string(),
        }),
    }
}

/// Create dummy variables for a categorical series
pub fn create_dummy_variables(series: &Series, drop_first: bool) -> FormulaResult<Array2<f64>> {
    match series {
        Series::Categorical(codes, categories) => {
            let n_samples = codes.len();
            let n_categories = categories.len();

            if drop_first && n_categories <= 1 {
                return Err(FormulaError::EvaluationError {
                    message: "Cannot drop first category when there is only one category"
                        .to_string(),
                    context: None,
                });
            }

            let n_columns = if drop_first {
                n_categories - 1
            } else {
                n_categories
            };
            let mut matrix = Array2::zeros((n_samples, n_columns));

            for (i, &code) in codes.iter().enumerate() {
                if code as usize >= n_categories {
                    return Err(FormulaError::EvaluationError {
                        message: format!(
                            "Invalid category code {} (max: {})",
                            code,
                            n_categories - 1
                        ),
                        context: None,
                    });
                }

                if drop_first {
                    // Skip the first category (reference category)
                    if code > 0 {
                        matrix[(i, code as usize - 1)] = 1.0;
                    }
                } else {
                    matrix[(i, code as usize)] = 1.0;
                }
            }

            Ok(matrix)
        }
        _ => Err(FormulaError::TypeMismatch {
            variable: "unknown".to_string(), // We don't have the column name here
            expected_type: "categorical",
            actual_type: series.dtype().to_string(),
        }),
    }
}

/// Center a vector (subtract mean)
pub fn center_vector(vec: &Array1<f64>) -> FormulaResult<Array1<f64>> {
    if vec.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let mean = vec.mean().unwrap_or(0.0);
    Ok(vec.mapv(|x| x - mean))
}

/// Scale a vector (divide by standard deviation)
pub fn scale_vector(vec: &Array1<f64>) -> FormulaResult<Array1<f64>> {
    if vec.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let std = vec.std(1.0);
    if std == 0.0 {
        return Err(FormulaError::NumericalError {
            message: "Cannot scale vector with zero standard deviation".to_string(),
            operation: "scale".to_string(),
        });
    }

    Ok(vec.mapv(|x| x / std))
}

/// Standardize a vector (center and scale)
pub fn standardize_vector(vec: &Array1<f64>) -> FormulaResult<Array1<f64>> {
    let centered = center_vector(vec)?;
    scale_vector(&centered)
}

/// Create polynomial features from a vector
pub fn polynomial_features(vec: &Array1<f64>, degree: usize) -> FormulaResult<Array2<f64>> {
    if degree < 1 {
        return Err(FormulaError::NumericalError {
            message: format!("Polynomial degree must be >= 1, got {}", degree),
            operation: "poly".to_string(),
        });
    }

    let n_samples = vec.len();
    let mut features = Array2::zeros((n_samples, degree));

    for d in 1..=degree {
        let col: Array1<f64> = vec.mapv(|x| x.powi(d as i32));
        features.column_mut(d - 1).assign(&col);
    }

    Ok(features)
}

/// Create interaction term between two columns
pub fn interaction_term(col1: &Array1<f64>, col2: &Array1<f64>) -> FormulaResult<Array1<f64>> {
    if col1.len() != col2.len() {
        return Err(FormulaError::DimensionMismatch {
            message: "Columns have different lengths".to_string(),
            expected: format!("{} elements (same as first column)", col1.len()),
            actual: format!("{} elements", col2.len()),
        });
    }

    Ok(col1 * col2)
}

/// Horizontally stack matrices
pub fn hstack_matrices(matrices: &[Array2<f64>]) -> FormulaResult<Array2<f64>> {
    if matrices.is_empty() {
        return Ok(Array2::zeros((0, 0)));
    }

    // Check that all matrices have the same number of rows
    let nrows = matrices[0].nrows();
    for (i, matrix) in matrices.iter().enumerate() {
        if matrix.nrows() != nrows {
            return Err(FormulaError::DimensionMismatch {
                message: format!("Matrix {} has different number of rows", i),
                expected: format!("{} rows", nrows),
                actual: format!("{} rows", matrix.nrows()),
            });
        }
    }

    // Calculate total columns
    let total_cols: usize = matrices.iter().map(|m| m.ncols()).sum();

    // Create result matrix
    let mut result = Array2::zeros((nrows, total_cols));

    // Copy data column by column
    let mut col_offset = 0;
    for matrix in matrices {
        let ncols = matrix.ncols();

        for col in 0..ncols {
            for row in 0..nrows {
                result[(row, col_offset + col)] = matrix[(row, col)];
            }
        }

        col_offset += ncols;
    }

    Ok(result)
}

/// Vertically stack matrices
pub fn vstack_matrices(matrices: &[Array2<f64>]) -> FormulaResult<Array2<f64>> {
    if matrices.is_empty() {
        return Ok(Array2::zeros((0, 0)));
    }

    // Check that all matrices have the same number of columns
    let ncols = matrices[0].ncols();
    for (i, matrix) in matrices.iter().enumerate() {
        if matrix.ncols() != ncols {
            return Err(FormulaError::DimensionMismatch {
                message: format!("Matrix {} has different number of columns", i),
                expected: format!("{} columns", ncols),
                actual: format!("{} columns", matrix.ncols()),
            });
        }
    }

    // Calculate total rows
    let total_rows: usize = matrices.iter().map(|m| m.nrows()).sum();

    // Create result matrix
    let mut result = Array2::zeros((total_rows, ncols));

    // Copy data row by row
    let mut row_offset = 0;
    for matrix in matrices {
        let nrows = matrix.nrows();

        for row in 0..nrows {
            for col in 0..ncols {
                result[(row_offset + row, col)] = matrix[(row, col)];
            }
        }

        row_offset += nrows;
    }

    Ok(result)
}

/// Safe concatenation of matrices along axis
pub fn safe_concatenate(axis: Axis, matrices: &[Array2<f64>]) -> FormulaResult<Array2<f64>> {
    if matrices.is_empty() {
        return Ok(Array2::zeros((0, 0)));
    }

    // Convert to views
    let views: Vec<_> = matrices.iter().map(|m| m.view()).collect();

    concatenate(axis, &views).map_err(|e| FormulaError::EvaluationError {
        message: format!("Failed to concatenate matrices: {}", e),
        context: Some("This can happen if matrices have incompatible dimensions".to_string()),
    })
}

/// Safe stacking of matrices along axis
// pub fn safe_stack(axis: Axis, matrices: &[Array2<f64>]) -> FormulaResult<Array2<f64>> {
//     if matrices.is_empty() {
//         return Ok(Array2::zeros((0, 0)));
//     }

//     // Convert to views
//     let views: Vec<_> = matrices.iter().map(|m| m.view()).collect();

//     stack(axis, &views)
//         .map(|m| m.into_owned())
//         .map_err(|e| FormulaError::EvaluationError {
//             message: format!("Failed to stack matrices: {}", e),
//             context: Some("This can happen if matrices have incompatible dimensions".to_string()),
//         })
// }

/// Compute column means
pub fn column_means(matrix: &Array2<f64>) -> FormulaResult<Array1<f64>> {
    if matrix.is_empty() {
        return Ok(Array1::zeros(0));
    }

    Ok(matrix
        .mean_axis(Axis(0))
        .ok_or_else(|| FormulaError::NumericalError {
            message: "Failed to compute column means".to_string(),
            operation: "mean".to_string(),
        })?)
}

/// Compute column standard deviations
pub fn column_stds(matrix: &Array2<f64>, ddof: f64) -> FormulaResult<Array1<f64>> {
    if matrix.is_empty() {
        return Ok(Array1::zeros(0));
    }

    Ok(matrix.std_axis(Axis(0), ddof))
}

/// Compute column sums
pub fn column_sums(matrix: &Array2<f64>) -> FormulaResult<Array1<f64>> {
    if matrix.is_empty() {
        return Ok(Array1::zeros(0));
    }

    Ok(matrix.sum_axis(Axis(0)))
}

/// Compute column minimums
pub fn column_mins(matrix: &Array2<f64>) -> FormulaResult<Array1<f64>> {
    if matrix.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let mut mins = Array1::zeros(matrix.ncols());
    for col in 0..matrix.ncols() {
        let column = matrix.column(col);
        mins[col] = column.fold(f64::INFINITY, |acc, &x| acc.min(x));
    }

    Ok(mins)
}

/// Compute column maximums
pub fn column_maxs(matrix: &Array2<f64>) -> FormulaResult<Array1<f64>> {
    if matrix.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let mut maxs = Array1::zeros(matrix.ncols());
    for col in 0..matrix.ncols() {
        let column = matrix.column(col);
        maxs[col] = column.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    }

    Ok(maxs)
}

/// Validate that a matrix is 2D
pub fn ensure_2d<T>(matrix: &Array2<T>) -> FormulaResult<()> {
    if matrix.ndim() != 2 {
        return Err(FormulaError::DimensionMismatch {
            message: "Expected 2-dimensional matrix".to_string(),
            expected: "2 dimensions".to_string(),
            actual: format!("{} dimensions", matrix.ndim()),
        });
    }
    Ok(())
}

/// Validate matrix shape
pub fn ensure_shape(
    matrix: &Array2<f64>,
    expected_rows: usize,
    expected_cols: Option<usize>,
) -> FormulaResult<()> {
    ensure_2d(matrix)?;

    if matrix.nrows() != expected_rows {
        return Err(FormulaError::DimensionMismatch {
            message: format!(
                "Matrix has {} rows, expected {}",
                matrix.nrows(),
                expected_rows
            ),
            expected: format!("{} rows", expected_rows),
            actual: format!("{} rows", matrix.nrows()),
        });
    }

    if let Some(expected_cols) = expected_cols {
        if matrix.ncols() != expected_cols {
            return Err(FormulaError::DimensionMismatch {
                message: format!(
                    "Matrix has {} columns, expected {}",
                    matrix.ncols(),
                    expected_cols
                ),
                expected: format!("{} columns", expected_cols),
                actual: format!("{} columns", matrix.ncols()),
            });
        }
    }

    Ok(())
}

/// Check for NaN or Inf values
pub fn check_finite(matrix: &Array2<f64>) -> FormulaResult<()> {
    for &value in matrix.iter() {
        if !value.is_finite() {
            return Err(FormulaError::NumericalError {
                message: "Matrix contains non-finite values (NaN or Inf)".to_string(),
                operation: "validation".to_string(),
            });
        }
    }
    Ok(())
}

/// Check for constant columns (zero variance)
pub fn check_constant_columns(matrix: &Array2<f64>, tolerance: f64) -> FormulaResult<Vec<usize>> {
    let mut constant_cols = Vec::new();

    for col in 0..matrix.ncols() {
        let column = matrix.column(col);
        let first = column[0];
        let mut is_constant = true;

        for &value in column.iter().skip(1) {
            if (value - first).abs() > tolerance {
                is_constant = false;
                break;
            }
        }

        if is_constant {
            constant_cols.push(col);
        }
    }

    Ok(constant_cols)
}

/// Get all variables mentioned in a term
pub fn term_variables(term: &Term) -> HashSet<String> {
    let mut vars = HashSet::new();

    match &term.kind {
        crate::formula::term::TermKind::Variable(name) => {
            vars.insert(name.clone());
        }
        crate::formula::term::TermKind::Interaction(interaction) => {
            for var in &interaction.variables {
                vars.insert(var.clone());
            }
        }
        crate::formula::term::TermKind::Function { args, .. } => {
            for arg in args {
                vars.extend(term_variables(arg));
            }
        }
        crate::formula::term::TermKind::Parenthesized(args) => {
            for arg in args {
                vars.extend(term_variables(arg));
            }
        }
    }

    vars
}

/// Check if a formula contains a specific variable
pub fn contains_variable(term: &Term, variable: &str) -> bool {
    match &term.kind {
        crate::formula::term::TermKind::Variable(name) => name == variable,
        crate::formula::term::TermKind::Interaction(interaction) => {
            interaction.variables.iter().any(|v| v == variable)
        }
        crate::formula::term::TermKind::Function { args, .. } => {
            args.iter().any(|arg| contains_variable(arg, variable))
        }
        crate::formula::term::TermKind::Parenthesized(args) => {
            args.iter().any(|arg| contains_variable(arg, variable))
        }
    }
}

/// Count the number of terms of a specific type
pub fn count_terms_by_type(terms: &[Term], term_type: crate::formula::term::TermType) -> usize {
    terms
        .iter()
        .filter(|term| term.term_type == term_type)
        .count()
}

/// Convert a series value to a string representation
pub fn series_value_to_string(value: &SeriesValue) -> String {
    match value {
        SeriesValue::Float(f) => format!("{}", f),
        SeriesValue::Int(i) => format!("{}", i),
        SeriesValue::Bool(b) => format!("{}", b),
        SeriesValue::String(s) => s.clone(),
    }
}

/// Parse a string to a series value
pub fn string_to_series_value(s: &str) -> FormulaResult<SeriesValue> {
    // Try to parse as float
    if let Ok(f) = s.parse::<f64>() {
        return Ok(SeriesValue::Float(f));
    }

    // Try to parse as int
    if let Ok(i) = s.parse::<i64>() {
        return Ok(SeriesValue::Int(i));
    }

    // Try to parse as bool
    match s.to_lowercase().as_str() {
        "true" | "t" | "1" | "yes" | "y" => return Ok(SeriesValue::Bool(true)),
        "false" | "f" | "0" | "no" | "n" => return Ok(SeriesValue::Bool(false)),
        _ => {}
    }

    // Default to string
    Ok(SeriesValue::String(s.to_string()))
}
