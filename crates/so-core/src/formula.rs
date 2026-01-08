//! R-style formula parsing and design matrix construction
//!
//! This module provides formula parsing similar to R's formula syntax,
//! used for specifying statistical models.

use crate::data::{DataFrame, Matrix, Series};
pub use crate::formula::error::{FormulaError, FormulaResult};

use std::collections::HashSet;
use std::str::FromStr;

pub mod error;
mod expander;
mod parser;
mod term;
mod utils;

#[cfg(test)]
mod tests;

use ndarray::Array1;
pub use term::{Interaction, Term, TermKind, TermType};
pub use utils::*;

pub use parser::FormulaParser;

pub type Result<T> = std::result::Result<T, FormulaError>;

/// A parsed formula specifying a statistical model
#[derive(Debug, Clone, PartialEq)]
pub struct Formula {
    /// Response variable (left-hand side)
    pub response: Option<String>,

    /// Terms on the right-hand side
    pub terms: Vec<Term>,

    /// Whether to include an intercept
    pub has_intercept: bool,

    /// Original formula string
    pub original: String,
}

impl Formula {
    /// Parse a formula from a string
    pub fn parse(formula: &str) -> Result<Self> {
        FormulaParser::parse(formula)
    }

    /// Create a formula with a response
    pub fn with_response(response: &str, terms: Vec<Term>) -> Self {
        Self {
            response: Some(response.to_string()),
            terms,
            has_intercept: true, // Default to include intercept
            original: String::new(),
        }
    }

    /// Create a formula without a response
    pub fn without_response(terms: Vec<Term>) -> Self {
        Self {
            response: None,
            terms,
            has_intercept: true,
            original: String::new(),
        }
    }

    /// Remove the intercept from the formula
    pub fn without_intercept(mut self) -> Self {
        self.has_intercept = false;
        self
    }

    /// Get all variable names mentioned in the formula
    pub fn variables(&self) -> HashSet<&str> {
        let mut vars = HashSet::new();

        if let Some(resp) = &self.response {
            vars.insert(resp.as_str());
        }

        for term in &self.terms {
            match &term.kind {
                TermKind::Variable(name) => {
                    vars.insert(name.as_str());
                }
                TermKind::Interaction(interaction) => {
                    for var in &interaction.variables {
                        vars.insert(var.as_str());
                    }
                }
                TermKind::Function { name: _, args } => {
                    for arg in args {
                        if let TermKind::Variable(name) = &arg.kind {
                            vars.insert(name.as_str());
                        }
                    }
                }
                TermKind::Parenthesized(terms) => {
                    for term in terms {
                        if let TermKind::Variable(name) = &term.kind {
                            vars.insert(name.as_str());
                        }
                    }
                }
            }
        }

        vars
    }

    /// Check if formula has a response variable
    pub fn has_response(&self) -> bool {
        self.response.is_some()
    }

    /// Create design matrix and response vector from DataFrame
    pub fn design_matrix(&self, df: &DataFrame) -> FormulaResult<(Matrix, Option<Array1<f64>>)> {
        // 1. Extract response if present
        let response = if let Some(resp_name) = &self.response {
            let series = df
                .get_column(resp_name)
                .ok_or_else(|| FormulaError::variable_not_found(resp_name, &df.column_names()))?;

            // Convert to float array
            match series {
                Series::Float(arr) => Some(arr.clone()),
                Series::Int(arr) => Some(arr.mapv(|v| v as f64)),
                Series::Bool(arr) => Some(arr.mapv(|v| if v { 1.0 } else { 0.0 })),
                _ => {
                    return Err(FormulaError::TypeMismatch {
                        variable: resp_name.clone(),
                        expected_type: "numeric",
                        actual_type: series.dtype().to_string(),
                    });
                }
            }
        } else {
            None
        };

        // 2. Build design matrix
        let design = self.build_design_matrix(df)?;

        Ok((design, response))
    }

    /// Build only the design matrix (without response)
    pub fn build_design_matrix(&self, df: &DataFrame) -> FormulaResult<Matrix> {
        let mut matrices = Vec::new();

        // Add intercept if requested
        if self.has_intercept {
            let intercept = Array1::ones(df.nrows() as usize);
            let intercept_matrix = intercept
                .to_shape((df.nrows(), 1))
                .map(|m| m.into_owned())
                .map_err(|e| FormulaError::EvaluationError {
                    message: format!("Failed to create intercept matrix: {}", e),
                    context: None,
                })?;
            matrices.push(("(Intercept)".to_string(), intercept_matrix));
        }

        // Process each term
        for term in &self.terms {
            let term_matrix = term.to_matrix(df)?;
            matrices.push((term.to_string(), term_matrix.into()));
        }

        // Check that all matrices have the same number of rows
        let nrows = df.nrows();
        for (name, matrix) in &matrices {
            if matrix.nrows() != nrows {
                return Err(FormulaError::DimensionMismatch {
                    message: format!("Term '{}' has incorrect number of rows", name),
                    expected: format!("{} rows (matching DataFrame)", nrows),
                    actual: format!("{} rows", matrix.nrows()),
                });
            }
        }

        // Concatenate all matrices horizontally
        if matrices.is_empty() {
            return Ok(Matrix::zeros((nrows, 0)));
        }

        let standard_matrices: Vec<Matrix> = matrices
            .into_iter()
            .map(|(_, matrix)| {
                if matrix.is_standard_layout() {
                    matrix.to_owned()
                } else {
                    matrix.as_standard_layout().to_owned()
                }
            })
            .collect();

        let matrix_views: Vec<ndarray::ArrayView2<f64>> = standard_matrices
            .iter()
            .map(|matrix| matrix.view())
            .collect();

        let design = ndarray::concatenate(ndarray::Axis(1), &matrix_views).map_err(|e| {
            FormulaError::EvaluationError {
                message: format!("Failed to stack design matrix columns: {}", e),
                context: Some(
                    "This can happen if matrices have incompatible dimensions".to_string(),
                ),
            }
        })?;

        Ok(design)
    }

    /// Build interaction matrices
    fn build_interaction_matrices(&self, df: &DataFrame) -> Result<Vec<(String, Matrix)>> {
        let mut interactions = Vec::new();

        for term in &self.terms {
            if let TermKind::Interaction(interaction) = &term.kind {
                let interaction_matrix = interaction.to_matrix(df)?;
                interactions.push((term.to_string(), interaction_matrix));
            }
        }

        Ok(interactions)
    }

    /// Get the names of columns in the design matrix
    pub fn design_matrix_names(&self, df: &DataFrame) -> Result<Vec<String>> {
        let mut names = Vec::new();

        // Add intercept name
        if self.has_intercept {
            names.push("(Intercept)".to_string());
        }

        // Add term names
        for term in &self.terms {
            match &term.kind {
                TermKind::Variable(name) => {
                    // Check if variable is categorical
                    if let Some(series) = df.get_column(name) {
                        if series.dtype() == "categorical" {
                            // For categorical variables, we'll create dummy variables
                            if let Series::Categorical(_, categories) = series {
                                for cat in categories {
                                    names.push(format!("{}[{}]", name, cat));
                                }
                            }
                        } else {
                            names.push(name.clone());
                        }
                    }
                }
                TermKind::Interaction(_) => {
                    names.push(term.to_string());
                }
                TermKind::Function { name, args } => {
                    let arg_names: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                    names.push(format!("{}({})", name, arg_names.join(", ")));
                }
                TermKind::Parenthesized(terms) => {
                    let mut names = Vec::new();
                    for term in terms {
                        names.push(term.to_string());
                    }
                    names.push(format!("({})", names.join(" + ")));
                }
            }
        }

        Ok(names)
    }
}

impl Formula {
    /// Expand parenthesized terms in the formula
    pub fn expand_parentheses(&mut self) {
        let mut expanded_terms = Vec::new();

        for term in &self.terms {
            if term.is_parenthesized() {
                // Expand parenthesized terms
                expanded_terms.extend(term.expand_parentheses());
            } else {
                expanded_terms.push(term.clone());
            }
        }

        self.terms = expanded_terms;
    }

    /// Apply distributive law for interactions
    pub fn distribute_interactions(&mut self) {
        // This is a simplified version that handles (a+b):c -> a:c + b:c
        let mut new_terms = Vec::new();

        for term in &self.terms {
            match &term.kind {
                TermKind::Interaction(interaction) => {
                    // Check if any factor is parenthesized
                    let mut has_parenthesized = false;
                    let mut expanded_interactions = Vec::new();

                    for _var in &interaction.variables {
                        // In a real implementation, we would need to check
                        // if the variable corresponds to a parenthesized term
                        // This is a simplified placeholder
                    }

                    if has_parenthesized {
                        new_terms.extend(expanded_interactions);
                    } else {
                        new_terms.push(term.clone());
                    }
                }
                _ => {
                    new_terms.push(term.clone());
                }
            }
        }

        self.terms = new_terms;
    }

    /// Fully expand the formula
    pub fn fully_expand(&mut self) {
        self.expand_parentheses();
        self.distribute_interactions();
        // Remove duplicates
        self.terms.dedup();
    }

    /// Get a string representation of the fully expanded formula
    pub fn to_expanded_string(&self) -> String {
        let mut cloned = self.clone();
        cloned.fully_expand();
        cloned.to_string()
    }
}

impl FromStr for Formula {
    type Err = FormulaError;

    fn from_str(s: &str) -> Result<Self> {
        Formula::parse(s)
    }
}

impl std::fmt::Display for Formula {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(response) = &self.response {
            write!(f, "{} ~ ", response)?;
        } else {
            write!(f, "~ ")?;
        }

        if self.terms.is_empty() {
            if self.has_intercept {
                write!(f, "1")?;
            } else {
                write!(f, "0")?;
            }
        } else {
            let mut first = true;

            if !self.has_intercept {
                write!(f, "0")?;
                first = false;
            }

            for term in &self.terms {
                if !first {
                    write!(f, " + ")?;
                }
                write!(f, "{}", term)?;
                first = false;
            }
        }

        Ok(())
    }
}
