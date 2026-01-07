//! Formula-specific error types
//!
//! This module provides detailed error types for formula parsing and evaluation.

use crate::data::DataError;
use thiserror::Error;

/// Errors that can occur during formula parsing and evaluation
#[derive(Debug, Error)]
pub enum FormulaError {
    /// Syntax errors in the formula string
    #[error("Syntax error at position {position}: {message}")]
    Syntax {
        position: usize,
        message: String,
        context: Option<String>,
    },

    /// Variable not found in the DataFrame
    #[error(
        "Variable '{variable}' not found in DataFrame. Available variables: {available_vars:?}"
    )]
    VariableNotFound {
        variable: String,
        available_vars: Vec<String>,
    },

    /// Variable type mismatch
    #[error("Variable '{variable}' has type {actual_type}, but {expected_type} was expected")]
    TypeMismatch {
        variable: String,
        expected_type: &'static str,
        actual_type: String,
    },

    /// Function application errors
    #[error("Error in function '{function}': {message}")]
    FunctionError {
        function: String,
        message: String,
        argument: Option<String>,
    },

    /// Interaction term errors
    #[error("Invalid interaction term: {message}")]
    InteractionError {
        message: String,
        variables: Vec<String>,
    },

    /// Dimension mismatch in formula evaluation
    #[error("Dimension mismatch: {message}. Expected {expected}, got {actual}")]
    DimensionMismatch {
        message: String,
        expected: String,
        actual: String,
    },

    /// Invalid formula structure
    #[error("Invalid formula structure: {message}")]
    InvalidStructure {
        message: String,
        suggestion: Option<String>,
    },

    /// Missing response variable
    #[error("Response variable is required but not provided")]
    MissingResponse,

    /// Formula evaluation errors
    #[error("Formula evaluation error: {message}")]
    EvaluationError {
        message: String,
        context: Option<String>,
    },

    /// Data-related errors that bubble up from the data layer
    #[error("Data error in formula evaluation: {0}")]
    Data(#[from] DataError),

    /// Numerical computation errors
    #[error("Numerical error: {message}")]
    NumericalError { message: String, operation: String },
}

/// Parser-specific errors
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Unexpected end of input at position {position}")]
    UnexpectedEof { position: usize },

    #[error("Unexpected character '{character}' at position {position}")]
    UnexpectedCharacter { character: char, position: usize },

    #[error("Expected {expected} at position {position}, found '{found}'")]
    Expected {
        expected: String,
        found: char,
        position: usize,
    },

    #[error("Missing closing delimiter '{delimiter}'")]
    MissingDelimiter { delimiter: char, position: usize },

    #[error("Invalid token '{token}' at position {position}")]
    InvalidToken { token: String, position: usize },

    #[error("Empty formula")]
    EmptyFormula,
}

/// Result type alias for formula operations
pub type FormulaResult<T> = std::result::Result<T, FormulaError>;

impl FormulaError {
    /// Create a syntax error with context
    pub fn syntax(position: usize, message: impl Into<String>) -> Self {
        FormulaError::Syntax {
            position,
            message: message.into(),
            context: None,
        }
    }

    /// Create a syntax error with context
    pub fn syntax_with_context(
        position: usize,
        message: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        FormulaError::Syntax {
            position,
            message: message.into(),
            context: Some(context.into()),
        }
    }

    /// Create a variable not found error
    pub fn variable_not_found(variable: &str, available_vars: &[&str]) -> Self {
        FormulaError::VariableNotFound {
            variable: variable.to_string(),
            available_vars: available_vars.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Create a function error
    pub fn function(function: &str, message: impl Into<String>) -> Self {
        FormulaError::FunctionError {
            function: function.to_string(),
            message: message.into(),
            argument: None,
        }
    }

    /// Create a function error with argument
    pub fn function_with_arg(function: &str, argument: &str, message: impl Into<String>) -> Self {
        FormulaError::FunctionError {
            function: function.to_string(),
            message: message.into(),
            argument: Some(argument.to_string()),
        }
    }
}
