//! Term types for formula specification
//!
//! This module defines the types representing terms in a formula,
//! such as variables, interactions, and function applications.

use crate::data::*;
use crate::formula::error::{FormulaError, FormulaResult};
use ndarray::{Array1, Array2};
use std::fmt;

/// Type of term in a formula
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermType {
    /// Main effect
    Main,
    /// Interaction effect
    Interaction,
    /// Function application
    Function,
}

/// Kind of term
#[derive(Debug, Clone, PartialEq)]
pub enum TermKind {
    /// Simple variable
    Variable(String),
    /// Interaction between variables
    Interaction(Box<Interaction>),
    /// Function application
    Function { name: String, args: Vec<Term> },
    /// Parenthesized expression (multiple terms)
    Parenthesized(Vec<Term>),
}

/// A term in a formula
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    /// The kind of term
    pub kind: TermKind,
    /// Type of term
    pub term_type: TermType,
    /// Whether the term is expanded (e.g., for categorical variables)
    pub expanded: bool,
}

impl Term {
    /// Create a new variable term
    pub fn variable(name: &str) -> Self {
        Self {
            kind: TermKind::Variable(name.to_string()),
            term_type: TermType::Main,
            expanded: false,
        }
    }

    /// Create a new interaction term
    pub fn interaction(variables: Vec<String>) -> Self {
        Self {
            kind: TermKind::Interaction(Box::new(Interaction::new(variables))),
            term_type: TermType::Interaction,
            expanded: false,
        }
    }

    /// Create a new function term
    pub fn function(name: &str, args: Vec<Term>) -> Self {
        Self {
            kind: TermKind::Function {
                name: name.to_string(),
                args,
            },
            term_type: TermType::Function,
            expanded: false,
        }
    }

    /// Get the term as a string representation
    pub fn as_str(&self) -> String {
        self.to_string()
    }

    /// Check if the term is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self.kind, TermKind::Variable(_))
    }

    /// Check if the term is an interaction
    pub fn is_interaction(&self) -> bool {
        matches!(self.kind, TermKind::Interaction(_))
    }

    /// Check if the term is a function
    pub fn is_function(&self) -> bool {
        matches!(self.kind, TermKind::Function { .. })
    }

    /// Get variable name if this is a variable term
    pub fn as_variable(&self) -> Option<&str> {
        if let TermKind::Variable(name) = &self.kind {
            Some(name)
        } else {
            None
        }
    }

    /// Get interaction if this is an interaction term
    pub fn as_interaction(&self) -> Option<&Interaction> {
        if let TermKind::Interaction(interaction) = &self.kind {
            Some(interaction)
        } else {
            None
        }
    }

    /// Convert term to design matrix column(s)
    pub fn to_matrix(&self, df: &DataFrame) -> FormulaResult<Matrix> {
        match &self.kind {
            TermKind::Variable(name) => self.variable_to_matrix(name, df),
            TermKind::Interaction(interaction) => interaction.to_matrix(df),
            TermKind::Function { name, args } => self.function_to_matrix(name, args, df),
            TermKind::Parenthesized(terms) => self.parenthesized_to_matrix(terms, df),
        }
    }

    /// Convert parenthesized term to matrix
    fn parenthesized_to_matrix(&self, terms: &[Term], df: &DataFrame) -> FormulaResult<Matrix> {
        if terms.is_empty() {
            return Ok(Array2::zeros((df.nrows(), 0)));
        }

        let matrices: std::result::Result<Vec<Array2<f64>>, FormulaError> =
            terms.iter().map(|term| term.to_matrix(df)).collect();
        let matrices = matrices?;

        crate::formula::utils::hstack_matrices(&matrices)
    }

    /// Convert variable term to matrix
    fn variable_to_matrix(&self, name: &str, df: &DataFrame) -> FormulaResult<Matrix> {
        let series = df
            .get_column(name)
            .ok_or_else(|| FormulaError::variable_not_found(name, &df.column_names()))?;

        match series {
            Series::Float(arr) => {
                // Reshape to column vector
                let matrix = arr
                    .to_shape((arr.len(), 1))
                    .map(|m| m.into_owned())
                    .map_err(|e| FormulaError::EvaluationError {
                        message: format!("Failed to reshape column '{}': {}", name, e),
                        context: None,
                    })?;
                Ok(matrix)
            }
            Series::Int(arr) => {
                let float_arr: Array1<f64> = arr.mapv(|v| v as f64);
                let matrix = float_arr
                    .to_shape((float_arr.len(), 1))
                    .map(|m| m.into_owned())
                    .map_err(|e| FormulaError::EvaluationError {
                        message: format!("Failed to reshape column '{}': {}", name, e),
                        context: None,
                    })?;
                Ok(matrix)
            }
            Series::Bool(arr) => {
                let float_arr: Array1<f64> = arr.mapv(|v| if v { 1.0 } else { 0.0 });
                let matrix = float_arr
                    .to_shape((float_arr.len(), 1))
                    .map(|m| m.into_owned())
                    .map_err(|e| FormulaError::EvaluationError {
                        message: format!("Failed to reshape column '{}': {}", name, e),
                        context: None,
                    })?;
                Ok(matrix)
            }
            Series::Categorical(_, categories) => {
                // Create dummy variables
                self.create_dummy_variables(series, categories)
            }
            Series::String(_) => Err(FormulaError::TypeMismatch {
                variable: name.to_string(),
                expected_type: "numeric or categorical",
                actual_type: "string".to_string(),
            }),
        }
    }

    /// Apply a function to create transformed variables
    fn function_to_matrix(
        &self,
        name: &str,
        args: &[Term],
        df: &DataFrame,
    ) -> FormulaResult<Matrix> {
        match name.to_lowercase().as_str() {
            "log" | "log10" | "log2" | "exp" | "sqrt" | "abs" => {
                // Unary functions
                if args.len() != 1 {
                    return Err(FormulaError::function(
                        name,
                        format!("Expected 1 argument, got {}", args.len()),
                    ));
                }

                let arg_matrix = args[0].to_matrix(df)?;

                // Ensure input is 2D
                if arg_matrix.ndim() != 2 {
                    return Err(FormulaError::function(
                        name,
                        format!(
                            "Function expects 2D matrix input, got {}-dimensional",
                            arg_matrix.ndim()
                        ),
                    ));
                }

                // Apply function transformation
                let transformed: Matrix = match name.to_lowercase().as_str() {
                    "log" => {
                        if arg_matrix.iter().any(|&x| x <= 0.0) {
                            return Err(FormulaError::NumericalError {
                                message: "log() requires positive values".to_string(),
                                operation: "log".to_string(),
                            });
                        }
                        arg_matrix.mapv(f64::ln)
                    }
                    "log10" => {
                        if arg_matrix.iter().any(|&x| x <= 0.0) {
                            return Err(FormulaError::NumericalError {
                                message: "log10() requires positive values".to_string(),
                                operation: "log10".to_string(),
                            });
                        }
                        arg_matrix.mapv(f64::log10)
                    }
                    "log2" => {
                        if arg_matrix.iter().any(|&x| x <= 0.0) {
                            return Err(FormulaError::NumericalError {
                                message: "log2() requires positive values".to_string(),
                                operation: "log2".to_string(),
                            });
                        }
                        arg_matrix.mapv(f64::log2)
                    }
                    "exp" => arg_matrix.mapv(f64::exp),
                    "sqrt" => {
                        if arg_matrix.iter().any(|&x| x < 0.0) {
                            return Err(FormulaError::NumericalError {
                                message: "sqrt() requires non-negative values".to_string(),
                                operation: "sqrt".to_string(),
                            });
                        }
                        arg_matrix.mapv(f64::sqrt)
                    }
                    "abs" => arg_matrix.mapv(f64::abs),
                    _ => unreachable!(),
                };

                Ok(transformed)
            }
            "poly" => {
                // Polynomial expansion
                self.create_polynomial(args, df)
            }
            "scale" => {
                // Standardization
                self.scale_variable(args, df)
            }
            "center" => {
                // Centering (subtract mean)
                self.center_variable(args, df)
            }
            "standardize" => {
                // Standardization (center and scale)
                self.standardize_variable(args, df)
            }
            "I" => {
                // Identity function (for protecting arithmetic operations)
                if args.len() != 1 {
                    return Err(FormulaError::function(
                        "I",
                        format!("Expected 1 argument, got {}", args.len()),
                    ));
                }
                args[0].to_matrix(df)
            }
            _ => Err(FormulaError::function(
                name,
                format!("Function '{}' not supported", name),
            )),
        }
    }

    /// Create dummy variables for categorical data
    fn create_dummy_variables(
        &self,
        series: &Series,
        categories: &[String],
    ) -> FormulaResult<Matrix> {
        if let Series::Categorical(codes, _) = series {
            let n_samples = codes.len();
            let n_categories = categories.len();

            // Create a 2D dummy variable matrix
            let mut matrix = Array2::zeros((n_samples, n_categories));

            for (i, &code) in codes.iter().enumerate() {
                if (code as usize) < n_categories {
                    matrix[(i, code as usize)] = 1.0;
                } else {
                    return Err(FormulaError::EvaluationError {
                        message: format!(
                            "Invalid category code {} for variable with {} categories",
                            code, n_categories
                        ),
                        context: None,
                    });
                }
            }

            Ok(matrix)
        } else {
            unreachable!("Should only be called with categorical series")
        }
    }

    /// Create polynomial expansion
    fn create_polynomial(&self, args: &[Term], df: &DataFrame) -> FormulaResult<Matrix> {
        if args.len() != 2 {
            return Err(FormulaError::function(
                "poly",
                format!("Expected 2 arguments, got {}", args.len()),
            ));
        }

        let var_matrix = args[0].to_matrix(df)?;

        // Ensure is 2D
        if var_matrix.ndim() != 2 {
            return Err(FormulaError::function(
                "poly",
                format!("Expected 2D matrix, got {}-dimensional", var_matrix.ndim()),
            ));
        }

        if var_matrix.ncols() != 1 {
            return Err(FormulaError::function(
                "poly",
                "poly() expects single column for first argument",
            ));
        }

        // Parse degree
        let degree_term = &args[1];
        let degree = if let TermKind::Variable(deg_str) = &degree_term.kind {
            deg_str.parse::<usize>().map_err(|_| {
                FormulaError::function("poly", format!("Invalid degree: {}", deg_str))
            })?
        } else {
            return Err(FormulaError::function(
                "poly",
                "poly() degree must be a constant",
            ));
        };

        if degree < 1 {
            return Err(FormulaError::function(
                "poly",
                format!("poly() degree must be >= 1, got {}", degree),
            ));
        }

        // Create polynomial columns
        let n_samples = var_matrix.nrows();
        let mut poly_matrix = Array2::zeros((n_samples, degree));

        let data = var_matrix.column(0);

        for d in 1..=degree {
            let pow_col: Array1<f64> = data.mapv(|x| x.powi(d as i32));
            poly_matrix.column_mut(d - 1).assign(&pow_col);
        }

        Ok(poly_matrix)
    }

    /// Scale/standardize a variable
    fn scale_variable(&self, args: &[Term], df: &DataFrame) -> FormulaResult<Matrix> {
        if args.len() != 1 {
            return Err(FormulaError::function_with_arg(
                "scale",
                "",
                format!("Expected 1 argument, got {}", args.len()),
            ));
        }

        let var_matrix = args[0].to_matrix(df)?;
        if var_matrix.ncols() != 1 {
            return Err(FormulaError::function_with_arg(
                "scale",
                &args[0].to_string(),
                "scale() expects single column input",
            ));
        }

        let data = var_matrix.column(0);
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(1.0);

        if std == 0.0 {
            return Err(FormulaError::function_with_arg(
                "scale",
                &args[0].to_string(),
                "Cannot scale constant variable (std = 0)",
            ));
        }

        let scaled: Array1<f64> = data.mapv(|x| (x - mean) / std);
        scaled
            .to_shape((data.len(), 1))
            .map(|m| m.into_owned())
            .map_err(|e| FormulaError::EvaluationError {
                message: format!("Failed to reshape scaled variable: {}", e),
                context: None,
            })
    }

    /// Center a variable (subtract mean)
    fn center_variable(&self, args: &[Term], df: &DataFrame) -> FormulaResult<Matrix> {
        if args.len() != 1 {
            return Err(FormulaError::function_with_arg(
                "center",
                "",
                format!("Expected 1 argument, got {}", args.len()),
            ));
        }

        let var_matrix = args[0].to_matrix(df)?;
        if var_matrix.ncols() != 1 {
            return Err(FormulaError::function_with_arg(
                "center",
                &args[0].to_string(),
                "center() expects single column input",
            ));
        }

        let data = var_matrix.column(0);
        let mean = data.mean().unwrap_or(0.0);

        let centered: Array1<f64> = data.mapv(|x| x - mean);
        centered
            .to_shape((data.len(), 1))
            .map(|m| m.into_owned())
            .map_err(|e| FormulaError::EvaluationError {
                message: format!("Failed to reshape centered variable: {}", e),
                context: None,
            })
    }

    /// Standardize a variable (center and scale)
    fn standardize_variable(&self, args: &[Term], df: &DataFrame) -> FormulaResult<Matrix> {
        // Same as scale
        self.scale_variable(args, df)
    }

    /// Create interactions between expanded categorical variables
    fn create_interactions_between_expanded(
        &self,
        expanded_sets: &[Vec<Term>],
        result: &mut Vec<Term>,
    ) {
        if expanded_sets.is_empty() {
            return;
        }

        if expanded_sets.len() == 1 {
            result.extend(expanded_sets[0].iter().cloned());
            return;
        }

        // Recursively create interactions
        self.create_interactions_recursive(expanded_sets, result, 0, Vec::new());
    }

    fn create_interactions_recursive(
        &self,
        expanded_sets: &[Vec<Term>],
        result: &mut Vec<Term>,
        depth: usize,
        current: Vec<Term>,
    ) {
        if depth == expanded_sets.len() {
            if current.len() > 1 {
                // Create interaction term
                let var_names: Vec<String> = current
                    .iter()
                    .filter_map(|term| term.as_variable())
                    .map(|s| s.to_string())
                    .collect();

                if var_names.len() == current.len() {
                    result.push(Term::interaction(var_names));
                }
            } else if current.len() == 1 {
                result.push(current[0].clone());
            }
            return;
        }

        for term in &expanded_sets[depth] {
            let mut new_current = current.clone();
            new_current.push(term.clone());
            self.create_interactions_recursive(expanded_sets, result, depth + 1, new_current);
        }
    }

    /// Combine function arguments after expansion
    fn combine_function_arguments(
        &self,
        name: &str,
        expanded_args: &[Vec<Term>],
        result: &mut Vec<Term>,
        depth: usize,
        current: Vec<Term>,
    ) {
        if depth == expanded_args.len() {
            result.push(Term::function(name, current));
            return;
        }

        for term in &expanded_args[depth] {
            let mut new_current = current.clone();
            new_current.push(term.clone());
            self.combine_function_arguments(name, expanded_args, result, depth + 1, new_current);
        }
    }
}

impl Term {
    // ... existing methods ...

    /// Create a new parenthesized term
    pub fn parenthesized(terms: Vec<Term>) -> Self {
        Self {
            kind: TermKind::Parenthesized(terms),
            term_type: TermType::Main,
            expanded: false,
        }
    }

    /// Check if the term is parenthesized
    pub fn is_parenthesized(&self) -> bool {
        matches!(self.kind, TermKind::Parenthesized(_))
    }

    /// Get parenthesized terms if this is a parenthesized term
    pub fn as_parenthesized(&self) -> Option<&[Term]> {
        if let TermKind::Parenthesized(terms) = &self.kind {
            Some(terms)
        } else {
            None
        }
    }

    /// Expand parenthesized terms
    pub fn expand_parentheses(&self) -> Vec<Term> {
        match &self.kind {
            TermKind::Parenthesized(terms) => {
                // For now, just return the terms
                // The actual expansion will be handled by Formula
                terms.clone()
            }
            _ => vec![self.clone()],
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            TermKind::Variable(name) => write!(f, "{}", name),
            TermKind::Interaction(interaction) => write!(f, "{}", interaction),
            TermKind::Function { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            TermKind::Parenthesized(terms) => {
                write!(f, "(")?;
                for (i, term) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", term)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl Term {
    /// Expand categorical variables into dummy variables
    ///
    /// This method expands categorical variables into multiple dummy variables
    /// and also handles interactions and functions that involve categorical variables.
    ///
    /// # Arguments
    /// * `df` - The DataFrame containing the data
    ///
    /// # Returns
    /// A vector of terms with categorical variables expanded
    pub fn expand_categorical(&self, df: &DataFrame) -> FormulaResult<Vec<Term>> {
        match &self.kind {
            TermKind::Variable(name) => self.expand_variable_categorical(name, df),
            TermKind::Interaction(interaction) => {
                self.expand_interaction_categorical(interaction, df)
            }
            TermKind::Function { name, args } => self.expand_function_categorical(name, args, df),
            TermKind::Parenthesized(terms) => self.expand_parenthesized_categorical(terms, df),
        }
    }

    /// Expand a variable term (categorical -> dummy variables)
    fn expand_variable_categorical(&self, name: &str, df: &DataFrame) -> FormulaResult<Vec<Term>> {
        let series = df
            .get_column(name)
            .ok_or_else(|| FormulaError::variable_not_found(name, &df.column_names()))?;

        match series {
            Series::Categorical(_, categories) => {
                // Expand categorical variable into dummy variables
                let mut dummy_terms = Vec::with_capacity(categories.len());

                for category in categories {
                    // Create dummy variable name: variable[category]
                    let dummy_name = format!("{}[{}]", name, category);
                    dummy_terms.push(Term::variable(&dummy_name));
                }

                Ok(dummy_terms)
            }
            _ => {
                // Non-categorical variable, return as is
                Ok(vec![self.clone()])
            }
        }
    }

    /// Expand an interaction term with categorical variables
    fn expand_interaction_categorical(
        &self,
        interaction: &Interaction,
        df: &DataFrame,
    ) -> FormulaResult<Vec<Term>> {
        // Expand each variable in the interaction
        let mut expanded_variables: Vec<Vec<Term>> = Vec::new();

        for var_name in &interaction.variables {
            let var_term = Term::variable(var_name);
            let expanded = var_term.expand_categorical(df)?;
            expanded_variables.push(expanded);
        }

        // Generate all combinations of expanded variables
        let mut result = Vec::new();
        self.generate_interaction_combinations(&expanded_variables, 0, Vec::new(), &mut result);

        Ok(result)
    }

    /// Generate all combinations for interaction expansion
    fn generate_interaction_combinations(
        &self,
        variable_sets: &[Vec<Term>],
        depth: usize,
        current: Vec<Term>,
        result: &mut Vec<Term>,
    ) {
        if depth == variable_sets.len() {
            if current.len() >= 2 {
                // Create interaction term from the current combination
                let var_names: Vec<String> = current
                    .iter()
                    .filter_map(|term| {
                        if let TermKind::Variable(name) = &term.kind {
                            Some(name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                if var_names.len() == current.len() {
                    result.push(Term::interaction(var_names));
                }
            } else if current.len() == 1 {
                result.push(current[0].clone());
            }
            return;
        }

        for term in &variable_sets[depth] {
            let mut new_current = current.clone();
            new_current.push(term.clone());
            self.generate_interaction_combinations(variable_sets, depth + 1, new_current, result);
        }
    }

    /// Expand a function term with categorical arguments
    fn expand_function_categorical(
        &self,
        name: &str,
        args: &[Term],
        df: &DataFrame,
    ) -> FormulaResult<Vec<Term>> {
        // Special handling for known functions
        match name.to_lowercase().as_str() {
            "log" | "log10" | "log2" | "exp" | "sqrt" | "abs" | "scale" | "center"
            | "standardize" => {
                // Unary functions - expand the argument
                if args.len() != 1 {
                    return Err(FormulaError::function(
                        name,
                        format!("Expected 1 argument, got {}", args.len()),
                    ));
                }

                let expanded_args = args[0].expand_categorical(df)?;

                // If argument expands to multiple terms, apply function to each
                let mut result = Vec::new();
                for arg in expanded_args {
                    result.push(Term::function(name, vec![arg]));
                }

                Ok(result)
            }
            "poly" => {
                // Polynomial expansion
                if args.len() != 2 {
                    return Err(FormulaError::function(
                        "poly",
                        format!("Expected 2 arguments, got {}", args.len()),
                    ));
                }

                // First argument should be a variable, second is degree
                let expanded_var = args[0].expand_categorical(df)?;

                if expanded_var.len() > 1 {
                    // Categorical variable was expanded, poly() doesn't make sense
                    return Err(FormulaError::function(
                        "poly",
                        "poly() cannot be applied to categorical variables",
                    ));
                }

                // Not categorical, return as is
                Ok(vec![self.clone()])
            }
            "I" => {
                // Identity function - expand all arguments
                let mut expanded_args_sets: Vec<Vec<Term>> = Vec::new();

                for arg in args {
                    let expanded = arg.expand_categorical(df)?;
                    expanded_args_sets.push(expanded);
                }

                // Generate all combinations
                let mut result = Vec::new();
                self.generate_function_combinations(
                    name,
                    &expanded_args_sets,
                    0,
                    Vec::new(),
                    &mut result,
                );

                Ok(result)
            }
            _ => {
                // Unknown function - try to expand arguments
                self.expand_generic_function_categorical(name, args, df)
            }
        }
    }

    /// Expand a generic function with categorical arguments
    fn expand_generic_function_categorical(
        &self,
        name: &str,
        args: &[Term],
        df: &DataFrame,
    ) -> FormulaResult<Vec<Term>> {
        // Expand all arguments
        let mut expanded_args_sets: Vec<Vec<Term>> = Vec::new();

        for arg in args {
            let expanded = arg.expand_categorical(df)?;
            expanded_args_sets.push(expanded);
        }

        // Generate all combinations
        let mut result = Vec::new();
        self.generate_function_combinations(name, &expanded_args_sets, 0, Vec::new(), &mut result);

        Ok(result)
    }

    /// Generate all combinations for function expansion
    fn generate_function_combinations(
        &self,
        func_name: &str,
        arg_sets: &[Vec<Term>],
        depth: usize,
        current: Vec<Term>,
        result: &mut Vec<Term>,
    ) {
        if depth == arg_sets.len() {
            result.push(Term::function(func_name, current));
            return;
        }

        for term in &arg_sets[depth] {
            let mut new_current = current.clone();
            new_current.push(term.clone());
            self.generate_function_combinations(
                func_name,
                arg_sets,
                depth + 1,
                new_current,
                result,
            );
        }
    }

    /// Expand a parenthesized term
    fn expand_parenthesized_categorical(
        &self,
        terms: &[Term],
        df: &DataFrame,
    ) -> FormulaResult<Vec<Term>> {
        // Expand each term in the parentheses
        let mut expanded_terms: Vec<Term> = Vec::new();

        for term in terms {
            let expanded = term.expand_categorical(df)?;
            expanded_terms.extend(expanded);
        }

        // Create a new parenthesized term with expanded terms
        Ok(vec![Term::parenthesized(expanded_terms)])
    }

    /// Check if a variable is categorical in the DataFrame
    pub fn is_categorical_variable(&self, df: &DataFrame) -> bool {
        if let TermKind::Variable(name) = &self.kind {
            df.get_column(name)
                .map(|series| matches!(series, Series::Categorical(_, _)))
                .unwrap_or(false)
        } else {
            false
        }
    }

    /// Get the variable names referenced in this term
    pub fn variable_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        self.collect_variable_names(&mut names);
        names
    }

    /// Helper method to collect variable names
    fn collect_variable_names(&self, names: &mut Vec<String>) {
        match &self.kind {
            TermKind::Variable(name) => {
                names.push(name.clone());
            }
            TermKind::Interaction(interaction) => {
                for var in &interaction.variables {
                    names.push(var.clone());
                }
            }
            TermKind::Function { args, .. } => {
                for arg in args {
                    arg.collect_variable_names(names);
                }
            }
            TermKind::Parenthesized(terms) => {
                for term in terms {
                    term.collect_variable_names(names);
                }
            }
        }
    }
}

/// Interaction between variables
#[derive(Debug, Clone, PartialEq)]
pub struct Interaction {
    /// Variables involved in the interaction
    pub variables: Vec<String>,
    /// Order of interaction (2-way, 3-way, etc.)
    pub order: usize,
}

impl Interaction {
    /// Create a new interaction
    pub fn new(variables: Vec<String>) -> Self {
        let order = variables.len();
        Self { variables, order }
    }

    /// Convert interaction to design matrix
    pub fn to_matrix(&self, df: &DataFrame) -> FormulaResult<Matrix> {
        if self.order < 2 {
            return Err(FormulaError::InteractionError {
                message: "Interaction must involve at least 2 variables".to_string(),
                variables: self.variables.clone(),
            });
        }

        // Get matrices for each variable
        let mut var_matrices = Vec::new();
        for var in &self.variables {
            let term = Term::variable(var);
            let matrix = term.to_matrix(df)?;

            // Ensure is 2D
            if matrix.ndim() != 2 {
                return Err(FormulaError::InteractionError {
                    message: format!(
                        "Variable '{}' returns {}-dimensional matrix, expected 2D",
                        var,
                        matrix.ndim()
                    ),
                    variables: self.variables.clone(),
                });
            }

            var_matrices.push(matrix);
        }

        // Compute interaction matrix
        self.compute_interaction(&var_matrices)
    }

    /// Compute interaction matrix
    fn compute_interaction(&self, matrices: &[Matrix]) -> FormulaResult<Matrix> {
        if matrices.is_empty() {
            return Ok(Matrix::zeros((0, 0)));
        }

        let nrows = matrices[0].nrows();

        // For 2-way interaction
        if self.order == 2 && matrices.len() == 2 {
            let a = &matrices[0];
            let b = &matrices[1];

            // Ensure both are 2D
            if a.ndim() != 2 || b.ndim() != 2 {
                return Err(FormulaError::InteractionError {
                    message: format!(
                        "Matrices have dimensions {:?} and {:?}, expected 2D",
                        a.shape(),
                        b.shape()
                    ),
                    variables: self.variables.clone(),
                });
            }

            if a.ncols() == 1 && b.ncols() == 1 {
                // Single column interaction: element-wise multiplication
                let col_a = a.column(0);
                let col_b = b.column(0);
                let interaction: Array1<f64> = &col_a * &col_b;
                return interaction
                    .to_shape((nrows, 1))
                    .map(|m| m.into_owned())
                    .map_err(|e| FormulaError::EvaluationError {
                        message: format!("Failed to reshape interaction: {}", e),
                        context: None,
                    });
            } else {
                // Multi-column interaction: compute all column combinations
                let ncols_a = a.ncols();
                let ncols_b = b.ncols();
                let mut result = Array2::zeros((nrows, ncols_a * ncols_b));

                let mut col_idx = 0;
                for i in 0..ncols_a {
                    for j in 0..ncols_b {
                        let col_a = a.column(i);
                        let col_b = b.column(j);
                        let mut col_result = result.column_mut(col_idx);
                        col_result.assign(&(&col_a * &col_b));
                        col_idx += 1;
                    }
                }

                return Ok(result);
            }
        }

        // For higher-order interactions, recursively compute
        let mut result = matrices[0].clone();

        for matrix in matrices.iter().skip(1) {
            result = self.multiply_matrices(&result, matrix)?;
        }

        Ok(result)
    }

    /// Multiply two matrices element-wise, handling broadcasting
    fn multiply_matrices(&self, a: &Matrix, b: &Matrix) -> FormulaResult<Matrix> {
        if a.nrows() != b.nrows() {
            return Err(FormulaError::DimensionMismatch {
                message: "Matrices have different row counts".to_string(),
                expected: format!("{} rows (same as first matrix)", a.nrows()),
                actual: format!("{} rows", b.nrows()),
            });
        }

        let nrows = a.nrows();
        let ncols_a = a.ncols();
        let ncols_b = b.ncols();

        // Handle broadcasting
        if ncols_a == 1 && ncols_b > 1 {
            // Broadcast a to all columns of b
            let col_a = a.column(0);
            let mut result = Array2::zeros((nrows, ncols_b));

            for j in 0..ncols_b {
                let col_b = b.column(j);
                let mut col_result = col_a.into_owned();
                col_result *= &col_b;
                result.column_mut(j).assign(&col_result);
            }

            Ok(result)
        } else if ncols_b == 1 && ncols_a > 1 {
            // Broadcast b to all columns of a
            let col_b = b.column(0);
            let mut result = Array2::zeros((nrows, ncols_a));

            for j in 0..ncols_a {
                let col_a = a.column(j);
                let mut col_result = col_a.into_owned();
                col_result *= &col_b;
                result.column_mut(j).assign(&col_result);
            }

            Ok(result)
        } else if ncols_a == ncols_b {
            // Element-wise multiplication
            let mut result = a.clone();
            result *= b;
            Ok(result)
        } else {
            Err(FormulaError::DimensionMismatch {
                message: format!(
                    "Cannot multiply matrices with shapes {}x{} and {}x{}",
                    nrows, ncols_a, nrows, ncols_b
                ),
                expected: "compatible column dimensions".to_string(),
                actual: format!("{} and {} columns", ncols_a, ncols_b),
            })
        }
    }

    /// Get the variables in this interaction
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Get the order of the interaction
    pub fn order(&self) -> usize {
        self.order
    }

    /// Check if this is a 2-way interaction
    pub fn is_two_way(&self) -> bool {
        self.order == 2
    }

    /// Check if this is a 3-way or higher interaction
    pub fn is_higher_order(&self) -> bool {
        self.order >= 3
    }
}

impl fmt::Display for Interaction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, var) in self.variables.iter().enumerate() {
            if i > 0 {
                write!(f, ":")?;
            }
            write!(f, "{}", var)?;
        }
        Ok(())
    }
}

/// Helper trait for converting to design matrices
pub trait ToDesignMatrix {
    /// Convert to design matrix
    fn to_design_matrix(&self, df: &DataFrame) -> FormulaResult<Matrix>;
}

impl ToDesignMatrix for Term {
    fn to_design_matrix(&self, df: &DataFrame) -> FormulaResult<Matrix> {
        self.to_matrix(df)
    }
}

impl ToDesignMatrix for Interaction {
    fn to_design_matrix(&self, df: &DataFrame) -> FormulaResult<Matrix> {
        self.to_matrix(df)
    }
}
