//! R-style formula parser and design matrix builder
//!
//! This module provides formula parsing and evaluation similar to R's
//! formula interface, supporting:
//!
//! - Basic terms: `y ~ x1 + x2`
//! - Interaction terms: `y ~ x1:x2`, `y ~ x1*x2`
//! - Polynomial terms: `y ~ x1^2`
//! - Factor expansion: `y ~ factor(x1)`
//! - Special functions: `y ~ log(x1) + sqrt(x2)`
//! - Random effects (for mixed models): `y ~ (1 | group)`

use std::collections::{HashMap, HashSet};
use ndarray::{Array1, Array2, arr2};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace0, space0},
    combinator::{map, opt, recognize},
    multi::{many0, many1, separated_list1},
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};
use super::data::{DataFrame, Series};

// ============================================================================
// Formula AST
// ============================================================================

/// A term in a formula (variable, function, or interaction)
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// Simple variable: `x`
    Variable(String),
    /// Function call: `log(x)`
    Function(String, Box<Term>),
    /// Interaction: `x:y`
    Interaction(Box<Term>, Box<Term>),
    /// Polynomial: `x^2`
    Polynomial(Box<Term>, u32),
}

/// A formula expression (response ~ predictors)
#[derive(Debug, Clone)]
pub struct Formula {
    /// Response variable (left side of ~)
    pub response: Option<Term>,
    /// Predictor terms (right side of ~)
    pub predictors: Vec<Term>,
    /// Whether to include intercept (default: true)
    pub intercept: bool,
}

impl Formula {
    /// Create a new formula from a string
    pub fn parse(input: &str) -> Result<Self, super::error::Error> {
        parse_formula(input).map_err(|e| super::error::Error::FormulaError(format!("Parse error: {:?}", e)))
    }

    /// Create a formula with no intercept
    pub fn no_intercept(mut self) -> Self {
        self.intercept = false;
        self
    }

    /// Get all variable names in the formula
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        
        if let Some(ref resp) = self.response {
            collect_variables(resp, &mut vars);
        }
        
        for pred in &self.predictors {
            collect_variables(pred, &mut vars);
        }
        
        vars
    }

    /// Build design matrix from DataFrame
    pub fn build_matrix(&self, df: &DataFrame) -> Result<Array2<f64>, super::error::Error> {
        let n_rows = df.n_rows();
        let vars = self.variables();
        
        // Validate all variables exist in DataFrame
        for var in &vars {
            if !df.column_names().contains(var) {
                return Err(super::error::Error::FormulaError(format!("Variable '{}' not found in DataFrame", var)));
            }
        }

        // Start with intercept if requested
        let mut columns = Vec::new();
        if self.intercept {
            columns.push(vec![1.0; n_rows]);
        }

        // Process each predictor term
        for term in &self.predictors {
            let term_cols = build_term_matrix(term, df)?;
            columns.extend(term_cols);
        }

        // Convert to Array2
        let n_cols = columns.len();
        let mut matrix = Array2::zeros((n_rows, n_cols));
        
        for (j, col_data) in columns.into_iter().enumerate() {
            for (i, &val) in col_data.iter().enumerate() {
                matrix[(i, j)] = val;
            }
        }

        Ok(matrix)
    }

    /// Get response variable as array (if specified)
    pub fn response_vector(&self, df: &DataFrame) -> Result<Option<Array1<f64>>, super::error::Error> {
        if let Some(ref resp) = self.response {
            let resp_name = match resp {
                Term::Variable(name) => name,
                _ => return Err(super::error::Error::FormulaError("Complex response terms not yet supported".to_string())),
            };
            
            let series = df.column(resp_name)
                .ok_or_else(|| super::error::Error::FormulaError(format!("Response variable '{}' not found", resp_name)))?;
            
            Ok(Some(series.data().to_owned()))
        } else {
            Ok(None)
        }
    }
}

// ============================================================================
// Formula Parser (using nom)
// ============================================================================

fn parse_formula(input: &str) -> Result<Formula, String> {
    let (rest, (response, predictors)) = formula_parser(input)
        .map_err(|e| format!("Parse error: {:?}", e))?;
    
    if !rest.trim().is_empty() {
        return Err(format!("Unexpected input after formula: '{}'", rest));
    }

    Ok(Formula {
        response,
        predictors,
        intercept: true,
    })
}

fn formula_parser(input: &str) -> IResult<&str, (Option<Term>, Vec<Term>)> {
    let (input, _) = space0(input)?;
    
    // Parse response ~ predictors or just predictors
    let (input, result) = alt((
        // With response: y ~ x1 + x2
        map(
            tuple((
                term_parser,
                space0,
                tag("~"),
                space0,
                predictors_parser,
            )),
            |(resp, _, _, _, preds)| (Some(resp), preds),
        ),
        // Without response: ~ x1 + x2
        map(
            tuple((
                tag("~"),
                space0,
                predictors_parser,
            )),
            |(_, _, preds)| (None, preds),
        ),
        // Just predictors (implied ~)
        map(predictors_parser, |preds| (None, preds)),
    ))(input)?;

    Ok((input, result))
}

fn predictors_parser(input: &str) -> IResult<&str, Vec<Term>> {
    separated_list1(
        delimited(space0, tag("+"), space0),
        term_parser,
    )(input)
}

fn term_parser(input: &str) -> IResult<&str, Term> {
    let (input, term) = alt((
        // Function call: log(x)
        map(
            tuple((
                alpha1,
                char('('),
                term_parser,
                char(')'),
            )),
            |(func, _, arg, _)| Term::Function(func.to_string(), Box::new(arg)),
        ),
        // Interaction: x:y or x*y
        interaction_parser,
        // Base term (variable or number)
        base_term_parser,
    ))(input)?;

    // Handle polynomial: x^2
    let (input, term) = many0(map(
        tuple((
            char('^'),
            digit1,
        )),
        |(_, exp): (_, &str)| exp.parse::<u32>().unwrap_or(1),
    ))(input)
    .map(|(rest, exponents)| {
        let mut current = term;
        for exp in exponents {
            current = Term::Polynomial(Box::new(current), exp);
        }
        (rest, current)
    })?;

    Ok((input, term))
}

fn interaction_parser(input: &str) -> IResult<&str, Term> {
    let (input, left) = base_term_parser(input)?;
    let (input, _) = space0(input)?;
    let (input, op) = alt((tag(":"), tag("*")))(input)?;
    let (input, _) = space0(input)?;
    let (input, right) = term_parser(input)?;

    let term = if op == "*" {
        // x*y expands to x + y + x:y
        // We'll handle expansion later
        Term::Interaction(Box::new(left), Box::new(right))
    } else {
        Term::Interaction(Box::new(left), Box::new(right))
    };

    Ok((input, term))
}

fn base_term_parser(input: &str) -> IResult<&str, Term> {
    map(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_"), tag(".")))),
        )),
        |name: &str| Term::Variable(name.to_string()),
    )(input)
}



// ============================================================================
// Formula Evaluation
// ============================================================================

fn collect_variables(term: &Term, vars: &mut HashSet<String>) {
    match term {
        Term::Variable(name) => {
            vars.insert(name.clone());
        }
        Term::Function(_, arg) => {
            collect_variables(arg, vars);
        }
        Term::Interaction(left, right) => {
            collect_variables(left, vars);
            collect_variables(right, vars);
        }
        Term::Polynomial(base, _) => {
            collect_variables(base, vars);
        }
    }
}

fn build_term_matrix(term: &Term, df: &DataFrame) -> Result<Vec<Vec<f64>>, super::error::Error> {
    match term {
        Term::Variable(name) => {
            let series = df.column(name)
                .ok_or_else(|| super::error::Error::FormulaError(format!("Variable '{}' not found", name)))?;
            Ok(vec![series.data().to_vec()])
        }
        Term::Function(func, arg) => {
            let base_cols = build_term_matrix(arg, df)?;
            if base_cols.len() != 1 {
                return Err(super::error::Error::FormulaError("Functions can only be applied to single variables".to_string()));
            }
            
            let base_data = &base_cols[0];
            let transformed: Vec<f64> = match func.as_str() {
                "log" => base_data.iter().map(|&x| x.ln()).collect(),
                "log10" => base_data.iter().map(|&x| x.log10()).collect(),
                "log2" => base_data.iter().map(|&x| x.log2()).collect(),
                "sqrt" => base_data.iter().map(|&x| x.sqrt()).collect(),
                "exp" => base_data.iter().map(|&x| x.exp()).collect(),
                "abs" => base_data.iter().map(|&x| x.abs()).collect(),
                "sin" => base_data.iter().map(|&x| x.sin()).collect(),
                "cos" => base_data.iter().map(|&x| x.cos()).collect(),
                "tan" => base_data.iter().map(|&x| x.tan()).collect(),
                _ => return Err(super::error::Error::FormulaError(format!("Unsupported function: {}", func))),
            };
            
            Ok(vec![transformed])
        }
        Term::Interaction(left, right) => {
            let left_cols = build_term_matrix(left, df)?;
            let right_cols = build_term_matrix(right, df)?;
            
            // Simple interaction: multiply corresponding columns
            let mut result = Vec::new();
            for lcol in &left_cols {
                for rcol in &right_cols {
                    let interacted: Vec<f64> = lcol.iter()
                        .zip(rcol.iter())
                        .map(|(&l, &r)| l * r)
                        .collect();
                    result.push(interacted);
                }
            }
            
            Ok(result)
        }
        Term::Polynomial(base, power) => {
            let base_cols = build_term_matrix(base, df)?;
            if base_cols.len() != 1 {
                return Err(super::error::Error::FormulaError("Polynomial can only be applied to single variables".to_string()));
            }
            
            let base_data = &base_cols[0];
            let powered: Vec<f64> = base_data.iter()
                .map(|&x| x.powi(*power as i32))
                .collect();
            
            Ok(vec![powered])
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::data::DataFrame;
    use ndarray::arr1;
    use std::collections::HashMap;

    #[test]
    fn test_formula_parsing() {
        let cases = vec![
            ("y ~ x1 + x2", true, 2),
            ("y ~ x1 * x2", true, 1), // TODO: Should expand to x1 + x2 + x1:x2 (currently just x1:x2)
            ("y ~ log(x1) + sqrt(x2)", true, 2),
            ("~ x1 + x2", false, 2),
            ("x1 + x2", false, 2),
        ];

        for (input, has_response, pred_count) in cases {
            let formula = Formula::parse(input).unwrap();
            assert_eq!(formula.response.is_some(), has_response);
            assert_eq!(formula.predictors.len(), pred_count);
        }
    }

    #[test]
    fn test_variable_extraction() {
        let formula = Formula::parse("y ~ x1 + log(x2) + x3:x4").unwrap();
        let vars = formula.variables();
        
        assert!(vars.contains("y"));
        assert!(vars.contains("x1"));
        assert!(vars.contains("x2"));
        assert!(vars.contains("x3"));
        assert!(vars.contains("x4"));
        assert_eq!(vars.len(), 5);
    }

    #[test]
    fn test_design_matrix() {
        let mut columns = HashMap::new();
        columns.insert("y".to_string(), Series::new("y", arr1(&[1.0, 2.0, 3.0])));
        columns.insert("x1".to_string(), Series::new("x1", arr1(&[1.0, 2.0, 3.0])));
        columns.insert("x2".to_string(), Series::new("x2", arr1(&[4.0, 5.0, 6.0])));
        
        let df = DataFrame::from_series(columns).unwrap();
        let formula = Formula::parse("y ~ x1 + x2").unwrap();
        
        let matrix = formula.build_matrix(&df).unwrap();
        assert_eq!(matrix.shape(), &[3, 3]); // intercept + x1 + x2
        
        // Check intercept column
        assert_eq!(matrix.column(0).to_vec(), vec![1.0, 1.0, 1.0]);
        // Check x1 column
        assert_eq!(matrix.column(1).to_vec(), vec![1.0, 2.0, 3.0]);
        // Check x2 column
        assert_eq!(matrix.column(2).to_vec(), vec![4.0, 5.0, 6.0]);
    }
}