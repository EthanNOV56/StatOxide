//! Comprehensive test suite for the formula module
//!
//! This module provides extensive testing of formula parsing, evaluation,
//! and design matrix construction, including edge cases and error conditions.

use crate::data::{DataFrame, DataFrameBuilder, Series};
use crate::formula::error::FormulaError;
use crate::formula::term::{Interaction, Term, TermKind};
use crate::formula::*;
use ndarray::array;

#[test]
fn test_formula_parsing_basic_syntax() {
    // Test formula with response
    let formula = Formula::parse("y ~ x1 + x2").unwrap();
    assert_eq!(formula.response, Some("y".to_string()));
    assert_eq!(formula.terms.len(), 2);
    assert!(formula.has_intercept);
    assert_eq!(formula.to_string(), "y ~ x1 + x2");

    // Test formula without response
    let formula = Formula::parse("~ x1 + x2").unwrap();
    assert_eq!(formula.response, None);
    assert_eq!(formula.terms.len(), 2);
    assert!(formula.has_intercept);

    // Test formula with intercept removal
    let formula = Formula::parse("y ~ 0 + x1").unwrap();
    assert!(!formula.has_intercept);
    assert_eq!(formula.terms.len(), 1);
    assert_eq!(formula.to_string(), "y ~ 0 + x1");

    // Test formula with only intercept
    let formula = Formula::parse("y ~ 1").unwrap();
    assert!(formula.has_intercept);
    assert!(formula.terms.is_empty());
    assert_eq!(formula.to_string(), "y ~ 1");

    // Test formula with no intercept
    let formula = Formula::parse("y ~ 0").unwrap();
    assert!(!formula.has_intercept);
    assert!(formula.terms.is_empty());
    assert_eq!(formula.to_string(), "y ~ 0");

    // Test formula with whitespace variations
    let formula1 = Formula::parse("y~x1+x2").unwrap();
    let formula2 = Formula::parse("y ~ x1 + x2").unwrap();
    let formula3 = Formula::parse("  y  ~  x1  +  x2  ").unwrap();

    assert_eq!(formula1.response, formula2.response);
    assert_eq!(formula1.terms.len(), formula2.terms.len());
    assert_eq!(formula2.response, formula3.response);
    assert_eq!(formula2.terms.len(), formula3.terms.len());
}

#[test]
fn test_formula_parsing_interactions() {
    // Test two-way interaction
    let formula = Formula::parse("y ~ x1:x2").unwrap();
    assert_eq!(formula.terms.len(), 1);

    if let TermKind::Interaction(interaction) = &formula.terms[0].kind {
        assert_eq!(interaction.variables, vec!["x1", "x2"]);
        assert_eq!(interaction.order, 2);
    } else {
        panic!("Expected interaction term");
    }

    // Test three-way interaction
    let formula = Formula::parse("y ~ x1:x2:x3").unwrap();
    assert_eq!(formula.terms.len(), 1);

    if let TermKind::Interaction(interaction) = &formula.terms[0].kind {
        assert_eq!(interaction.variables, vec!["x1", "x2", "x3"]);
        assert_eq!(interaction.order, 3);
    } else {
        panic!("Expected interaction term");
    }

    // Test mixed terms with interactions
    let formula = Formula::parse("y ~ x1 + x2 + x1:x2").unwrap();
    assert_eq!(formula.terms.len(), 3);

    // Check first two are variables, third is interaction
    assert!(matches!(&formula.terms[0].kind, TermKind::Variable(_)));
    assert!(matches!(&formula.terms[1].kind, TermKind::Variable(_)));
    assert!(matches!(&formula.terms[2].kind, TermKind::Interaction(_)));
}

#[test]
fn test_formula_parsing_functions() {
    // Test simple function
    let formula = Formula::parse("y ~ log(x)").unwrap();
    assert_eq!(formula.terms.len(), 1);

    if let TermKind::Function { name, args } = &formula.terms[0].kind {
        assert_eq!(name, "log");
        assert_eq!(args.len(), 1);

        if let TermKind::Variable(arg_name) = &args[0].kind {
            assert_eq!(arg_name, "x");
        } else {
            panic!("Expected variable argument");
        }
    } else {
        panic!("Expected function term");
    }

    // Test function with multiple arguments
    let formula = Formula::parse("y ~ poly(x, 3)").unwrap();
    assert_eq!(formula.terms.len(), 1);

    if let TermKind::Function { name, args } = &formula.terms[0].kind {
        assert_eq!(name, "poly");
        assert_eq!(args.len(), 2);
    } else {
        panic!("Expected function term");
    }

    // Test nested function
    let formula = Formula::parse("y ~ log(sqrt(x))").unwrap();
    assert_eq!(formula.terms.len(), 1);

    if let TermKind::Function {
        name: outer_name,
        args: outer_args,
    } = &formula.terms[0].kind
    {
        assert_eq!(outer_name, "log");
        assert_eq!(outer_args.len(), 1);

        if let TermKind::Function {
            name: inner_name,
            args: inner_args,
        } = &outer_args[0].kind
        {
            assert_eq!(inner_name, "sqrt");
            assert_eq!(inner_args.len(), 1);
        } else {
            panic!("Expected nested function");
        }
    } else {
        panic!("Expected function term");
    }
}

#[test]
fn test_formula_parsing_parentheses() {
    // Test parentheses for grouping
    let formula = Formula::parse("y ~ (x1 + x2)").unwrap();
    assert_eq!(formula.terms.len(), 2);

    // Test parentheses in interaction
    let formula = Formula::parse("y ~ (x1):(x2)").unwrap();
    assert_eq!(formula.terms.len(), 1);

    if let TermKind::Interaction(interaction) = &formula.terms[0].kind {
        assert_eq!(interaction.variables, vec!["x1", "x2"]);
    } else {
        panic!("Expected interaction term");
    }

    // Test complex parentheses
    let formula = Formula::parse("y ~ (x1 + x2):x3").unwrap();
    assert_eq!(formula.terms.len(), 2);
    assert!(matches!(&formula.terms[0].kind, TermKind::Variable(_)));
    assert!(matches!(&formula.terms[1].kind, TermKind::Interaction(_)));
}

#[test]
fn test_formula_parsing_errors() {
    // Test empty formula
    assert!(Formula::parse("").is_err());
    assert!(Formula::parse("   ").is_err());

    // Test missing tilde
    assert!(Formula::parse("y x1 + x2").is_err());
    assert!(Formula::parse("y").is_err());

    // Test missing term after plus
    assert!(Formula::parse("y ~ x1 +").is_err());
    assert!(Formula::parse("y ~ + x1").is_err());

    // Test invalid intercept specification
    assert!(Formula::parse("y ~ 2 + x1").is_err());
    assert!(Formula::parse("y ~ -1 + x1").is_err());

    // Test missing closing parenthesis
    assert!(Formula::parse("y ~ log(x").is_err());
    assert!(Formula::parse("y ~ (x1 + x2").is_err());

    // Test missing comma in function
    assert!(Formula::parse("y ~ poly(x 2)").is_err());

    // Test invalid identifier
    assert!(Formula::parse("y ~ 123var").is_err());
    assert!(Formula::parse("y ~ _var").is_err());

    // Test trailing characters
    assert!(Formula::parse("y ~ x1 + x2 extra").is_err());
    assert!(Formula::parse("y ~ x1 + x2;").is_err());

    // Test empty function arguments
    assert!(Formula::parse("y ~ log()").is_err());

    // Test invalid interaction syntax
    assert!(Formula::parse("y ~ x1::x2").is_err());
    assert!(Formula::parse("y ~ x1:x2:").is_err());
}

#[test]
fn test_formula_variables() {
    let formula = Formula::parse("y ~ x1 + x2 + x1:x2 + log(x3) + sqrt(x4)").unwrap();
    let vars: std::collections::HashSet<&str> = formula.variables();

    assert_eq!(vars.len(), 5);
    assert!(vars.contains("y"));
    assert!(vars.contains("x1"));
    assert!(vars.contains("x2"));
    assert!(vars.contains("x3"));
    assert!(vars.contains("x4"));
}

#[test]
fn test_formula_display() {
    let formula = Formula::parse("y ~ x1 + x2 + x1:x2").unwrap();
    assert_eq!(formula.to_string(), "y ~ x1 + x2 + x1:x2");

    let formula = Formula::parse("~ x1 + log(x2)").unwrap();
    assert_eq!(formula.to_string(), "~ x1 + log(x2)");

    let formula = Formula::parse("y ~ 0 + x1").unwrap();
    assert_eq!(formula.to_string(), "y ~ 0 + x1");

    let formula = Formula::parse("y ~ 1").unwrap();
    assert_eq!(formula.to_string(), "y ~ 1");

    let formula = Formula::parse("y ~ 0").unwrap();
    assert_eq!(formula.to_string(), "y ~ 0");

    let formula = Formula::parse("y ~ (x1 + x2):x3").unwrap();
    assert_eq!(formula.to_string(), "y ~ x1 + x2 + x1:x3 + x2:x3");
}

#[test]
fn test_design_matrix_basic_numeric() {
    // Create test data
    let df = DataFrameBuilder::new()
        .with_column("y", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("x1", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("x2", Series::float(vec![4.0, 5.0, 6.0]))
        .unwrap()
        .build()
        .unwrap();

    // Test simple formula with intercept
    let formula = Formula::parse("y ~ x1 + x2").unwrap();
    let (design, response) = formula.design_matrix(&df).unwrap();

    assert_eq!(design.shape(), &[3, 3]); // intercept + x1 + x2
    assert!(response.is_some());

    // Check intercept column
    assert_eq!(design[[0, 0]], 1.0);
    assert_eq!(design[[1, 0]], 1.0);
    assert_eq!(design[[2, 0]], 1.0);

    // Check x1 column
    assert_eq!(design[[0, 1]], 1.0);
    assert_eq!(design[[1, 1]], 2.0);
    assert_eq!(design[[2, 1]], 3.0);

    // Check x2 column
    assert_eq!(design[[0, 2]], 4.0);
    assert_eq!(design[[1, 2]], 5.0);
    assert_eq!(design[[2, 2]], 6.0);

    // Check response
    let response_vec = response.unwrap();
    assert_eq!(response_vec.to_vec(), vec![1.0, 2.0, 3.0]);

    // Test without intercept
    let formula = Formula::parse("y ~ 0 + x1").unwrap();
    let (design, _) = formula.design_matrix(&df).unwrap();
    assert_eq!(design.shape(), &[3, 1]); // just x1
    assert_eq!(design[[0, 0]], 1.0);
    assert_eq!(design[[1, 0]], 2.0);
    assert_eq!(design[[2, 0]], 3.0);

    // Test with only intercept
    let formula = Formula::parse("y ~ 1").unwrap();
    let (design, _) = formula.design_matrix(&df).unwrap();
    assert_eq!(design.shape(), &[3, 1]); // just intercept
    assert_eq!(design[[0, 0]], 1.0);
    assert_eq!(design[[1, 0]], 1.0);
    assert_eq!(design[[2, 0]], 1.0);
}

#[test]
fn test_design_matrix_mixed_types() {
    // Create test data with mixed types
    let df = DataFrameBuilder::new()
        .with_column("y", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("x_float", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("x_int", Series::int(vec![4, 5, 6]))
        .unwrap()
        .with_column("x_bool", Series::bool(vec![true, false, true]))
        .unwrap()
        .build()
        .unwrap();

    let formula = Formula::parse("y ~ x_float + x_int + x_bool").unwrap();
    let (design, _) = formula.design_matrix(&df).unwrap();

    assert_eq!(design.shape(), &[3, 4]); // intercept + 3 variables

    // Check integer column converted to float
    assert_eq!(design[[0, 2]], 4.0);
    assert_eq!(design[[1, 2]], 5.0);
    assert_eq!(design[[2, 2]], 6.0);

    // Check boolean column converted to float
    assert_eq!(design[[0, 3]], 1.0);
    assert_eq!(design[[1, 3]], 0.0);
    assert_eq!(design[[2, 3]], 1.0);
}

#[test]
fn test_design_matrix_categorical() {
    // Create test data with categorical variable
    let df = DataFrameBuilder::new()
        .with_column("y", Series::float(vec![1.0, 2.0, 3.0, 4.0]))
        .unwrap()
        .with_column("group", Series::categorical(&["A", "B", "A", "B"]))
        .unwrap()
        .build()
        .unwrap();

    let formula = Formula::parse("y ~ group").unwrap();
    let (design, _) = formula.design_matrix(&df).unwrap();

    // Should have intercept + 2 dummy variables for 2 categories
    assert_eq!(design.shape(), &[4, 3]);

    // Check dummy variable encoding
    // Category A should be [1, 0], B should be [0, 1]
    // First sample: group A
    assert_eq!(design[[0, 1]], 1.0);
    assert_eq!(design[[0, 2]], 0.0);

    // Second sample: group B
    assert_eq!(design[[1, 1]], 0.0);
    assert_eq!(design[[1, 2]], 1.0);

    // Third sample: group A
    assert_eq!(design[[2, 1]], 1.0);
    assert_eq!(design[[2, 2]], 0.0);

    // Fourth sample: group B
    assert_eq!(design[[3, 1]], 0.0);
    assert_eq!(design[[3, 2]], 1.0);

    // Test with single category
    let df2 = DataFrameBuilder::new()
        .with_column("y", Series::float(vec![1.0, 2.0]))
        .unwrap()
        .with_column("group", Series::categorical(&["A", "A"]))
        .unwrap()
        .build()
        .unwrap();

    let formula2 = Formula::parse("y ~ group").unwrap();
    let (design2, _) = formula2.design_matrix(&df2).unwrap();

    // Single category creates a column of all ones (plus intercept)
    assert_eq!(design2.shape(), &[2, 2]);
    assert_eq!(design2[[0, 1]], 1.0);
    assert_eq!(design2[[1, 1]], 1.0);
}

#[test]
fn test_design_matrix_interactions() {
    let df = DataFrameBuilder::new()
        .with_column("x1", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("x2", Series::float(vec![4.0, 5.0, 6.0]))
        .unwrap()
        .build()
        .unwrap();

    // Test 2-way interaction
    let formula = Formula::parse("~ x1:x2").unwrap();
    let design = formula.build_design_matrix(&df).unwrap();

    // Interaction should be element-wise multiplication
    assert_eq!(design.shape(), &[3, 2]);
    assert_eq!(design[[0, 0]], 4.0); // 1.0 * 4.0
    assert_eq!(design[[1, 0]], 10.0); // 2.0 * 5.0
    assert_eq!(design[[2, 0]], 18.0); // 3.0 * 6.0

    // Test interaction with categorical variable
    let df2 = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("cat", Series::categorical(&["A", "B", "A"]))
        .unwrap()
        .build()
        .unwrap();

    let formula2 = Formula::parse("~ x:cat").unwrap();
    let design2 = formula2.build_design_matrix(&df2).unwrap();

    // x is 1 column, cat has 2 dummy columns, interaction has 2 columns
    assert_eq!(design2.shape(), &[3, 2]);
}

#[test]
fn test_design_matrix_functions() {
    let df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .build()
        .unwrap();

    // Test log transformation
    let formula = Formula::parse("~ log(x)").unwrap();
    let design = formula.build_design_matrix(&df).unwrap();

    assert_eq!(design.shape(), &[3, 2]);
    assert!((design[[0, 0]] - 0.0).abs() < 1e-10); // log(1) = 0
    assert!((design[[1, 0]] - 0.693147).abs() < 1e-5); // log(2) ≈ 0.693147

    // Test sqrt transformation
    let formula = Formula::parse("~ sqrt(x)").unwrap();
    let design = formula.build_design_matrix(&df).unwrap();

    assert!((design[[0, 0]] - 1.0).abs() < 1e-10); // sqrt(1) = 1
    assert!((design[[1, 0]] - 1.414214).abs() < 1e-5); // sqrt(2) ≈ 1.414214

    // Test polynomial expansion
    let formula = Formula::parse("~ poly(x, 3)").unwrap();
    let design = formula.build_design_matrix(&df).unwrap();

    assert_eq!(design.shape(), &[3, 3]);

    // x^1
    assert_eq!(design[[0, 0]], 1.0);
    assert_eq!(design[[1, 0]], 2.0);
    assert_eq!(design[[2, 0]], 3.0);

    // x^2
    assert_eq!(design[[0, 1]], 1.0);
    assert_eq!(design[[1, 1]], 4.0);
    assert_eq!(design[[2, 1]], 9.0);

    // x^3
    assert_eq!(design[[0, 2]], 1.0);
    assert_eq!(design[[1, 2]], 8.0);
    assert_eq!(design[[2, 2]], 27.0);

    // Test scale function
    let formula = Formula::parse("~ scale(x)").unwrap();
    let design = formula.build_design_matrix(&df).unwrap();

    // Scaled data should have mean 0 and std 1
    let scaled_data: Vec<f64> = design.column(0).to_vec();
    let mean = scaled_data.iter().sum::<f64>() / scaled_data.len() as f64;
    let variance = scaled_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
        / (scaled_data.len() as f64 - 1.0);

    assert!(mean.abs() < 1e-10);
    assert!((variance.sqrt() - 1.0).abs() < 1e-10);
}

#[test]
fn test_design_matrix_complex_formulas() {
    let df = DataFrameBuilder::new()
        .with_column("y", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("x1", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("x2", Series::float(vec![4.0, 5.0, 6.0]))
        .unwrap()
        .with_column("cat", Series::categorical(&["A", "B", "A"]))
        .unwrap()
        .build()
        .unwrap();

    // Test complex formula with all features
    let formula = Formula::parse("y ~ x1 + x2 + cat + x1:x2 + log(x1)").unwrap();
    let (design, _) = formula.design_matrix(&df).unwrap();

    // intercept + x1 + x2 + 2 dummies + interaction + log(x1) = 7 columns
    assert_eq!(design.shape(), &[3, 7]);

    // Test even more complex formula
    let formula = Formula::parse("y ~ x1 + x2 + cat + x1:x2 + x1:cat + poly(x2, 2)").unwrap();
    let (design, _) = formula.design_matrix(&df).unwrap();
    // intercept + x1 + x2 + 2 dummies + x1:x2 + 2*x1:cat + 2*poly = 9 columns
    assert_eq!(design.shape(), &[3, 9]);
}

#[test]
fn test_design_matrix_errors() {
    // Create test data
    let df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column(
            "y",
            Series::string(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .unwrap()
        .build()
        .unwrap();

    // Test missing variable
    let formula = Formula::parse("z ~ x").unwrap();
    assert!(formula.design_matrix(&df).is_err());

    if let Err(FormulaError::VariableNotFound { variable, .. }) = formula.design_matrix(&df) {
        assert_eq!(variable, "z");
    } else {
        panic!("Expected VariableNotFound error");
    }

    // Test non-numeric response
    let formula = Formula::parse("y ~ x").unwrap();
    assert!(formula.design_matrix(&df).is_err());

    if let Err(FormulaError::TypeMismatch { variable, .. }) = formula.design_matrix(&df) {
        assert_eq!(variable, "y");
    } else {
        panic!("Expected TypeMismatch error");
    }

    // Test invalid function argument
    let formula = Formula::parse("~ log(y)").unwrap();
    assert!(formula.build_design_matrix(&df).is_err());

    // Test log of non-positive values
    let df2 = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![-1.0, 0.0, 1.0]))
        .unwrap()
        .build()
        .unwrap();

    let formula2 = Formula::parse("~ log(x)").unwrap();
    assert!(formula2.build_design_matrix(&df2).is_err());

    if let Err(FormulaError::NumericalError { message, .. }) = formula2.build_design_matrix(&df2) {
        assert!(message.contains("requires positive values"));
    } else {
        panic!("Expected NumericalError");
    }

    // Test sqrt of negative values
    let formula3 = Formula::parse("~ sqrt(x)").unwrap();
    assert!(formula3.build_design_matrix(&df2).is_err());

    if let Err(FormulaError::NumericalError { message, .. }) = formula3.build_design_matrix(&df2) {
        assert!(message.contains("requires non-negative values"));
    } else {
        panic!("Expected NumericalError");
    }
}

#[test]
fn test_term_creation() {
    // Test variable term
    let var_term = Term::variable("x");
    assert!(var_term.is_variable());
    assert_eq!(var_term.as_variable(), Some("x"));
    assert!(!var_term.expanded);
    assert_eq!(var_term.to_string(), "x");

    // Test interaction term
    let interaction = Term::interaction(vec!["x".to_string(), "y".to_string()]);
    assert!(interaction.is_interaction());

    if let TermKind::Interaction(interaction) = &interaction.kind {
        assert_eq!(interaction.variables, vec!["x", "y"]);
        assert_eq!(interaction.order, 2);
    } else {
        panic!("Expected interaction");
    }

    // Test function term
    let func_term = Term::function("log", vec![Term::variable("x")]);
    assert!(func_term.is_function());
    assert_eq!(func_term.to_string(), "log(x)");
}

#[test]
fn test_interaction_structure() {
    // Test 2-way interaction
    let interaction = Interaction::new(vec!["x".to_string(), "y".to_string()]);
    assert_eq!(interaction.variables(), &["x", "y"]);
    assert_eq!(interaction.order(), 2);
    assert!(interaction.is_two_way());
    assert!(!interaction.is_higher_order());
    assert_eq!(interaction.to_string(), "x:y");

    // Test 3-way interaction
    let interaction = Interaction::new(vec!["x".to_string(), "y".to_string(), "z".to_string()]);
    assert_eq!(interaction.variables(), &["x", "y", "z"]);
    assert_eq!(interaction.order(), 3);
    assert!(!interaction.is_two_way());
    assert!(interaction.is_higher_order());
    assert_eq!(interaction.to_string(), "x:y:z");
}

#[test]
fn test_matrix_operations() {
    let m1 = array![[1.0, 2.0], [3.0, 4.0]];
    let m2 = array![[5.0, 6.0], [7.0, 8.0]];

    // Test hstack_matrices
    let hstacked = crate::formula::utils::hstack_matrices(&[m1.clone(), m2.clone()]).unwrap();
    assert_eq!(hstacked.shape(), &[2, 4]);
    assert_eq!(hstacked[(0, 0)], 1.0);
    assert_eq!(hstacked[(0, 3)], 6.0);

    // Test vstack_matrices
    let vstacked = crate::formula::utils::vstack_matrices(&[m1.clone(), m2.clone()]).unwrap();
    assert_eq!(vstacked.shape(), &[4, 2]);
    assert_eq!(vstacked[(0, 0)], 1.0);
    assert_eq!(vstacked[(3, 1)], 8.0);

    // Test safe_concatenate
    let hconcat =
        crate::formula::utils::safe_concatenate(ndarray::Axis(1), &[m1.clone(), m2.clone()])
            .unwrap();
    assert_eq!(hconcat.shape(), &[2, 4]);

    // Test safe_stack
    // let stacked = crate::formula::utils::safe_stack(ndarray::Axis(0), &[m1, m2]).unwrap();
    // assert_eq!(stacked.shape(), &[4, 2]);
}

#[test]
fn test_edge_cases() {
    // Test empty DataFrame
    let empty_df = DataFrame::new();
    let formula = Formula::parse("~ 1").unwrap();
    let design = formula.build_design_matrix(&empty_df).unwrap();
    assert_eq!(design.shape(), &[0, 1]); // 0 rows, 1 column (intercept)

    // Test formula with no terms
    let df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 2.0]))
        .unwrap()
        .build()
        .unwrap();

    let formula = Formula::parse("~ 0").unwrap();
    let design = formula.build_design_matrix(&df).unwrap();
    assert_eq!(design.shape(), &[2, 0]); // 2 rows, 0 columns

    // Test single sample
    let single_df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0]))
        .unwrap()
        .build()
        .unwrap();

    let formula = Formula::parse("~ x").unwrap();
    let design = formula.build_design_matrix(&single_df).unwrap();
    assert_eq!(design.shape(), &[1, 2]); // intercept + x

    // Test constant column
    let const_df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 1.0, 1.0]))
        .unwrap()
        .build()
        .unwrap();

    let formula = Formula::parse("~ scale(x)").unwrap();
    assert!(formula.build_design_matrix(&const_df).is_err());
}

#[test]
fn test_performance_smoke_test() {
    // Create larger dataset for performance smoke test
    let n_samples = 1000;
    let x_values: Vec<f64> = (0..n_samples).map(|i| i as f64 + 1.0).collect();
    let y_values: Vec<f64> = (0..n_samples).map(|i| (i as f64) * 2.0).collect();
    let categories: Vec<&str> = (0..n_samples)
        .map(|i| if i % 2 == 0 { "A" } else { "B" })
        .collect();

    let df = DataFrameBuilder::new()
        .with_column("y", Series::float(y_values))
        .unwrap()
        .with_column("x1", Series::float(x_values.clone()))
        .unwrap()
        .with_column("x2", Series::float(x_values))
        .unwrap()
        .with_column("cat", Series::categorical(&categories))
        .unwrap()
        .build()
        .unwrap();

    // Test complex formula on larger dataset
    let formula = Formula::parse("y ~ x1 + x2 + cat + x1:x2 + log(x1) + poly(x2, 3)").unwrap();

    // This should complete without panicking
    let (design, response) = formula.design_matrix(&df).unwrap();

    assert_eq!(design.nrows(), n_samples);
    assert!(response.is_some());
    assert_eq!(response.unwrap().len(), n_samples);
}

#[cfg(feature = "proptest")]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_formula_parsing_roundtrip(formula_str in r"[a-z]{1,3} ~ ([a-z]{1,3}(:[a-z]{1,3})*(\+[a-z]{1,3}(:[a-z]{1,3})*)*)?") {
            if let Ok(formula) = Formula::parse(&formula_str) {
                // The parsed formula should display as valid
                let display = formula.to_string();
                // Re-parse should succeed
                assert!(Formula::parse(&display).is_ok());
            }
        }
    }
}

// ==================== 集成测试 ====================

#[test]
fn test_formula_integration() {
    // Create a realistic dataset
    let df = DataFrameBuilder::new()
        .with_column("mpg", Series::float(vec![21.0, 22.8, 21.4, 18.7, 18.1]))
        .unwrap()
        .with_column("cyl", Series::categorical(&["4", "6", "4", "8", "6"]))
        .unwrap()
        .with_column(
            "disp",
            Series::float(vec![160.0, 108.0, 258.0, 360.0, 225.0]),
        )
        .unwrap()
        .with_column("hp", Series::float(vec![110.0, 93.0, 110.0, 175.0, 105.0]))
        .unwrap()
        .build()
        .unwrap();

    // Test realistic formula similar to R's formula syntax
    let formula = Formula::parse("mpg ~ cyl + disp + hp + cyl:disp + log(disp)").unwrap();

    // Should parse successfully
    assert!(formula.has_response());
    assert_eq!(formula.response, Some("mpg".to_string()));

    // Should build design matrix
    let (design, response) = formula.design_matrix(&df).unwrap();

    // Check dimensions
    assert_eq!(design.nrows(), 5);
    assert!(response.is_some());
    assert_eq!(response.unwrap().len(), 5);

    // Check that design matrix has no NaN or Inf values
    for &value in design.iter() {
        assert!(
            value.is_finite(),
            "Design matrix contains non-finite value: {}",
            value
        );
    }

    // Test prediction formula
    let pred_formula = Formula::parse("~ cyl + disp + hp").unwrap();
    let pred_design = pred_formula.build_design_matrix(&df).unwrap();

    // Should have intercept + 2 dummy columns + disp + hp = 5 columns
    assert_eq!(pred_design.shape(), &[5, 5]);
}

#[test]
fn test_formula_serialization_roundtrip() {
    // Test that formula can be parsed and displayed in a consistent way
    let test_cases = vec![
        "y ~ x1 + x2",
        "y ~ 0 + x1",
        "y ~ 1",
        "y ~ 0",
        "y ~ x1:x2",
        "y ~ x1 + x2 + x1:x2",
        "y ~ log(x1)",
        "y ~ poly(x1, 3)",
        "y ~ x1 + scale(x2)",
        "~ x1 + x2",
    ];

    for &formula_str in &test_cases {
        let formula = Formula::parse(formula_str).unwrap();
        let displayed = formula.to_string();

        // Re-parse the displayed formula
        let re_parsed = Formula::parse(&displayed).unwrap();

        // The re-parsed formula should be equivalent
        assert_eq!(formula.response, re_parsed.response);
        assert_eq!(formula.has_intercept, re_parsed.has_intercept);
        assert_eq!(formula.terms.len(), re_parsed.terms.len());
    }
}

#[test]
fn test_error_messages() {
    // Test that error messages are informative
    let df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .build()
        .unwrap();

    // Missing variable
    let formula = Formula::parse("y ~ x").unwrap();
    let err = formula.design_matrix(&df).unwrap_err();
    assert!(err.to_string().contains("Variable 'y' not found"));

    // Invalid function argument
    // let formula = Formula::parse("~ log(-1)").unwrap();
    // let err = formula.build_design_matrix(&df).unwrap_err();
    // assert!(err.to_string().contains("requires positive values"));

    // Syntax error
    let err = Formula::parse("y ~ x1 +").unwrap_err();
    assert!(err.to_string().contains("Syntax error"));
}

#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_formula_parsing(b: &mut Bencher) {
        let formula_str =
            "y ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + log(x1) + sqrt(x2) + poly(x3, 3)";
        b.iter(|| {
            Formula::parse(formula_str).unwrap();
        });
    }

    #[bench]
    fn bench_design_matrix_large(b: &mut Bencher) {
        let n_samples = 10000;
        let x: Vec<f64> = (0..n_samples).map(|i| i as f64).collect();

        let df = DataFrameBuilder::new()
            .with_column("y", Series::float(x.clone()))
            .unwrap()
            .with_column("x1", Series::float(x.clone()))
            .unwrap()
            .with_column("x2", Series::float(x))
            .unwrap()
            .build()
            .unwrap();

        let formula = Formula::parse("y ~ x1 + x2 + x1:x2").unwrap();

        b.iter(|| {
            formula.design_matrix(&df).unwrap();
        });
    }
}
