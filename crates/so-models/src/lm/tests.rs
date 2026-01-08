//! Tests for linear regression models
//!
//! This module contains comprehensive tests for the linear regression
//! implementation, including OLS, WLS, ridge, lasso, and elastic net.

use approx::assert_abs_diff_eq;
use ndarray::array;
use rand_distr::Distribution;

use crate::{
    base::ModelError,
    lm::{LinearConfig, LinearModel, LinearRegression, StandardErrorType, lm},
};
use so_core::data::{DataFrame, DataFrameBuilder, Series};

// ==================== Test Fixtures ====================

/// Simple linear relationship: y = 2x + 1
fn simple_linear_data() -> DataFrame {
    DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap()
        .with_column("y", Series::float(vec![3.0, 5.0, 7.0, 9.0, 11.0]))
        .unwrap()
        .build()
        .unwrap()
}

/// Multiple regression: y = 1 + 2x1 + 3x2
fn multiple_regression_data() -> DataFrame {
    DataFrameBuilder::new()
        .with_column("x1", Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap()
        .with_column("x2", Series::float(vec![2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap()
        .with_column("y", Series::float(vec![9.0, 13.0, 17.0, 21.0, 25.0]))
        .unwrap()
        .build()
        .unwrap()
}

/// No intercept: y = 2x
fn no_intercept_data() -> DataFrame {
    DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap()
        .with_column("y", Series::float(vec![2.0, 4.0, 6.0, 8.0, 10.0]))
        .unwrap()
        .build()
        .unwrap()
}

/// Realistic dataset with some noise
fn noisy_data() -> DataFrame {
    let n = 100;
    let mut rng = rand::rng();

    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let mut y = Vec::new();

    for i in 0..n {
        let x1_val = i as f64 * 0.1;
        let x2_val = (i as f64).sin();
        let noise = rand_distr::Normal::new(0.0, 0.1).unwrap().sample(&mut rng);
        let y_val = 1.0 + 2.0 * x1_val + 3.0 * x2_val + noise;

        x1.push(x1_val);
        x2.push(x2_val);
        y.push(y_val);
    }

    DataFrameBuilder::new()
        .with_column("x1", Series::float(x1))
        .unwrap()
        .with_column("x2", Series::float(x2))
        .unwrap()
        .with_column("y", Series::float(y))
        .unwrap()
        .build()
        .unwrap()
}

/// Categorical variable data
fn categorical_data() -> DataFrame {
    DataFrameBuilder::new()
        .with_column(
            "group",
            Series::categorical(&["A", "A", "B", "B", "C", "C"]),
        )
        .unwrap()
        .with_column("x", Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap()
        .with_column("y", Series::float(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]))
        .unwrap()
        .build()
        .unwrap()
}

// ==================== Basic Tests ====================

#[test]
fn test_linear_regression_basic_fit() {
    let df = simple_linear_data();

    let model = LinearRegression::new("y ~ x")
        .unwrap()
        .data(&df)
        .fit()
        .unwrap();

    let coeffs = model.coefficients().unwrap();
    assert_eq!(coeffs.len(), 2);

    // Should be close to y = 1 + 2x
    assert_abs_diff_eq!(coeffs[0], 1.0, epsilon = 1e-10); // Intercept
    assert_abs_diff_eq!(coeffs[1], 2.0, epsilon = 1e-10); // Slope

    // Check fitted values
    let fitted = model.fitted_values().unwrap();
    let expected = array![3.0, 5.0, 7.0, 9.0, 11.0];
    assert_abs_diff_eq!(fitted, &expected, epsilon = 1e-10);

    // Check residuals
    let residuals = model.residuals().unwrap();
    assert_abs_diff_eq!(residuals.sum(), 0.0, epsilon = 1e-10);

    // Check R-squared (should be 1.0 for perfect fit)
    let summary = model.summary().unwrap();
    let r2 = summary.model_statistics.r_squared.unwrap();
    assert_abs_diff_eq!(r2, 1.0, epsilon = 1e-10);
}

#[test]
fn test_linear_regression_no_intercept() {
    let df = no_intercept_data();

    let model = LinearRegression::new("y ~ 0 + x")
        .unwrap()
        .data(&df)
        .fit()
        .unwrap();

    let coeffs = model.coefficients().unwrap();
    assert_eq!(coeffs.len(), 1); // Only slope, no intercept

    // Should be close to y = 2x
    assert_abs_diff_eq!(coeffs[0], 2.0, epsilon = 1e-10);

    // R-squared should still be 1.0
    let summary = model.summary().unwrap();
    assert_abs_diff_eq!(
        summary.model_statistics.r_squared.unwrap(),
        1.0,
        epsilon = 1e-10
    );
}

#[test]
fn test_linear_regression_multiple_predictors() {
    let df = multiple_regression_data();

    let model = LinearRegression::new("y ~ x1 + x2")
        .unwrap()
        .data(&df)
        .fit()
        .unwrap();

    let coeffs = model.coefficients().unwrap();
    assert_eq!(coeffs.len(), 3); // Intercept + 2 predictors

    // Should be close to y = 1 + 2x1 + 3x2
    assert_abs_diff_eq!(coeffs[0], 1.0, epsilon = 1e-10); // Intercept
    assert_abs_diff_eq!(coeffs[1], 2.0, epsilon = 1e-10); // x1 coefficient
    assert_abs_diff_eq!(coeffs[2], 3.0, epsilon = 1e-10); // x2 coefficient

    // Check R-squared
    let summary = model.summary().unwrap();
    assert_abs_diff_eq!(
        summary.model_statistics.r_squared.unwrap(),
        1.0,
        epsilon = 1e-10
    );
}

#[test]
fn test_linear_regression_interaction() {
    // Create data with interaction: y = 1 + 2x1 + 3x2 + 4x1:x2
    let df = DataFrameBuilder::new()
        .with_column("x1", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("x2", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("y", Series::float(vec![10.0, 21.0, 34.0]))
        .unwrap()
        .build()
        .unwrap();

    let model = LinearRegression::new("y ~ x1 + x2 + x1:x2")
        .unwrap()
        .data(&df)
        .fit()
        .unwrap();

    let coeffs = model.coefficients().unwrap();
    assert_eq!(coeffs.len(), 4); // Intercept + 2 main effects + interaction

    // Coefficients should match the generating equation
    assert_abs_diff_eq!(coeffs[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(coeffs[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(coeffs[2], 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(coeffs[3], 4.0, epsilon = 1e-10);
}

#[test]
fn test_linear_regression_with_categorical() {
    let df = categorical_data();

    let model = LinearRegression::new("y ~ group + x")
        .unwrap()
        .data(&df)
        .fit()
        .unwrap();

    // Should have intercept + 2 dummy variables (3 groups -> 2 dummies) + x
    let coeffs = model.coefficients().unwrap();
    assert!(coeffs.len() >= 4);

    // Check that we can get a summary
    let summary = model.summary().unwrap();
    assert!(!summary.coefficients.is_empty());

    // R-squared should be high (perfect linear relationship within groups)
    let r2 = summary.model_statistics.r_squared.unwrap();
    assert!(r2 > 0.9);
}

// ==================== Error Handling Tests ====================

#[test]
fn test_linear_regression_insufficient_data() {
    let df = DataFrameBuilder::new()
        .with_column("y", Series::float(vec![1.0, 2.0]))
        .unwrap()
        .with_column("x1", Series::float(vec![1.0, 2.0]))
        .unwrap()
        .with_column("x2", Series::float(vec![3.0, 4.0]))
        .unwrap()
        .with_column("x3", Series::float(vec![5.0, 6.0]))
        .unwrap()
        .build()
        .unwrap();

    let result = LinearRegression::new("y ~ x1 + x2 + x3")
        .unwrap()
        .data(&df)
        .fit();

    assert!(result.is_err());

    match result.unwrap_err() {
        ModelError::InsufficientData {
            n_samples,
            n_predictors,
        } => {
            assert_eq!(n_samples, 2);
            assert_eq!(n_predictors, 4); // Intercept + 3 predictors
        }
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_linear_regression_missing_variable() {
    let df = simple_linear_data();

    let result = LinearRegression::new("y ~ z").unwrap().data(&df).fit();

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ModelError::Formula(_)));
}

#[test]
fn test_linear_regression_no_data() {
    let model = LinearRegression::new("y ~ x").unwrap();
    let result = model.fit();

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::Custom { message } if message.contains("No data provided")
    ));
}

#[test]
fn test_linear_regression_no_response() {
    let df = simple_linear_data();

    let result = LinearRegression::new("~ x").unwrap().data(&df).fit();

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ModelError::Custom { message } if message.contains("No response variable")
    ));
}

// ==================== Prediction Tests ====================

#[test]
fn test_linear_regression_prediction() {
    let train_df = simple_linear_data();
    let test_df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![6.0, 7.0]))
        .unwrap()
        .build()
        .unwrap();

    let model = LinearRegression::new("y ~ x")
        .unwrap()
        .data(&train_df)
        .fit()
        .unwrap();

    let predictions = model.predict(&test_df).unwrap();
    assert_eq!(predictions.len(), 2);

    // y = 1 + 2x
    assert_abs_diff_eq!(predictions[0], 13.0, epsilon = 1e-10); // 1 + 2 * 6
    assert_abs_diff_eq!(predictions[1], 15.0, epsilon = 1e-10); // 1 + 2 * 7
}

#[test]
fn test_linear_regression_predict_with_missing_vars() {
    let train_df = multiple_regression_data();
    let test_df = DataFrameBuilder::new()
        .with_column("x1", Series::float(vec![6.0, 7.0]))
        .unwrap()
        .build()
        .unwrap();

    let model = LinearRegression::new("y ~ x1 + x2")
        .unwrap()
        .data(&train_df)
        .fit()
        .unwrap();

    // Should fail because x2 is missing
    let result = model.predict(&test_df);
    assert!(result.is_err());
}

// ==================== Statistics Tests ====================

#[test]
fn test_linear_regression_statistics() {
    let df = noisy_data();

    let model = LinearRegression::new("y ~ x1 + x2")
        .unwrap()
        .data(&df)
        .fit()
        .unwrap();

    let summary = model.summary().unwrap();

    // Check that we have all expected statistics
    assert!(summary.model_statistics.r_squared.is_some());
    assert!(summary.model_statistics.adj_r_squared.is_some());
    assert!(summary.model_statistics.f_statistic.is_some());
    assert!(summary.model_statistics.f_p_value.is_some());
    assert!(summary.model_statistics.log_likelihood.is_some());
    assert!(summary.model_statistics.aic.is_some());
    assert!(summary.model_statistics.bic.is_some());
    assert!(summary.model_statistics.residual_std_error.is_some());
    assert!(summary.model_statistics.df_residual.is_some());
    assert!(summary.model_statistics.df_model.is_some());

    // R-squared should be between 0 and 1
    let r2 = summary.model_statistics.r_squared.unwrap();
    assert!(r2 >= 0.0 && r2 <= 1.0);

    // Adjusted R-squared should be <= R-squared
    let adj_r2 = summary.model_statistics.adj_r_squared.unwrap();
    assert!(adj_r2 <= r2);

    // F-statistic should be positive
    let f_stat = summary.model_statistics.f_statistic.unwrap();
    assert!(f_stat >= 0.0);

    // p-value should be between 0 and 1
    let p_val = summary.model_statistics.f_p_value.unwrap();
    assert!(p_val >= 0.0 && p_val <= 1.0);
}

// ==================== Configuration Tests ====================

#[test]
fn test_linear_regression_robust_standard_errors() {
    let df = simple_linear_data();

    // Test with different standard error types
    let se_types = [
        StandardErrorType::Standard,
        StandardErrorType::HC0,
        StandardErrorType::HC1,
        StandardErrorType::HC2,
        StandardErrorType::HC3,
    ];

    for se_type in se_types {
        let model = LinearRegression::new("y ~ x")
            .unwrap()
            .data(&df)
            .robust(se_type)
            .fit()
            .unwrap();

        let std_errors = model.standard_errors().unwrap();
        assert_eq!(std_errors.len(), 2);
        assert!(std_errors[0] > 0.0);
        assert!(std_errors[1] > 0.0);
    }
}

#[test]
fn test_linear_regression_with_scaling() {
    let df = multiple_regression_data();

    let config = LinearConfig {
        center_predictors: true,
        scale_predictors: true,
        ..LinearConfig::default()
    };

    let model = LinearRegression::new("y ~ x1 + x2")
        .unwrap()
        .data(&df)
        .config(config)
        .fit()
        .unwrap();

    // Model should still fit successfully
    let coeffs = model.coefficients().unwrap();
    assert_eq!(coeffs.len(), 3);

    // Check that predictions are correct
    let test_df = DataFrameBuilder::new()
        .with_column("x1", Series::float(vec![1.0]))
        .unwrap()
        .with_column("x2", Series::float(vec![2.0]))
        .unwrap()
        .build()
        .unwrap();

    let pred = model.predict(&test_df).unwrap();
    assert_abs_diff_eq!(pred[0], 9.0, epsilon = 1e-10); // 1 + 2 * 1 + 3 * 2
}

// ==================== Convenience Function Tests ====================

#[test]
fn test_lm_convenience_function() {
    let df = simple_linear_data();

    // Using the convenience function
    let model = lm("y ~ x", &df).unwrap();

    let coeffs = model.coefficients().unwrap();
    assert_eq!(coeffs.len(), 2);
    assert_abs_diff_eq!(coeffs[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(coeffs[1], 2.0, epsilon = 1e-10);
}

// ==================== Serialization Tests ====================

#[test]
fn test_linear_regression_serialization() {
    let df = simple_linear_data();

    let model = LinearRegression::new("y ~ x")
        .unwrap()
        .data(&df)
        .fit()
        .unwrap();

    // Serialize to JSON
    // let json = serde_json::to_string(&model).unwrap();
    // assert!(!json.is_empty());

    // Deserialize
    // let deserialized: LinearRegression = serde_json::from_str(&json).unwrap();
    unimplemented!("Ser&Deser");

    // Check that coefficients are the same
    // let original_coeffs = model.coefficients().unwrap();
    // let deserialized_coeffs = deserialized.coefficients().unwrap();
    // assert_abs_diff_eq!(original_coeffs, deserialized_coeffs, epsilon = 1e-10);
}

// ==================== Property-Based Tests ====================

#[cfg(feature = "proptest")]
mod proptest_tests {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_linear_regression_r_squared_bounds(
            data in vec(vec(-100.0f64..100.0, 2..20), 5..50)
        ) {
            // Generate random data
            let n = data.len();
            let mut x = Vec::new();
            let mut y = Vec::new();

            for row in &data {
                if row.len() >= 2 {
                    x.push(row[0]);
                    y.push(row[1]);
                }
            }

            if x.len() >= 5 && y.len() >= 5 {
                let df = DataFrameBuilder::new()
                    .with_column("x", Series::float(x))
                    .unwrap()
                    .with_column("y", Series::float(y))
                    .unwrap()
                    .build()
                    .unwrap();

                let result = LinearRegression::new("y ~ x")
                    .unwrap()
                    .data(&df)
                    .fit();

                if let Ok(model) = result {
                    let summary = model.summary().unwrap();
                    if let Some(r2) = summary.model_statistics.r_squared {
                        prop_assert!(r2 >= 0.0 && r2 <= 1.0);
                    }
                }
            }
        }
    }
}

// ==================== Display Tests ====================

#[test]
fn test_linear_regression_display() {
    let df = simple_linear_data();

    let model = LinearRegression::new("y ~ x")
        .unwrap()
        .data(&df)
        .fit()
        .unwrap();

    let summary = model.summary().unwrap();

    // Convert to string
    let display = format!("{}", summary);
    assert!(!display.is_empty());

    // Should contain expected sections
    assert!(display.contains("Model Summary"));
    assert!(display.contains("Coefficients"));
    assert!(display.contains("Term"));
    assert!(display.contains("Estimate"));
    assert!(display.contains("Model Statistics"));
    assert!(display.contains("R-squared"));
}

// ==================== Edge Case Tests ====================

#[test]
fn test_linear_regression_single_observation() {
    let df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0]))
        .unwrap()
        .with_column("y", Series::float(vec![2.0]))
        .unwrap()
        .build()
        .unwrap();

    let result = LinearRegression::new("y ~ x").unwrap().data(&df).fit();

    // Should fail - need at least 2 observations for 2 parameters
    assert!(result.is_err());
}

#[test]
fn test_linear_regression_constant_predictor() {
    let df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 1.0, 1.0, 1.0, 1.0]))
        .unwrap()
        .with_column("z", Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap()
        .with_column("y", Series::float(vec![2.0, 4.0, 6.0, 8.0, 10.0]))
        .unwrap()
        .build()
        .unwrap();

    // This should work, but x will be perfectly collinear with intercept
    let result = LinearRegression::new("y ~ x + z").unwrap().data(&df).fit();

    // May succeed with warnings or fail due to singularity
    // The exact behavior depends on the solver implementation
    println!("Result: {:?}", result);
}

#[test]
fn test_linear_regression_perfect_collinearity() {
    let df = DataFrameBuilder::new()
        .with_column("x1", Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap()
        .with_column("x2", Series::float(vec![2.0, 4.0, 6.0, 8.0, 10.0])) // x2 = 2*x1
        .unwrap()
        .with_column("y", Series::float(vec![3.0, 6.0, 9.0, 12.0, 15.0]))
        .unwrap()
        .build()
        .unwrap();

    let result = LinearRegression::new("y ~ x1 + x2")
        .unwrap()
        .data(&df)
        .fit();

    // Should fail due to perfect collinearity
    assert!(result.is_err());
    match result.unwrap_err() {
        ModelError::SingularMatrix => {} // Expected
        _ => panic!("Expected SingularMatrix error"),
    }
}

#[test]
fn test_linear_regression_large_dataset() {
    // Test with larger dataset
    let n = 1000;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi).collect();

    let df = DataFrameBuilder::new()
        .with_column("x", Series::float(x))
        .unwrap()
        .with_column("y", Series::float(y))
        .unwrap()
        .build()
        .unwrap();

    let model = LinearRegression::new("y ~ x")
        .unwrap()
        .data(&df)
        .fit()
        .unwrap();

    let coeffs = model.coefficients().unwrap();
    assert_abs_diff_eq!(coeffs[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(coeffs[1], 2.0, epsilon = 1e-10);
}

// ==================== Benchmark Tests ====================

#[cfg(feature = "bench")]
mod bench {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_linear_regression_small(b: &mut Bencher) {
        let df = simple_linear_data();

        b.iter(|| {
            let _ = LinearRegression::new("y ~ x")
                .unwrap()
                .data(&df)
                .fit()
                .unwrap();
        });
    }

    #[bench]
    fn bench_linear_regression_medium(b: &mut Bencher) {
        let n = 1000;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi).collect();

        let df = DataFrameBuilder::new()
            .with_column("x", Series::float(x))
            .unwrap()
            .with_column("y", Series::float(y))
            .unwrap()
            .build()
            .unwrap();

        b.iter(|| {
            let _ = LinearRegression::new("y ~ x")
                .unwrap()
                .data(&df)
                .fit()
                .unwrap();
        });
    }

    #[bench]
    fn bench_linear_regression_large(b: &mut Bencher) {
        let n = 10000;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi).collect();

        let df = DataFrameBuilder::new()
            .with_column("x", Series::float(x))
            .unwrap()
            .with_column("y", Series::float(y))
            .unwrap()
            .build()
            .unwrap();

        b.iter(|| {
            let _ = LinearRegression::new("y ~ x")
                .unwrap()
                .data(&df)
                .fit()
                .unwrap();
        });
    }
}
