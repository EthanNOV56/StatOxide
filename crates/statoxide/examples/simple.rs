//! Simple example demonstrating StatOxide unified API

use std::collections::HashMap;
use statoxide::ndarray::Array1;

use statoxide::{
    DataFrame, Series, Formula,
    stats::{mean, std, correlation},
    prelude::*,
};

fn main() {
    println!("=== StatOxide Simple Example ===\n");
    
    // 1. Create a DataFrame
    println!("1. Creating DataFrame:");
    let mut columns = HashMap::new();
    columns.insert("x".to_string(), Series::new("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0])));
    columns.insert("y".to_string(), Series::new("y", Array1::from_vec(vec![2.0, 4.0, 5.0, 4.0, 5.0])));
    
    let df = DataFrame::from_series(columns).unwrap();
    
    println!("   DataFrame shape: {} rows × {} columns", df.n_rows(), df.n_cols());
    println!("   Column names: {:?}\n", df.column_names());
    
    // 2. Parse a formula
    println!("2. Parsing formula:");
    let formula = Formula::parse("y ~ x").unwrap();
    println!("   Formula parsed successfully, variables: {:?}\n", formula.variables());
    
    // 3. Compute statistics
    println!("3. Computing statistics:");
    let x_view = df.column("x").unwrap().data();
    let y_view = df.column("y").unwrap().data();
    
    // Convert views to owned arrays
    let x_data = x_view.to_owned();
    let y_data = y_view.to_owned();
    
    let x_mean = mean(&x_data).unwrap();
    let x_std = std(&x_data, 1.0).unwrap();
    let corr = correlation(&x_data, &y_data).unwrap();
    
    println!("   Mean of x: {:.4}", x_mean);
    println!("   Std of x: {:.4}", x_std);
    println!("   Correlation(x, y): {:.4}\n", corr);
    
    // 4. Demonstrate module structure
    println!("4. Module structure available:");
    println!("   - statoxide::models::*  (GLM, linear regression, mixed effects)");
    println!("   - statoxide::stats::*   (statistical functions and tests)");
    println!("   - statoxide::tsa::*     (time series analysis)");
    println!("   - statoxide::linalg::*  (linear algebra)");
    println!("   - statoxide::utils::*   (utilities)\n");
    
    println!("=== StatOxide version {} ===", statoxide::version());
}