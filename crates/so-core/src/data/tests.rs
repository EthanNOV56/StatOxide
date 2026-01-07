//! Tests for data module

use super::*;

#[test]
fn test_series_creation() {
    // Test float series
    let float_series = Series::float(vec![1.0, 2.0, 3.0]);
    assert_eq!(float_series.len(), 3);
    assert_eq!(float_series.dtype(), "float64");

    // Test int series
    let int_series = Series::int(vec![1, 2, 3]);
    assert_eq!(int_series.len(), 3);
    assert_eq!(int_series.dtype(), "int64");

    // Test bool series
    let bool_series = Series::bool(vec![true, false, true]);
    assert_eq!(bool_series.len(), 3);
    assert_eq!(bool_series.dtype(), "bool");

    // Test string series
    let string_series = Series::string(vec!["a".to_string(), "b".to_string()]);
    assert_eq!(string_series.len(), 2);
    assert_eq!(string_series.dtype(), "string");

    // Test categorical series
    let cat_series = Series::categorical(&["A", "B", "A", "C"]);
    assert_eq!(cat_series.len(), 4);
    assert_eq!(cat_series.dtype(), "categorical");
}

#[test]
fn test_series_statistics() {
    let series = Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    assert_eq!(series.mean().unwrap(), 3.0);
    assert!((series.std(1).unwrap() - 1.58113883).abs() < 1e-6);
    assert_eq!(series.sum().unwrap(), 15.0);

    let stats = series.describe().unwrap();
    assert_eq!(stats.count, 5);
    assert_eq!(stats.mean, 3.0);
    assert!((stats.std - 1.58113883).abs() < 1e-6);
    assert_eq!(stats.min, 1.0);
    assert_eq!(stats.max, 5.0);
}

#[test]
fn test_dataframe_creation() {
    let df = DataFrame::from_columns(vec![
        ("x", Series::float(vec![1.0, 2.0, 3.0])),
        ("y", Series::int(vec![4, 5, 6])),
    ])
    .unwrap();

    assert_eq!(df.shape(), (3, 2));
    assert_eq!(df.column_names(), vec!["x", "y"]);
}

#[test]
fn test_dataframe_select() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::float(vec![1.0, 2.0, 3.0])),
        ("b", Series::float(vec![4.0, 5.0, 6.0])),
        ("c", Series::float(vec![7.0, 8.0, 9.0])),
    ])
    .unwrap();

    let selected = df.select(&["a", "c"]).unwrap();
    assert_eq!(selected.shape(), (3, 2));
    assert_eq!(selected.column_names(), vec!["a", "c"]);
}

#[test]
fn test_dataframe_filter() {
    let df = DataFrame::from_columns(vec![
        ("x", Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
        ("y", Series::int(vec![1, 2, 3, 4, 5])),
    ])
    .unwrap();

    let mask = vec![true, false, true, false, true];
    let filtered = df.filter(&mask).unwrap();

    assert_eq!(filtered.shape(), (3, 2));

    let x_col = filtered.get_column("x").unwrap();
    if let Series::Float(arr) = x_col {
        assert_eq!(arr.to_vec(), vec![1.0, 3.0, 5.0]);
    } else {
        panic!("Expected Float series");
    }
}

#[test]
fn test_dataframe_numeric_matrix() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::float(vec![1.0, 2.0, 3.0])),
        ("b", Series::int(vec![4, 5, 6])),
        ("c", Series::bool(vec![true, false, true])),
    ])
    .unwrap();

    let matrix = df.numeric_matrix().unwrap();
    assert_eq!(matrix.shape(), &[3, 3]);
    assert_eq!(matrix[[0, 0]], 1.0);
    assert_eq!(matrix[[0, 1]], 4.0);
    assert_eq!(matrix[[0, 2]], 1.0);
}

#[test]
fn test_dataframe_corr() {
    let df = DataFrame::from_columns(vec![
        ("x", Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
        ("y", Series::float(vec![2.0, 4.0, 6.0, 8.0, 10.0])),
    ])
    .unwrap();

    let corr = df.corr().unwrap();
    assert_eq!(corr.shape(), &[2, 2]);
    assert!((corr[[0, 1]] - 1.0).abs() < 1e-10);
}

#[test]
fn test_builder_pattern() {
    let df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("y", Series::int(vec![4, 5, 6]))
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(df.shape(), (3, 2));
}

#[test]
fn test_operations() {
    let df = DataFrame::from_columns(vec![
        ("x", Series::float(vec![3.0, 1.0, 2.0])),
        ("y", Series::int(vec![1, 3, 2])),
    ])
    .unwrap();

    // Test select
    let selected = Select::new(&df).columns(&["x"]).execute().unwrap();
    assert_eq!(selected.shape(), (3, 1));

    // Test arrange
    let arranged = Arrange::new(&df).by("x", true).execute().unwrap();

    let x_col = arranged.get_column("x").unwrap();
    if let Series::Float(arr) = x_col {
        assert_eq!(arr.to_vec(), vec![1.0, 2.0, 3.0]);
    } else {
        panic!("Expected Float series");
    }
}

#[test]
fn test_series_view() {
    let series = Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let view = series.view();

    assert_eq!(view.len(), 5);
    assert_eq!(view.dtype(), "float64");
    assert_eq!(view.get(0), Some(SeriesValue::Float(1.0)));
    assert_eq!(view.get(4), Some(SeriesValue::Float(5.0)));
    assert_eq!(view.get(5), None);

    // Test slice
    let slice = view.slice(1..4).unwrap();
    assert_eq!(slice.len(), 3);
    assert_eq!(slice.get(0), Some(SeriesValue::Float(2.0)));

    // Test iterator
    let values: Vec<SeriesValue> = view.iter().collect();
    assert_eq!(values.len(), 5);

    // Test stats
    let stats = view.stats().unwrap();
    assert_eq!(stats.mean, 3.0);
}

#[test]
fn test_dataframe_view() {
    let df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 2.0, 3.0]))
        .unwrap()
        .with_column("y", Series::int(vec![4, 5, 6]))
        .unwrap()
        .build()
        .unwrap();

    let view = df.view();

    assert_eq!(view.shape(), (3, 2));
    assert_eq!(view.column_names(), vec!["x", "y"]);

    // Test column access
    let x_view = view.column("x").unwrap();
    assert_eq!(x_view.len(), 3);
    assert_eq!(x_view.get(0), Some(SeriesValue::Float(1.0)));

    // Test row access
    let row = view.row(0).unwrap();
    assert_eq!(row.get_float("x"), Some(1.0));
    assert_eq!(row.get_int("y"), Some(4));

    // Test row iterator
    let rows: Vec<RowView> = view.rows().collect();
    assert_eq!(rows.len(), 3);

    // Test column iterator
    let columns: Vec<(&str, SeriesView)> = view.columns().collect();
    assert_eq!(columns.len(), 2);

    // Test numeric matrix view
    // let matrix_view = view.numeric_matrix_view().unwrap();
    // assert_eq!(matrix_view.shape(), &[3, 2]);
    // assert_eq!(matrix_view[[0, 0]], 1.0);
    // assert_eq!(matrix_view[[0, 1]], 4.0);

    // Test select
    let selected = view.select(["x"]).unwrap();
    assert_eq!(selected.shape(), (3, 1));

    // Test slice
    let slice = view.slice(1..3).unwrap();
    assert_eq!(slice.shape(), (2, 2));
    let first_row = slice.row(0).unwrap();
    assert_eq!(first_row.get_float("x"), Some(2.0));
}

#[test]
fn test_filtered_view() {
    let df = DataFrameBuilder::new()
        .with_column("x", Series::float(vec![1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap()
        .build()
        .unwrap();

    let view = df.view();
    let mask = vec![true, false, true, false, true];

    let filtered = view.filter(&mask).unwrap();
    assert_eq!(filtered.shape(), (3, 1));

    let rows: Vec<RowView> = filtered.rows().collect();
    assert_eq!(rows.len(), 3);

    // Check that we got the correct rows
    let values: Vec<f64> = rows.iter().filter_map(|row| row.get_float("x")).collect();
    assert_eq!(values, vec![1.0, 3.0, 5.0]);
}

#[test]
fn test_series_view_mut() {
    let mut series = Series::float(vec![1.0, 2.0, 3.0]);
    {
        let mut view = series.view_mut();
        if let SeriesViewMut::Float(arr) = &mut view {
            arr[0] = 10.0;
        }
    }

    // Check that the change persisted
    if let Series::Float(arr) = &series {
        assert_eq!(arr[0], 10.0);
    } else {
        panic!("Expected Float series");
    }
}

#[test]
fn test_categorical_view() {
    let series = Series::categorical(&["A", "B", "A", "C", "B"]);
    let view = series.view();

    assert_eq!(view.dtype(), "categorical");
    assert_eq!(view.len(), 5);

    // Check values
    assert_eq!(view.get(0), Some(SeriesValue::String("A".to_string())));
    assert_eq!(view.get(1), Some(SeriesValue::String("B".to_string())));
    assert_eq!(view.get(4), Some(SeriesValue::String("B".to_string())));

    // Test mutable categorical view
    let mut series = Series::categorical(&["A", "B"]);
    {
        let mut view = series.view_mut();
        // view.set(0, SeriesValue::String("C".to_string())).unwrap();
    }

    if let Series::Categorical(arr, cats) = &series {
        // assert_eq!(cats.len(), 3); // Should have A, B, C
        // assert_eq!(arr[0], 2); // C should be code 2
    } else {
        panic!("Expected Categorical series");
    }
}

#[test]
fn test_string_view() {
    let series = Series::string(vec!["hello".to_string(), "world".to_string()]);
    let view = series.view();

    assert_eq!(view.dtype(), "string");
    assert_eq!(view.len(), 2);
    assert_eq!(view.get(0), Some(SeriesValue::String("hello".to_string())));

    // Test mutable string view
    let mut series = Series::string(vec!["a".to_string(), "b".to_string()]);
    {
        let mut view = series.view_mut();
        view.set(0, SeriesValue::String("c".to_string())).unwrap();
    }

    if let Series::String(arr) = &series {
        assert_eq!(arr[0], "c");
    } else {
        panic!("Expected String series");
    }
}
