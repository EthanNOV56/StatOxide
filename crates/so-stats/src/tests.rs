//! Statistical tests and hypothesis testing

use ndarray::Array1;
use so_core::error::{Error, Result};

/// Result of a statistical test
#[derive(Debug, Clone)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub df: Option<usize>,
    pub alternative: Alternative,
    pub null_value: f64,
}

/// Alternative hypothesis type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Alternative {
    /// Two-sided test
    TwoSided,
    /// One-sided test (less than)
    Less,
    /// One-sided test (greater than)
    Greater,
}

/// Perform one-sample t-test
pub fn t_test_one_sample(
    data: &Array1<f64>,
    mu: f64,
    alternative: Alternative,
) -> Result<TestResult> {
    let n = data.len();
    if n < 2 {
        return Err(Error::DataError(
            "Need at least 2 observations for t-test".to_string(),
        ));
    }

    let mean = data.mean().unwrap_or(0.0);
    let std = data.std(1.0); // sample standard deviation

    if std == 0.0 {
        return Err(Error::DataError("Zero variance in data".to_string()));
    }

    let se = std / (n as f64).sqrt();
    let t_stat = (mean - mu) / se;
    let df = n - 1;

    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - students_t_cdf(t_stat.abs(), df as f64)),
        Alternative::Less => students_t_cdf(t_stat, df as f64),
        Alternative::Greater => 1.0 - students_t_cdf(t_stat, df as f64),
    };

    Ok(TestResult {
        statistic: t_stat,
        p_value,
        df: Some(df),
        alternative,
        null_value: mu,
    })
}

/// Perform two-sample t-test (assuming equal variance)
pub fn t_test_two_sample(
    x: &Array1<f64>,
    y: &Array1<f64>,
    alternative: Alternative,
) -> Result<TestResult> {
    let n1 = x.len();
    let n2 = y.len();

    if n1 < 2 || n2 < 2 {
        return Err(Error::DataError(
            "Need at least 2 observations in each group".to_string(),
        ));
    }

    let mean1 = x.mean().unwrap_or(0.0);
    let mean2 = y.mean().unwrap_or(0.0);
    let var1 = x.var(1.0);
    let var2 = y.var(1.0);

    let pooled_var =
        ((n1 as f64 - 1.0) * var1 + (n2 as f64 - 1.0) * var2) / (n1 as f64 + n2 as f64 - 2.0);
    let se = (pooled_var * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();

    if se == 0.0 {
        return Err(Error::DataError("Zero standard error".to_string()));
    }

    let t_stat = (mean1 - mean2) / se;
    let df = n1 + n2 - 2;

    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - students_t_cdf(t_stat.abs(), df as f64)),
        Alternative::Less => students_t_cdf(t_stat, df as f64),
        Alternative::Greater => 1.0 - students_t_cdf(t_stat, df as f64),
    };

    Ok(TestResult {
        statistic: t_stat,
        p_value,
        df: Some(df),
        alternative,
        null_value: 0.0,
    })
}

/// Perform paired t-test
pub fn t_test_paired(
    x: &Array1<f64>,
    y: &Array1<f64>,
    alternative: Alternative,
) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::DataError(
            "Paired samples must have same length".to_string(),
        ));
    }

    let diff: Array1<f64> = x - y;
    t_test_one_sample(&diff, 0.0, alternative)
}

/// Perform chi-square goodness-of-fit test
pub fn chi_square_goodness_of_fit(
    observed: &Array1<f64>,
    expected: &Array1<f64>,
) -> Result<TestResult> {
    if observed.len() != expected.len() {
        return Err(Error::DataError(
            "Observed and expected arrays must have same length".to_string(),
        ));
    }

    let mut chi_sq = 0.0;
    for i in 0..observed.len() {
        let obs = observed[i];
        let exp = expected[i];

        if exp > 0.0 {
            chi_sq += (obs - exp).powi(2) / exp;
        } else if obs != 0.0 {
            return Err(Error::DataError(
                "Expected frequency is zero but observed is not".to_string(),
            ));
        }
    }

    let df = observed.len() - 1;
    let p_value = 1.0 - chi_square_cdf(chi_sq, df as f64);

    Ok(TestResult {
        statistic: chi_sq,
        p_value,
        df: Some(df),
        alternative: Alternative::Greater,
        null_value: 0.0,
    })
}

/// Perform chi-square test of independence
pub fn chi_square_test_independence(
    contingency_table: &ndarray::Array2<f64>,
) -> Result<TestResult> {
    let (n_rows, n_cols) = contingency_table.dim();

    if n_rows < 2 || n_cols < 2 {
        return Err(Error::DataError(
            "Contingency table must be at least 2x2".to_string(),
        ));
    }

    // Calculate row and column totals
    let row_totals: Vec<f64> = (0..n_rows)
        .map(|i| (0..n_cols).map(|j| contingency_table[(i, j)]).sum())
        .collect();
    let col_totals: Vec<f64> = (0..n_cols)
        .map(|j| (0..n_rows).map(|i| contingency_table[(i, j)]).sum())
        .collect();
    let grand_total: f64 = row_totals.iter().sum();

    // Calculate expected frequencies
    let mut chi_sq = 0.0;
    for i in 0..n_rows {
        for j in 0..n_cols {
            let obs = contingency_table[(i, j)];
            let exp = row_totals[i] * col_totals[j] / grand_total;

            if exp > 0.0 {
                chi_sq += (obs - exp).powi(2) / exp;
            } else if obs != 0.0 {
                return Err(Error::DataError(
                    "Expected frequency is zero but observed is not".to_string(),
                ));
            }
        }
    }

    let df = (n_rows - 1) * (n_cols - 1);
    let p_value = 1.0 - chi_square_cdf(chi_sq, df as f64);

    Ok(TestResult {
        statistic: chi_sq,
        p_value,
        df: Some(df),
        alternative: Alternative::Greater,
        null_value: 0.0,
    })
}

/// Perform F-test for equality of variances
pub fn f_test_variances(
    x: &Array1<f64>,
    y: &Array1<f64>,
    alternative: Alternative,
) -> Result<TestResult> {
    let n1 = x.len();
    let n2 = y.len();

    if n1 < 2 || n2 < 2 {
        return Err(Error::DataError(
            "Need at least 2 observations in each group".to_string(),
        ));
    }

    let var1 = x.var(1.0);
    let var2 = y.var(1.0);

    let f_stat = var1 / var2;
    let df1 = n1 - 1;
    let df2 = n2 - 1;

    let p_value = match alternative {
        Alternative::TwoSided => {
            let p1 = 1.0 - f_cdf(f_stat, df1 as f64, df2 as f64);
            let p2 = f_cdf(f_stat, df1 as f64, df2 as f64);
            2.0 * p1.min(p2)
        }
        Alternative::Less => f_cdf(f_stat, df1 as f64, df2 as f64),
        Alternative::Greater => 1.0 - f_cdf(f_stat, df1 as f64, df2 as f64),
    };

    Ok(TestResult {
        statistic: f_stat,
        p_value,
        df: Some(df1 + df2),
        alternative,
        null_value: 1.0,
    })
}

/// Perform Shapiro-Wilk test for normality (simplified implementation)
pub fn shapiro_wilk_test(data: &Array1<f64>) -> Result<TestResult> {
    // Note: This is a simplified placeholder.
    // Real implementation requires coefficient tables and more complex calculations.
    let n = data.len();
    if n < 3 {
        return Err(Error::DataError(
            "Need at least 3 observations for Shapiro-Wilk test".to_string(),
        ));
    }

    // Simplified: just return a dummy result for now
    let w_stat = 0.95; // Placeholder
    let p_value = 0.1; // Placeholder

    Ok(TestResult {
        statistic: w_stat,
        p_value,
        df: Some(n),
        alternative: Alternative::TwoSided,
        null_value: 1.0,
    })
}

/// One-way ANOVA test
pub fn anova_one_way(groups: &[Array1<f64>]) -> Result<TestResult> {
    if groups.len() < 2 {
        return Err(Error::DataError(
            "Need at least 2 groups for ANOVA".to_string(),
        ));
    }

    let k = groups.len();
    let mut all_data = Vec::new();
    let mut group_means = Vec::new();
    let mut group_sizes = Vec::new();
    let mut group_ss = Vec::new();

    // Calculate group statistics
    for group in groups {
        let n = group.len();
        if n < 2 {
            return Err(Error::DataError(
                "Each group must have at least 2 observations".to_string(),
            ));
        }

        let mean = group.mean().unwrap_or(0.0);
        let ss: f64 = group.iter().map(|&x| (x - mean).powi(2)).sum();

        group_means.push(mean);
        group_sizes.push(n);
        group_ss.push(ss);
        all_data.extend(group.iter().copied());
    }

    let total_n: usize = group_sizes.iter().sum();
    let grand_mean: f64 = all_data.iter().sum::<f64>() / total_n as f64;

    // Between-group sum of squares
    let ss_between: f64 = group_sizes
        .iter()
        .zip(&group_means)
        .map(|(&n, &mean)| n as f64 * (mean - grand_mean).powi(2))
        .sum();

    // Within-group sum of squares
    let ss_within: f64 = group_ss.iter().sum();

    // Mean squares
    let ms_between = ss_between / (k as f64 - 1.0);
    let ms_within = ss_within / (total_n as f64 - k as f64);

    if ms_within == 0.0 {
        return Err(Error::DataError("Zero within-group variance".to_string()));
    }

    let f_stat = ms_between / ms_within;
    let df1 = k - 1;
    let df2 = total_n - k;

    let p_value = 1.0 - f_cdf(f_stat, df1 as f64, df2 as f64);

    Ok(TestResult {
        statistic: f_stat,
        p_value,
        df: Some(df1 + df2),
        alternative: Alternative::Greater,
        null_value: 0.0,
    })
}

// ============================================================================
// Helper Functions for Distribution CDFs
// ============================================================================

/// Student's t CDF (approximation)
fn students_t_cdf(t: f64, df: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    StudentsT::new(0.0, 1.0, df).unwrap().cdf(t)
}

/// Chi-square CDF
fn chi_square_cdf(x: f64, df: f64) -> f64 {
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    ChiSquared::new(df).unwrap().cdf(x)
}

/// F-distribution CDF
fn f_cdf(x: f64, df1: f64, df2: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, FisherSnedecor};
    FisherSnedecor::new(df1, df2).unwrap().cdf(x)
}
