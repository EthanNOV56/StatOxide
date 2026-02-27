//! Correlation measures and related statistics

use ndarray::Array1;

/// Compute Pearson correlation coefficient between two arrays
pub fn pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    
    let mean_x = x.mean()?;
    let mean_y = y.mean()?;
    
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }
    
    if sum_xx == 0.0 || sum_yy == 0.0 {
        return Some(0.0);
    }
    
    Some(sum_xy / (sum_xx.sqrt() * sum_yy.sqrt()))
}

/// Compute Spearman rank correlation coefficient
pub fn spearman_correlation(x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    
    // Rank the data
    let rank_x = rank(x);
    let rank_y = rank(y);
    
    // Compute Pearson correlation on ranks
    pearson_correlation(&rank_x, &rank_y)
}

/// Compute Kendall's tau rank correlation coefficient
pub fn kendall_tau(x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    
    let n = x.len();
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    
    for i in 0..(n - 1) {
        for j in (i + 1)..n {
            let x_order = (x[i] - x[j]).signum();
            let y_order = (y[i] - y[j]).signum();
            
            if x_order * y_order > 0.0 {
                concordant += 1;
            } else if x_order * y_order < 0.0 {
                discordant += 1;
            }
            // ties are ignored in this simple implementation
        }
    }
    
    let total = concordant + discordant;
    if total == 0 {
        return Some(0.0);
    }
    
    Some((concordant as f64 - discordant as f64) / total as f64)
}

/// Compute point-biserial correlation (between continuous and binary variable)
pub fn point_biserial_correlation(continuous: &Array1<f64>, binary: &Array1<f64>) -> Option<f64> {
    if continuous.len() != binary.len() || continuous.len() < 2 {
        return None;
    }
    
    // Split continuous variable by binary groups
    let mut group0 = Vec::new();
    let mut group1 = Vec::new();
    
    for i in 0..continuous.len() {
        if binary[i] == 0.0 {
            group0.push(continuous[i]);
        } else if binary[i] == 1.0 {
            group1.push(continuous[i]);
        } else {
            // Not a proper binary variable
            return None;
        }
    }
    
    if group0.is_empty() || group1.is_empty() {
        return None;
    }
    
    let mean0: f64 = group0.iter().sum::<f64>() / group0.len() as f64;
    let mean1: f64 = group1.iter().sum::<f64>() / group1.len() as f64;
    
    let n0 = group0.len() as f64;
    let n1 = group1.len() as f64;
    let n = n0 + n1;
    
    // Pooled standard deviation
    let var0: f64 = group0.iter().map(|&x| (x - mean0).powi(2)).sum::<f64>() / (n0 - 1.0);
    let var1: f64 = group1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let pooled_var = ((n0 - 1.0) * var0 + (n1 - 1.0) * var1) / (n - 2.0);
    let pooled_std = pooled_var.sqrt();
    
    if pooled_std == 0.0 {
        return Some(0.0);
    }
    
    let r_pb = (mean1 - mean0) / pooled_std * (n0 * n1 / (n * n)).sqrt();
    Some(r_pb)
}

/// Compute partial correlation between x and y controlling for z
pub fn partial_correlation(
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array1<f64>,
) -> Option<f64> {
    if x.len() != y.len() || x.len() != z.len() || x.len() < 3 {
        return None;
    }
    
    let r_xy = pearson_correlation(x, y)?;
    let r_xz = pearson_correlation(x, z)?;
    let r_yz = pearson_correlation(y, z)?;
    
    let numerator = r_xy - r_xz * r_yz;
    let denominator = ((1.0 - r_xz.powi(2)) * (1.0 - r_yz.powi(2))).sqrt();
    
    if denominator == 0.0 {
        return None;
    }
    
    Some(numerator / denominator)
}

/// Compute multiple correlation coefficient (R) for multiple regression
pub fn multiple_correlation(y: &Array1<f64>, y_pred: &Array1<f64>) -> Option<f64> {
    if y.len() != y_pred.len() || y.len() < 2 {
        return None;
    }
    
    let ss_residual: f64 = y.iter().zip(y_pred.iter())
        .map(|(&yi, &yhat)| (yi - yhat).powi(2))
        .sum();
    
    let mean_y = y.mean()?;
    let ss_total: f64 = y.iter()
        .map(|&yi| (yi - mean_y).powi(2))
        .sum();
    
    if ss_total == 0.0 {
        return Some(0.0);
    }
    
    let r2 = 1.0 - ss_residual / ss_total;
    Some(r2.sqrt())
}

/// Compute autocorrelation for time series data
pub fn autocorrelation(series: &Array1<f64>, lag: usize) -> Option<f64> {
    let n = series.len();
    if n <= lag || n < 2 {
        return None;
    }
    
    let mean = series.mean()?;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..(n - lag) {
        numerator += (series[i] - mean) * (series[i + lag] - mean);
        denominator += (series[i] - mean).powi(2);
    }
    
    if denominator == 0.0 {
        return Some(0.0);
    }
    
    Some(numerator / denominator)
}

/// Compute cross-correlation between two time series
pub fn cross_correlation(x: &Array1<f64>, y: &Array1<f64>, lag: i32) -> Option<f64> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return None;
    }
    
    let mean_x = x.mean()?;
    let mean_y = y.mean()?;
    let mut numerator = 0.0;
    let mut denom_x = 0.0;
    let mut denom_y = 0.0;
    
    if lag >= 0 {
        let start = lag as usize;
        for i in start..n {
            let j = i - start;
            numerator += (x[i] - mean_x) * (y[j] - mean_y);
            denom_x += (x[i] - mean_x).powi(2);
            denom_y += (y[j] - mean_y).powi(2);
        }
    } else {
        let lag_abs = (-lag) as usize;
        for i in 0..(n - lag_abs) {
            let j = i + lag_abs;
            numerator += (x[i] - mean_x) * (y[j] - mean_y);
            denom_x += (x[i] - mean_x).powi(2);
            denom_y += (y[j] - mean_y).powi(2);
        }
    }
    
    if denom_x == 0.0 || denom_y == 0.0 {
        return Some(0.0);
    }
    
    Some(numerator / (denom_x.sqrt() * denom_y.sqrt()))
}

/// Helper function to compute ranks with tie handling
fn rank(data: &Array1<f64>) -> Array1<f64> {
    let n = data.len();
    let mut indices: Vec<usize> = (0..n).collect();
    
    // Sort indices by data values
    indices.sort_by(|&i, &j| {
        data[i].partial_cmp(&data[j]).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let mut ranks = Array1::zeros(n);
    let mut i = 0;
    
    while i < n {
        let mut j = i;
        // Find ties
        while j + 1 < n && data[indices[j]] == data[indices[j + 1]] {
            j += 1;
        }
        
        // Average rank for tied values
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        
        for k in i..=j {
            ranks[indices[k]] = avg_rank;
        }
        
        i = j + 1;
    }
    
    ranks
}