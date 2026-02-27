//! Statistical distances and divergences

use ndarray::{Array1, Array2};

/// Compute Euclidean distance between two vectors
pub fn euclidean_distance(x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
    if x.len() != y.len() {
        return None;
    }
    
    let mut sum = 0.0;
    for i in 0..x.len() {
        let diff = x[i] - y[i];
        sum += diff * diff;
    }
    
    Some(sum.sqrt())
}

/// Compute Manhattan distance (L1 distance)
pub fn manhattan_distance(x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
    if x.len() != y.len() {
        return None;
    }
    
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (x[i] - y[i]).abs();
    }
    
    Some(sum)
}

/// Compute Chebyshev distance (L-infinity distance)
pub fn chebyshev_distance(x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
    if x.len() != y.len() {
        return None;
    }
    
    let mut max_diff = 0.0;
    for i in 0..x.len() {
        let diff = (x[i] - y[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    Some(max_diff)
}

/// Compute Minkowski distance with given p parameter
pub fn minkowski_distance(x: &Array1<f64>, y: &Array1<f64>, p: f64) -> Option<f64> {
    if x.len() != y.len() || p <= 0.0 {
        return None;
    }
    
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (x[i] - y[i]).abs().powf(p);
    }
    
    Some(sum.powf(1.0 / p))
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
    if x.len() != y.len() {
        return None;
    }
    
    let mut dot = 0.0;
    let mut norm_x = 0.0;
    let mut norm_y = 0.0;
    
    for i in 0..x.len() {
        dot += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }
    
    if norm_x == 0.0 || norm_y == 0.0 {
        return Some(0.0);
    }
    
    Some(dot / (norm_x.sqrt() * norm_y.sqrt()))
}

/// Compute cosine distance (1 - cosine similarity)
pub fn cosine_distance(x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
    cosine_similarity(x, y).map(|sim| 1.0 - sim)
}

/// Compute Jensen-Shannon divergence between two probability distributions
pub fn jensen_shannon_divergence(p: &Array1<f64>, q: &Array1<f64>) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    
    // Normalize to probability distributions
    let p_sum: f64 = p.iter().sum();
    let q_sum: f64 = q.iter().sum();
    
    if p_sum <= 0.0 || q_sum <= 0.0 {
        return None;
    }
    
    let p_norm: Array1<f64> = p / p_sum;
    let q_norm: Array1<f64> = q / q_sum;
    
    // Compute m = (p + q) / 2
    let m: Array1<f64> = &p_norm + &q_norm;
    let m = m / 2.0;
    
    // JS divergence = (KL(p || m) + KL(q || m)) / 2
    let kl_pm = kl_divergence(&p_norm, &m)?;
    let kl_qm = kl_divergence(&q_norm, &m)?;
    
    Some((kl_pm + kl_qm) / 2.0)
}

/// Compute Kullback-Leibler divergence (relative entropy)
pub fn kl_divergence(p: &Array1<f64>, q: &Array1<f64>) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    
    let mut divergence = 0.0;
    
    for i in 0..p.len() {
        if p[i] > 0.0 {
            if q[i] <= 0.0 {
                return None; // KL divergence is infinite
            }
            divergence += p[i] * (p[i] / q[i]).ln();
        }
    }
    
    Some(divergence)
}

/// Compute Bhattacharyya distance between two probability distributions
pub fn bhattacharyya_distance(p: &Array1<f64>, q: &Array1<f64>) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    
    // Normalize to probability distributions
    let p_sum: f64 = p.iter().sum();
    let q_sum: f64 = q.iter().sum();
    
    if p_sum <= 0.0 || q_sum <= 0.0 {
        return None;
    }
    
    let p_norm: Array1<f64> = p / p_sum;
    let q_norm: Array1<f64> = q / q_sum;
    
    // Compute Bhattacharyya coefficient
    let mut bc = 0.0;
    for i in 0..p_norm.len() {
        bc += (p_norm[i] * q_norm[i]).sqrt();
    }
    
    // Distance = -ln(BC)
    if bc <= 0.0 {
        return None;
    }
    
    Some(-bc.ln())
}

/// Compute Hellinger distance between two probability distributions
pub fn hellinger_distance(p: &Array1<f64>, q: &Array1<f64>) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    
    // Normalize to probability distributions
    let p_sum: f64 = p.iter().sum();
    let q_sum: f64 = q.iter().sum();
    
    if p_sum <= 0.0 || q_sum <= 0.0 {
        return None;
    }
    
    let p_norm: Array1<f64> = p / p_sum;
    let q_norm: Array1<f64> = q / q_sum;
    
    let mut sum = 0.0;
    for i in 0..p_norm.len() {
        let diff = p_norm[i].sqrt() - q_norm[i].sqrt();
        sum += diff * diff;
    }
    
    Some((sum / 2.0).sqrt())
}

/// Compute Wasserstein distance (Earth Mover's Distance) for 1D distributions
pub fn wasserstein_distance_1d(p: &Array1<f64>, q: &Array1<f64>) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    
    // For 1D distributions, Wasserstein distance is the L1 distance between CDFs
    let mut p_sorted = p.to_vec();
    let mut q_sorted = q.to_vec();
    p_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    q_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mut distance = 0.0;
    for i in 0..p_sorted.len() {
        distance += (p_sorted[i] - q_sorted[i]).abs();
    }
    
    Some(distance / p_sorted.len() as f64)
}

/// Compute Mahalanobis distance between a point and a distribution
pub fn mahalanobis_distance(
    x: &Array1<f64>,
    mean: &Array1<f64>,
    cov_inv: &Array2<f64>,
) -> Option<f64> {
    if x.len() != mean.len() || x.len() != cov_inv.nrows() || cov_inv.nrows() != cov_inv.ncols() {
        return None;
    }
    
    let diff = x - mean;
    let n = x.len();
    
    // Compute diff^T * cov_inv * diff
    let mut temp = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in 0..n {
            temp[i] += diff[j] * cov_inv[(j, i)];
        }
    }
    
    let mut distance_sq: f64 = 0.0;
    for i in 0..n {
        distance_sq += diff[i] * temp[i];
    }
    
    Some(distance_sq.sqrt())
}