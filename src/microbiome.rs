use crate::compute::compute_size_factors;

/*
Gaurav Sablok
gsablok@proton.me
*/

/// apply size factors -> normalized_count = raw_count / size_factor
#[tokio::main]
pub async fn apply_size_factors(counts: &[Vec<f64>], size_factors: &[f64]) -> Vec<Vec<f64>> {
    counts
        .iter()
        .zip(size_factors.iter())
        .map(|(row, &sf)| row.iter().map(|&c| c / sf).collect())
        .collect()
}

/// pseudocount (avoids ln(0) in the CLR step)
#[tokio::main]
pub async fn add_pseudocount(counts: &[Vec<f64>], pseudocount: f64) -> Vec<Vec<f64>> {
    counts
        .iter()
        .map(|row| row.iter().map(|&c| c + pseudocount).collect())
        .collect()
}

/// CLR transform (ln(x_i) minus mean log-abundance per sample)
#[tokio::main]
pub async fn clr_transform(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|row| {
            let log_vals: Vec<f64> = row.iter().map(|&x| x.ln()).collect();
            let mean_log = log_vals.iter().sum::<f64>() / log_vals.len() as f64;
            log_vals.iter().map(|&lx| lx - mean_log).collect()
        })
        .collect()
}

/// Full preprocessing pipeline: raw counts -> normalized -> CLR features
#[tokio::main]
pub async fn preprocess(raw_counts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let size_factors = compute_size_factors(raw_counts);
    let normalized = apply_size_factors(raw_counts, &size_factors);
    let with_pseudocount = add_pseudocount(&normalized, 1.0);
    clr_transform(&with_pseudocount)
}
