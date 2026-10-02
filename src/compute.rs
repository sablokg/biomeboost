/*
Gaurav Sablok
gsablok@proton.me
*/

///median-of-ratios size factors (DESeq2-style), correcting for uneven sequencing depth across samples.

#[tokio::main]
pub async fn compute_size_factors(counts: &[Vec<f64>]) -> Vec<f64> {
    let n_samples = counts.len();
    let n_taxa = counts[0].len();

    let geo_means: Vec<f64> = (0..n_taxa)
        .map(|j| {
            let log_sum: f64 = counts
                .iter()
                .map(|row| if row[j] > 0.0 { row[j].ln() } else { 0.0 })
                .sum();
            let present = counts.iter().filter(|row| row[j] > 0.0).count();
            if present == 0 {
                0.0
            } else {
                (log_sum / present as f64).exp()
            }
        })
        .collect();

    (0..n_samples)
        .map(|i| {
            let mut ratios: Vec<f64> = (0..n_taxa)
                .filter_map(|j| {
                    if geo_means[j] > 0.0 && counts[i][j] > 0.0 {
                        Some(counts[i][j] / geo_means[j])
                    } else {
                        None
                    }
                })
                .collect();

            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = ratios.len() / 2;
            if ratios.is_empty() {
                1.0
            } else if ratios.len() % 2 == 0 {
                (ratios[mid - 1] + ratios[mid]) / 2.0
            } else {
                ratios[mid]
            }
        })
        .collect()
}
