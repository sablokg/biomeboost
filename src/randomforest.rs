use crate::microbiome::preprocess;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use std::error::Error;
use tokio::fs::File;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;

/*
Gaurav Sablok
gsablok@proton.me
 */
#[tokio::main]
pub async fn random(pathfile: &str) -> Result<String, Box<dyn Error>> {
    let pathfile_open = File::open(pathfile).await?;
    let pathfile_read = BufReader::new(pathfile_open);
    let mut lines = pathfile_read.lines();

    let mut rawcount: Vec<Vec<String>> = Vec::new();
    let mut label: Vec<i32> = Vec::new();

    while let Some(line) = lines.next_line().await? {
        let fields = line.split(',').collect::<Vec<_>>();
        let (features, lbl) = fields.split_at(fields.len() - 1);

        rawcount.push(features.iter().map(|s| s.to_string()).collect());
        label.push(lbl[0].parse::<i32>().unwrap());
    }

    let finalrawcounts = rawcount
        .iter()
        .map(|row| {
            row.iter()
                .map(|s| s.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let clr_data = preprocess(&finalrawcounts);

    println!("CLR-transformed, size-factor-normalized features:");
    for row in &clr_data {
        println!("{:?}", row);
    }

    let x = DenseMatrix::from_2d_vec(&clr_data).unwrap();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &label, 0.25, true, None);

    let model = RandomForestClassifier::fit(
        &x_train,
        &y_train,
        RandomForestClassifierParameters::default()
            .with_max_depth(10)
            .with_criterion(smartcore::tree::decision_tree_classifier::SplitCriterion::Gini)
            .with_keep_samples(true)
            .with_n_trees(100)
            .with_seed(4849),
    )
    .unwrap();

    let predictions = model.predict(&x_test).unwrap();

    println!("\nPredictions: {:?}", predictions);
    println!("True labels:  {:?}", y_test);
    println!("Accuracy: {:.2}%", accuracy(&y_test, &predictions) * 100.0);

    Ok("decision tree has been finished".to_string())
}
