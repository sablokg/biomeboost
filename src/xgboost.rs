use crate::csvprepare::csvprepapre;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::model_selection::train_test_split;
use std::error::Error;
use xgb::{Booster, DMatrix, parameters};

/*
Gaurav Sablok
gsablok@proton.me
 */
#[tokio::main]
pub async fn xgboost(pathfile: &str) -> Result<String, Box<dyn Error>> {
    let vecunrap = csvprepapre(pathfile).unwrap();

    let numfeatures = vecunrap[0].len();

    let vecunwrap_dense = DenseMatrix::from_2d_vec(&vecunrap).unwrap();

    let mut labels: Vec<i32> = Vec::new();

    for i in 0..=vecunrap.len() {
        labels.push(i.to_string().parse::<i32>().unwrap());
    }

    let trainsplit = train_test_split(&vecunwrap_dense, &labels, 0.2, true, Some(48));

    let trainsplit_2 = trainsplit
        .2
        .iter()
        .map(|x| x.to_string().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    let trainsplit_3 = trainsplit
        .3
        .iter()
        .map(|x| x.to_string().parse::<f32>().unwrap())
        .collect::<Vec<_>>();

    let trainsplit_0_vec = trainsplit
        .0
        .iter()
        .map(move |x| x.clone())
        .collect::<Vec<_>>();
    let testsplit_0_vec = trainsplit
        .1
        .iter()
        .map(move |x| x.clone())
        .collect::<Vec<_>>();

    let mut dtrain = DMatrix::from_dense(&trainsplit_0_vec, numfeatures)?;
    dtrain.set_labels(&trainsplit_2)?;

    let mut dtest = DMatrix::from_dense(&testsplit_0_vec, numfeatures)?;
    dtest.set_labels(&trainsplit_3)?;

    let watchlist = &[(&dtrain, "train"), (&dtest, "eval")];

    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::MultiSoftmax(
            vecunrap.len() as u32,
        ))
        .eval_metrics(parameters::learning::Metrics::Custom(vec![
            parameters::learning::EvaluationMetric::MultiClassErrorRate,
            parameters::learning::EvaluationMetric::MultiClassLogLoss,
        ]))
        .build()?;

    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
        .max_depth(4)
        .eta(0.3)
        .subsample(0.8)
        .colsample_bytree(0.8)
        .build()?;

    let booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(true)
        .build()?;

    let params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain)
        .boost_rounds(50)
        .booster_params(booster_params)
        .evaluation_sets(Some(watchlist))
        .build()?;

    let booster = Booster::train(&params)?;
    booster.save("multiclass_model.bin")?;
    println!("Model saved");

    Ok("xgboost has finished".to_string())
}
