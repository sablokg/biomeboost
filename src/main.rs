mod args;
use self::decisiontree::decisiontree;
use self::xgboost::xgboost;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
mod compute;
mod csvprepare;
mod decisiontree;
mod knn;
mod logistic;
mod microbiome;
mod randomforest;
mod xgboost;
use knn::knn;
use logistic::log;
use randomforest::random;
use vornix_banner::{Banner, BuiltinFont, Style, rgb};

/*
Gaurav Sablok
gsablok@proton.me
*/

#[tokio::main]
async fn main() {
    let style = Style::new().fg(rgb(255, 100, 20)).bold();

    let mut banner = Banner::new("biomeBOOST")
        .with_builtin_font(BuiltinFont::Block)
        .with_style(style)
        .centered(true);

    banner.display().unwrap();

    let args = CommandParse::parse();
    match &args.command {
        Commands::XGBoost { pathname, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = xgboost(pathname).unwrap();

                println!(
                    "The command has finished and the neural logistic has been trained:{:?}",
                    command
                );
            });
        }
        Commands::DecisionTree { otutable, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("threads failed");
            value.install(|| {
                let command = decisiontree(otutable).unwrap();
                println!("The command has completed:{}", command);
            });
        }
        Commands::Logistic { otutable, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("threads failed");
            value.install(|| {
                let command = log(otutable).unwrap();
                println!("The command has completed:{}", command);
            });
        }
        Commands::RandomForest { otutable, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("threads failed");
            value.install(|| {
                let command = random(otutable).unwrap();
                println!("The command has completed:{}", command);
            });
        }
        Commands::KNNClassifier { otutable, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("threads failed");
            value.install(|| {
                let command = knn(otutable).unwrap();
                println!("The command has completed:{}", command);
            });
        }
    }
}
