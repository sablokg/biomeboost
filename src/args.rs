use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "biomeboost",
    version = "1.0",
    about = "Entire machine learning for microbiome
       ************************************************
       Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// XGBoost classification
    XGBoost {
        /// path to the filename
        pathname: String,
        /// Threads for the analysis
        thread: String,
    },
    /// Decision Tree
    DecisionTree {
        /// path to the OTU table
        otutable: String,
        /// Threads for the analysis
        thread: String,
    },
    /// Logistic Classification
    Logistic {
        /// path to the OTU table
        otutable: String,
        /// Threads for the analysis
        thread: String,
    },
    /// RandomForest
    RandomForest {
        /// path to the OTU table
        otutable: String,
        /// Threads for the analysis
        thread: String,
    },
    /// KNNClassifier
    KNNClassifier {
        /// path to the OTU table
        otutable: String,
        /// Threads for the analysis
        thread: String,
    },
}
