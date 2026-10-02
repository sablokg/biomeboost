# biomeboost

- all machine learning on microbiome from a single standalone crate.
- all preparocessing steps added OTU Table -> size factor -> pseudocounts -> CLR -> Machine learning. 
- In RandomForest, I added Gini as split criterion like it should be used in microbiome.

#### Entire machine learning for microbiome I wrote using Rust. Much faster and async runtime. 

```
cargo build
```

```
                                                                                                                                                           
                                            _|         _|                                          _|_|_|       _|_|       _|_|       _|_|_|   _|_|_|_|_|  
                                            _|_|_|            _|_|     _|_|_|  _|_|       _|_|     _|    _|   _|    _|   _|    _|   _|             _|      
                                            _|    _|   _|   _|    _|   _|    _|    _|   _|_|_|_|   _|_|_|     _|    _|   _|    _|     _|_|         _|      
                                            _|    _|   _|   _|    _|   _|    _|    _|   _|         _|    _|   _|    _|   _|    _|         _|       _|      
                                            _|_|_|     _|     _|_|     _|    _|    _|     _|_|_|   _|_|_|       _|_|       _|_|     _|_|_|         _|      
                                                                                                                                                           
                                                                                                                                                           
Entire machine learning for microbiome
       ************************************************
       Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************

Usage: biomeboost <COMMAND>

Commands:
  xg-boost        XGBoost classification
  decision-tree   Decision Tree
  logistic        Logistic Classification
  random-forest   RandomForest
  knn-classifier  KNNClassifier
  help            Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

Gaurav Sablok \
gsablok@proton.me
