Python-Regression-Tree-Forest
=============================

Python implementation of regression trees and random forests. See "Classification and Regression Trees" by Breiman et al. (1984).

The regression_tree_cart.py module contains the functions to grow and use a regression tree given some training data.

football_parserf.py is an example implementation of regression_tree_cart.py that predicts an NFL player's fantasy points given their statistics from the previous year. The data is stored in football.csv. 

The random_forest.py module contains the functions to grow a random forest and use it for prediction. 

football_forest.py is an example implementation of random_forest.py.

## Run

```
python football_cart.py [options]
```
<b>Options</b>
* --load filename: loads model from filename, else, construct model from training dataset
* --save filename: saves pruned model to filename
* --plot-tree: plots the entire tree into image
* --plot-weights: plots the weights of split features in the entire tree
* --plot-feature: plots weights of split features for the first test case
* --test: runs the test cases

## Setup

```
pip3 install -r requirements.txt
```
