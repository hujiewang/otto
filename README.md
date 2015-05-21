# Otto Group Product Classification Challenge
# Final results: 58th 

## Models 

### Neural Networks with 3 hidden layers  
  ```
  93->BatchNormal->Dropout(0.3)->[Linear(512)->ReLU->BatchNormal->Dropout(0.5)]x3->Linear(9)->LogSoftMax
  ```
  The model performs 10% better if we use PReLU, but the training is too slow since CUDA version of PReLU has not been implemented yet.
  
* Initialization: Sparse Initialization

### Mixture of Experts of 3 Neural Networks
Experts: 
  
```
[93->BatchNormal->Dropout(0.25)->[Linear(512)->ReLU->BatchNormal->Dropout(0.5)]x3->Linear(9)->LogSoftMax]x3 
```
Gater:
  
```
93->[Linear(512)->ReLU]x3->Linear(3)->SoftMax
```
* Initialization: Sparse Initialization  

### Gradient Boosted Regression Trees (XGBoost)
  ```
  learning_rate=0.05,
  max_depth=16,
  max_samples=0.6,
  max_features=0.6,
  max_delta_step=0,
  min_child_weight=4,
  min_loss_reduction=1,
  l1_weight=0.0,
  l2_weight=0.0,
  l2_on_bias=False,
  gamma=0.02,
  inital_bias=0.5,
  ```
### Random Forests
  ```
  criterion='entropy',
  max_depth=29008, 
  max_features=36,
  max_leaf_nodes=None, 
  min_samples_leaf=5, 
  min_samples_split=3,
  min_weight_fraction_leaf=0.0, 
  n_estimators=4494,
  ```
  
###  SVM with RBF kernel

## Feature extraction methods  
  * Random Indexing 
  * Standard Nomalization   
 

## Ensemble Method   
* Assigned a weight([0.0,1.0]) to each model
* constraints: weights added up to 1.0
* Optimization: Sequential Least Squares Programming
* Initialization of weights: Random initialization

## Software
* python 2.7
* numpy
* scipy
* scikit-learn 
* Torch7
* CUDA
* xgboost

## Usage
* Put dataset into ./data folder
* th main.lua with options:
```
opt={
  createData = true,
  epochs = 3000,
  batch_size = 10000,
  predict = false,
  save_gap = 10,
  cuda=true,
  plot=false,
  sparse_init = true,
  standardize = true,
  --model_file = 'model.dat'
  RF = false,
  randFeat = false,
}
```
to create training and validation dataset under ./data folder
* ./start.sh
    * Trains a list of NN with Torch7 given a NN configuration (model.lua) 
* python rf.py (Random Forest model)
    * Output files: 
    * ./valid_results/valid_[model_id].csv (validation dataset)
    * ./results/results_[model_id].csv  (test dataset)
* python xgb.py (XGB model)
    * Output files: 
    * ./valid_results/valid_[model_id].csv (validation dataset)
    * ./results/results_[model_id].csv  (test dataset)
* python svm.py (SVM model)
    * Output files: 
    * ./valid_results/valid_[model_id].csv (validation dataset)
    * ./results/results_[model_id].csv  (test dataset)
* python blender.py
    * Output file: ./final_results/final_results.csv for submissions

# References
The sparse initialization method can be found in
[Deep learning via Hessian-free optimization](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_Martens10.pdf)

[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/pdf/1502.03167v3.pdf)

[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
