# **Mitigating Data Center Bias in Cancer Classification: Feature Reduction and Transfer Unlearning via Multi-objective Conflict-Of-Interest Free Optimization**

## **Overview**
This repository contains the code, required dependencies, and instructions for running the project.

## Getting Started
The project is developed using following packages:
- **`Python Vesrion`**: 3.10
- **`Tensorflow Vesrion`**: 2.10
- **`Pymoo`**: Installable via
  `pip install -U pymoo`

## Dataset
The dataset used for this project (a balanced subset of TCGA dataset) can be downloaded from the following link: [Dataset - Google Drive](https://drive.google.com/drive/folders/1FU40tvcKCHGqQnmzPha2dtCAWNZpViR-?usp=sharing)

## `data_loading.py`
The code uses the data provided in the given path and the pretrained model to extract the features of cancerous samples. The outputs are loaded into the methods,  `Multi-objective-Conflict-Of-Interest-Free-Optimization.py`, `adversarial_learning.py`, and `multi_task_learning.py` to complete the unlearning procedures.

## Key Functions in `Multi-objective-Conflict-Of-Interest-Free-Optimization.py`
### 1. KNN_class(finetuned_training_features, finetuned_validation_features)
This function performs K-Nearest Neighbors (KNN) classification to predict labels for features of validation samples while avoiding samples with conflicts of interest in terms of center labels from search space (finetuned_training_features).
### Parameters
`finetuned_training_features`: Training sample features extracted via the unlearning layer.
`finetuned_validation_features`: Validation sample features extracted via the unlearning layer.

### Returns:
`predictions`: predicted cancer labels for the validation samples.
### Example Usage: 
`predictions = KNN_class (finetuned_training_features, finetuned_validation_features)`

### Note:
- The `n_neighbors` parameter in the KNN classifier is set to `3` by default; this can be adjusted as needed.
- The function ensures that the training search space does not include samples from the same center as the validation samples to maintain fairness and avoid bias.

### 2. WeightOptimizationProblem
This class defines a custom optimization problem for tuning weights in a multi-objective optimization context using the `pymoo` library.

### Purpose:
The problem optimizes unlearning weights to achieve two conflicting objectives:
- Minimize the classification performance on center labels (to mitigate bias).
- Maximize the classification performance on cancer labels (to improve accuracy).

### Implementation: 
The class extends the `ElementwiseProblem` from the `pymoo` library and evaluates solutions by computing `F1 scores`:
**Key Components**
- **Initialization**:
   - `n_var`: Number of variables (weights to optimize).
   - `n_obj`: Number of objectives (2 in this case).
   - `xl` and `xu`: Lower and upper bounds for the weights (set to -1 and 1, respectively).
 
- **Evaluation (`_evaluate method`):**
    - Computes the `F1 scores` for center and cancer predictions.
    - Defines the objectives
        - `center_f1score` (to be minimized).
        - `-cancer_f1score` (negative value to maximize the cancer F1 score).
- **Additional Functions:**
  - `compute_cancer_f1_score`: Uses the `KNN_class` function to predict cancer labels on the validation dataset.
  - `compute_center_f1_score`: Uses typical form of `KNN` classifier.
