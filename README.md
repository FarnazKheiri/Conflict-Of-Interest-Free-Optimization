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
The dataset used for this project (a balanced subset of TCGA dataset) can be downloaded from the following link: [Dataset - Google Drive](https://drive.google.com/drive/folders/18FWc68M_XItQXtOUoIDemxwagLYTsIkg?usp=sharing)

**`Key Functions in Multi-objective-Conflict-Of-Interest-Free-Optimization.py`
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

