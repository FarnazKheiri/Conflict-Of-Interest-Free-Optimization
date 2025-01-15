#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pickle


from numpy import argmax
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

## optimization packages
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


# ### load Data 

# In[3]:


#load
filename = './data/train_balanced_images.pickle'
with open(filename, 'rb') as file:
    train_balanced_images = pickle.load(file)
filename = './data/train_balanced_center_labels.pickle'
with open(filename, 'rb') as file:
    train_balanced_center_labels = pickle.load(file)
filename = './data/train_balanced_cancer_labels.pickle'
with open(filename, 'rb') as file:
    train_balanced_cancer_labels = pickle.load(file)
    


# In[4]:


#load external data
filename = './data/external_images.pickle'
with open(filename, 'rb') as file:
    external_images = pickle.load(file)
filename = './data/external_center_labels.pickle'
with open(filename, 'rb') as file:
    external_center_labels = pickle.load(file)
filename = './data/external_cancer_labels.pickle'
with open(filename, 'rb') as file:
    external_cancer_labels = pickle.load(file)


# ### Data Spilit

# In[ ]:


# First split: training and temporary (for validation and test)
train_data, temp_data, train_cancer_labels, temp_cancer_labels, train_center_labels, temp_center_labels = train_test_split(
    train_balanced_images,
    train_balanced_cancer_labels,
    train_balanced_center_labels,
    test_size=0.2  # 20% for validation and test
)

# Second split: validation and test
validation_data, test_data, validation_cancer_labels, test_cancer_labels, validation_center_labels, test_center_labels = train_test_split(
    temp_data,
    temp_cancer_labels,
    temp_center_labels,
    test_size=0.5  # Split the remaining 20% into two halves
)


# ### load pretrained model

# In[6]:


path = "./pretrained_model" 
pretrained_model = load_model(path)


# In[7]:


#  extract features
layer_output_model = tf.keras.Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[-2].output)
train_feature = layer_output_model.predict(training_data)
validation_feature = layer_output_model.predict(validation_data)
test_feature = layer_output_model.predict(test_data)
external_feature = layer_output_model.predict(external_images)


# In[ ]:


# cancer classification
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(train_feature, training_cancer_labels)
cancer_y_pred = classifier.predict(test_feature)
# metrics.accuracy_score(cancer_y_pred,test_cancer_labels)
metrics.f1_score(cancer_y_pred,test_cancer_labels ,average='weighted')


# In[ ]:


# External cancer classification
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(train_feature, training_cancer_labels)
cancer_y_pred = classifier.predict(external_feature)
# metrics.accuracy_score(cancer_y_pred,external_cancer_labels)
metrics.f1_score(cancer_y_pred,external_cancer_labels,average='weighted')


# ### Finetuning parameters 

# In[10]:


# pretrained
pretarined_output = pretrained_model.layers[-2].output
input_size = pretarined_output.shape[1]

# Fine-tuning layers
# Dense128 & Dense2
layer_nodes = [train_feature.shape[1],8]

num_weights = sum( layer_nodes[idx]*layer_nodes[idx+1] for idx in range(len(layer_nodes )-1))


# In[11]:


def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result


# In[12]:


def feature_extractor(data,sol_weight):
    features = [] 
    for sample_idx in range(data.shape[0]):
        r1 = data[sample_idx,:]

        r1 = np.matmul(r1, sol_weight)
        r1 = relu(r1)         
        features.append(r1)  
    return features


# ### Define the Optimization Problem

# In[13]:


import numpy as np
from pymoo.core.problem import ElementwiseProblem

class WeightOptimizationProblem(ElementwiseProblem):
    def __init__(self, num_weights):
        super().__init__(n_var=num_weights,  
                         n_obj=2,            
                         n_constr=0,         
                         xl=-1*(np.ones(num_weights)),  # Lower bounds of weights
                         xu=np.ones(num_weights))   # Upper bounds of weights

    def _evaluate(self, weights_vector, out, *args, **kwargs):
        weights_mat = weights_vector.reshape(layer_nodes[0], layer_nodes[1])
        
        finetuned_training_features = np.array(feature_extractor(train_feature,weights_mat))
        finetuned_validation_features = np.array(feature_extractor(validation_feature,weights_mat))    
        
        cancer_f1score = compute_cancer_f1_score(finetuned_training_features,finetuned_validation_features) 
        center_f1score = compute_center_f1_score(finetuned_training_features,finetuned_validation_features) 

        out["F"] = [center_f1score, -cancer_f1score]

def compute_cancer_f1_score(finetuned_training_features,finetuned_validation_features):
    validation_cancer_pred = KNN_class(finetuned_training_features,finetuned_validation_features)
    cancer_f1score = metrics.f1_score(validation_cancer_labels,validation_cancer_pred, average='weighted')
    return cancer_f1score  

def compute_center_f1_score(finetuned_training_features,finetuned_validation_features):
    classifier = KNeighborsClassifier(n_neighbors = 3)
    classifier.fit(finetuned_training_features, training_center_labels)
    validation_center_pred = classifier.predict(finetuned_validation_features)
    center_f1score = metrics.f1_score(validation_center_labels,validation_center_pred, average='weighted')
    return center_f1score  


# In[14]:


def KNN_class(finetuned_training_features, finetuned_validation_features):
    # Initialize an empty list to store the predictions
    predictions = []

    # Find unique center labels in the validation set
    unique_center_labels = np.unique(validation_center_labels)

    # Iterate over each unique center label
    for center_label in unique_center_labels:
        # Filter training samples that do not have the same center label as the current center label
        mask = training_center_labels != center_label
        mask = mask.reshape(-1)

        filtered_training_features = finetuned_training_features[mask]
        filtered_training_labels = training_cancer_labels[mask]
        
        # Create a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=3)  # Adjust n_neighbors as needed
        knn.fit(filtered_training_features, filtered_training_labels)

        
        # Find all validation samples that have the current center label
        validation_mask = validation_center_labels == center_label
        validation_mask = validation_mask.reshape(-1)
        validation_samples = finetuned_validation_features[validation_mask]


        predicted_labels = knn.predict(validation_samples)

        # Store the predicted labels
        predictions.extend(predicted_labels)


    predictions = np.array(predictions)
    return predictions


# In[ ]:


# number of weights
num_weights = layer_nodes[0]*layer_nodes[1] 

# Create the problem object with the specified number of weights
problem = WeightOptimizationProblem(num_weights)
algorithm = NSGA2(
    pop_size=200,
    eliminate_duplicates=True
)

result = minimize(problem,
                  algorithm,
                  termination=('n_gen', 100),
                  seed=1,
                  save_history=False,
                  verbose=True)


# In[ ]:




Scatter().add(result.F).show()


# In[ ]:


min_index_center = np.argmin(result.F[:, 0])
min_index_cancer = np.argmin(result.F[:, 1])


# In[ ]:


best_weight_vector = result.X[min_index_center]
best_weight_mat = best_weight_vector.reshape(layer_nodes[0], layer_nodes[1])


# In[ ]:


## Extract Unleared features
best_finetuned_train_features = np.array(feature_extractor(train_feature,best_weight_mat))

best_finetuned_validation_features = np.array(feature_extractor(validation_feature,best_weight_mat))

best_finetuned_external_features = np.array(feature_extractor(external_feature,best_weight_mat))

best_finetuned_test_features = np.array(feature_extractor(test_feature,best_weight_mat))


# ## After Optimization Results

# In[ ]:


# internal cancer classification => test in train
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(best_finetuned_train_features, training_cancer_labels)
cancer_y_pred = classifier.predict(best_finetuned_test_features)
metrics.f1_score(cancer_y_pred,test_cancer_labels,average='weighted')


# In[ ]:


# external cancer classification => test in train
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(best_finetuned_train_features, training_cancer_labels)
cancer_y_pred = classifier.predict(best_finetuned_external_features)
metrics.f1_score(cancer_y_pred,external_cancer_labels,average='weighted')

