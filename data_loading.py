#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle


# ### Data Load 
# ### The path could be replaced with any other data path

# In[ ]:


#load data 
filename = './data/train_balanced_images.pickle'
with open(filename, 'rb') as file:
    train_balanced_images = pickle.load(file)
filename = './data/train_balanced_center_labels.pickle'
with open(filename, 'rb') as file:
    train_balanced_center_labels = pickle.load(file)
filename = './data/train_balanced_cancer_labels.pickle'
with open(filename, 'rb') as file:
    train_balanced_cancer_labels = pickle.load(file)


# In[ ]:


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


# In[ ]:


# upload pretrained model
path = "./pretrained_model" 
pretrained_model = load_model(path)


# In[ ]:


#  extract features
layer_output_model = tf.keras.Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[-2].output)
train_feature = layer_output_model.predict(training_data)
validation_feature = layer_output_model.predict(validation_data)
test_feature = layer_output_model.predict(test_data)
external_feature = layer_output_model.predict(external_images)


# In[ ]:


# Save extracted features
with open('./output/train_feature.pickle', 'wb') as f:
    pickle.dump(train_feature, f)
with open('./output/validation_feature.pickle', 'wb') as f:
    pickle.dump(validation_feature, f)
with open('./output/test_feature.pickle', 'wb') as f:
    pickle.dump(test_feature, f)
with open('./output/external_feature.pickle', 'wb') as f:
    pickle.dump(external_feature, f)


# In[ ]:




