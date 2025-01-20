#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import numpy as np
import pickle
import pdb

from numpy.random import rand
from numpy import argmax

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, categorical_crossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


# In[2]:


#load
filename = './data/shuffled_train_reshaped_balanced_images.pickle'
with open(filename, 'rb') as file:
    shuffled_train_reshaped_balanced_images = pickle.load(file)
filename = './data/shuffled_train_balanced_center_labels.pickle'
with open(filename, 'rb') as file:
    shuffled_train_balanced_center_labels = pickle.load(file)
filename = './data/shuffled_train_balanced_cancer_labels.pickle'
with open(filename, 'rb') as file:
    shuffled_train_balanced_cancer_labels = pickle.load(file)


# In[3]:


#load
filename = './data/external_images.pickle'
with open(filename, 'rb') as file:
    external_images = pickle.load(file)
filename = './data/external_center_labels.pickle'
with open(filename, 'rb') as file:
    external_center_labels = pickle.load(file)
filename = './data/external_cancer_labels.pickle'
with open(filename, 'rb') as file:
    external_cancer_labels = pickle.load(file)


# In[4]:


training_size = int(len(shuffled_train_balanced_center_labels) * 0.8)   # 80% of the array
validation_size = int(len(shuffled_train_balanced_center_labels) * 0.1)   # 10% of the array

# Split the array into three parts
# training 
training_data = shuffled_train_reshaped_balanced_images[:training_size]
training_cancer_labels = shuffled_train_balanced_cancer_labels[:training_size]
training_center_labels = shuffled_train_balanced_center_labels[:training_size]
# training_slide_names = shuffled_train_balanced_slidenames[:training_size]

#validation
validation_data = shuffled_train_reshaped_balanced_images[training_size:training_size+validation_size]
validation_cancer_labels = shuffled_train_balanced_cancer_labels[training_size:training_size+validation_size]
validation_center_labels = shuffled_train_balanced_center_labels[training_size:training_size+validation_size]
# validation_slide_names = shuffled_train_balanced_slidenames[training_size:training_size+validation_size]

#test
test_data = shuffled_train_reshaped_balanced_images[training_size+validation_size:]
test_cancer_labels = shuffled_train_balanced_cancer_labels[training_size+validation_size:]
test_center_labels = shuffled_train_balanced_center_labels[training_size+validation_size:]
# test_slide_names = shuffled_train_balanced_slidenames[training_size:training_size+validation_size]


# In[5]:


path = "./data/EfModel"
pretrained_model = load_model(path)


# In[6]:


layer_output_model = tf.keras.Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[-2].output)
train_feature = layer_output_model.predict(training_data)
validation_feature = layer_output_model.predict(validation_data)
test_feature = layer_output_model.predict(test_data)
external_feature = layer_output_model.predict(external_images)


# ### before MTL 

# In[7]:


# cancer classification
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(train_feature, training_cancer_labels)
cancer_y_pred = classifier.predict(test_feature)

metrics.f1_score(cancer_y_pred,test_cancer_labels ,average='micro')


# In[8]:


# External cancer classification
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(train_feature, training_cancer_labels)
cancer_y_pred = classifier.predict(external_feature)
# metrics.accuracy_score(cancer_y_pred,external_cancer_labels)
metrics.f1_score(cancer_y_pred,external_cancer_labels ,average='micro')


# In[9]:


# center classification
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(train_feature, training_center_labels)
cancer_y_pred = classifier.predict(validation_feature)
metrics.f1_score(cancer_y_pred,validation_center_labels,average='micro')


# # MTL

# In[10]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

encoded_center_labels = label_encoder.fit_transform(training_center_labels)


# In[11]:


encoded_cancer_labels = np.argmax(training_cancer_labels, axis=1)


# In[12]:


num_classes_task1 = 2       # Number of classes for cancer types
num_classes_task2 = 3        # Number of classes for data centers


# In[13]:


def create_shared_backbone(input_shape):
    
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Dense(8, activation="relu")(inputs)
#     x = tf.keras.layers.Dense(512, activation="relu")(x)
#     x = tf.keras.layers.Dense(256, activation="relu")(x)
#     x = tf.keras.layers.Dense(4, activation="relu")(x)    
    return inputs, x


# In[14]:


input_shape = train_feature[0].shape 

# extract shared features
inputs, shared_feature = create_shared_backbone(input_shape)

# task1: cancer classification
cancer_task_output = tf.keras.layers.Dense(num_classes_task1, activation="softmax", name = "cancer_task_output")(shared_feature)

# task2: center classification
center_task_output = tf.keras.layers.Dense(num_classes_task2, activation="softmax", name = "center_task_output")(shared_feature)


# In[15]:


multi_task_model = Model(inputs = inputs, outputs = [cancer_task_output,center_task_output])


# In[16]:


import tensorflow as tf
from tensorflow.keras import backend as K

def f1_score_metric(y_true, y_pred):
    
    # Convert y_pred probabilities to class predictions
    y_pred = tf.argmax(y_pred, axis=-1)
    # Ensure y_true and y_pred have the same data type
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Compute true positives, false positives, and false negatives
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)



# In[17]:


multi_task_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'cancer_task_output': "SparseCategoricalCrossentropy",
        'center_task_output': "SparseCategoricalCrossentropy",
    },
    metrics={
        'cancer_task_output': [SparseCategoricalAccuracy(), f1_score_metric],
        'center_task_output': [SparseCategoricalAccuracy(), f1_score_metric],
    },
    loss_weights={'cancer_task_output': 1.0, 'center_task_output': -1.0}  
)


# In[ ]:


multi_task_model.summary()


# In[18]:


multi_task_model.fit(train_feature, {'cancer_task_output': encoded_cancer_labels, 'center_task_output': encoded_center_labels},
          epochs=10, batch_size=64, validation_split=0.1)


# In[ ]:


train_feature.shape


# In[19]:


mtl_layer_output_model = tf.keras.Model(inputs=multi_task_model.input, outputs=multi_task_model.layers[-3].output)

mtl_train_features = mtl_layer_output_model.predict(train_feature)
mtl_test_features = mtl_layer_output_model.predict(test_feature)
mtl_validation_feature = mtl_layer_output_model.predict(validation_feature)
mtl_external_feature = mtl_layer_output_model.predict(external_feature)


# In[60]:


finetuned_weights = multi_task_model.layers[-3].get_weights()
###save finetuned weights in size of [256,8]
weight_path= './data/multitask_learning_weight.pickle'
pickle.dump(finetuned_weights, open(weight_path, 'wb'))


# ### After MTL

# In[20]:


# internal cancer classification => test in train
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(mtl_train_features, training_cancer_labels)
cancer_y_pred = classifier.predict(mtl_test_features)
metrics.f1_score(cancer_y_pred,test_cancer_labels,average='micro')


# In[21]:


# external cancer classification => test in train
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(mtl_train_features, training_cancer_labels)
cancer_y_pred = classifier.predict(mtl_external_feature)
metrics.f1_score(cancer_y_pred,external_cancer_labels,average='micro')


# In[ ]:


# center classification
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(mtl_train_features, training_center_labels)
cancer_y_pred = classifier.predict(mtl_validation_feature)
metrics.accuracy_score(cancer_y_pred,validation_center_labels)


# In[ ]:


# center classification after adv
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(mtl_train_features, training_center_labels)
center_y_pred = classifier.predict(mtl_validation_feature)
print("f1_score:",metrics.f1_score(center_y_pred,validation_center_labels,average='weighted'))
print("accuracy:",metrics.accuracy_score(center_y_pred,validation_center_labels))


# In[ ]:




