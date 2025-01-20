#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


cancer_types = 2
center_types = 3
input_dim = 256

shared_feature_size = 8


# In[ ]:


label_encoder = LabelEncoder()

encoded_center_labels = label_encoder.fit_transform(training_center_labels)


# In[ ]:


encoded_cancer_labels = np.argmax(training_cancer_labels, axis=1)


# In[ ]:


# Gradient Reversal Layer (GRL)

class GradientReversalLayer(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, alpha=1.0):
        @tf.custom_gradient
        def reverse_gradients(x):
            def grad(dy):
                return -alpha*dy
            return x, grad
        return reverse_gradients(inputs)
        


# In[ ]:


# define primary model for cancer classifcation
def primary_model_fn(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(8, activation='relu', name="shared_features")(inputs)
    shared_features = x  # Save shared features
    outputs = layers.Dense(cancer_types, activation='softmax')(x)
    return models.Model(inputs, [shared_features, outputs])


# In[ ]:


## Define Adversary Model for center clasification
def adversary_model_fn(shared_feature_size):
    
    inputs = layers.Input(shape = (shared_feature_size,))
    x = GradientReversalLayer()(inputs)  
    outputs = layers.Dense(3, activation="softmax")(x) 
    return models.Model(inputs,outputs)


# In[ ]:


## initialize model 
primary_model= primary_model_fn(input_dim)
adversary_model = adversary_model_fn(shared_feature_size)


# In[ ]:


# Combined Training Loop
optimizer = Adam(learning_rate= 0.001)
alpha = 1.0
epochs = 100

for epoch in range(epochs):
    with tf.GradientTape(persistent=True) as tape:
        
        # forward pass
        
        shared_features, main_prediction = primary_model(train_feature)
       
        adversary_prediction = adversary_model(shared_features)
        
        
        # Compute Losses
        task_loss = CategoricalCrossentropy()(training_cancer_labels,main_prediction)
        adversary_loss = CategoricalCrossentropy()(new_training_center_labels,adversary_prediction)
        total_loss = task_loss - alpha * adversary_loss  # Min-max optimization

        
        
    # Compute Gradients
    grads_main = tape.gradient(total_loss, primary_model.trainable_variables)
    grads_adv = tape.gradient(adversary_loss, adversary_model.trainable_variables)
    
    
    
    # Apply Gradients
    optimizer.apply_gradients(zip(grads_main, primary_model.trainable_variables))
    optimizer.apply_gradients(zip(grads_adv, adversary_model.trainable_variables))
    
    print(f"Epoch {epoch + 1}, Task Loss: {task_loss.numpy():.4f}, Adversary Loss: {adversary_loss.numpy():.4f}")

        


# In[ ]:


adl_train_features, _ = primary_model(train_feature)
adl_test_features, _ = primary_model(test_feature)
adl_validation_features, _ = primary_model(validation_feature)
adl_external_features, _ = primary_model(external_feature)


# In[ ]:


# internal cancer classification => test in train
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(adl_train_features, training_cancer_labels)
cancer_y_pred = classifier.predict(adl_test_features)
metrics.f1_score(cancer_y_pred,test_cancer_labels,average='weighted')


# In[ ]:


# external cancer classification => test in train
k = 3
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(adl_train_features, training_cancer_labels)
cancer_y_pred = classifier.predict(adl_test_features)
metrics.f1_score(cancer_y_pred,adl_external_features,average='weighted')

