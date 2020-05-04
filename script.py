#!/usr/bin/env python
# coding: utf-8

# In[33]:


import dlib
import glob
import os
import sys
import numpy as np

predictor_path = 'shape_predictor_68_face_landmarks.dat'
faces_path = 'images'
file_types = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')

race_map = {'AM': 'Asian Male', 'AF': 'Asian Female', 'BM': 'Black Male', 'BF': 'Black Female', 'LM':'Latin Male', 'LF': 'Latin Female', 'WM': 'White Male', 'WF':'White Female'}


# In[2]:


cl_array = []
for label in os.listdir('faces'):
    cl = label.split("-")[0]
    if cl not in cl_array:
        cl_array.append(cl)


# In[3]:


print(cl_array)


# In[4]:


i = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

temp_y = np.zeros((1,len(cl_array)))
for label in os.listdir('faces'):
    cl = label.split("-")[0]
    for f in os.listdir('faces/' + label):
        if f.endswith(file_types):
            img = dlib.load_rgb_image('faces/' + label + '/' + f)
            dets = detector(img, 1)
            if(len(dets) == 1):
                temp_y[0][cl_array.index(cl)] = 1
                faces = dlib.full_object_detections()
                for detection in dets:
                    faces.append(predictor(img, detection))
                images = dlib.get_face_chips(img, faces, size=64)
                images = np.array(images)
                if i == 0:
                    X = images
                    y = temp_y
                    i = i + 1
                else:
                    temp = np.append(X, images, axis=0)
                    X = temp
                    temp_y_temp = np.append(y, temp_y, axis=0)
                    y = temp_y_temp
                temp_y = np.zeros((1,len(cl_array)))


# In[17]:


import pandas as pd
import keras
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_uniform
import scipy.misc
import tensorflow as tf
from matplotlib.pyplot import imshow
get_ipython().magic('matplotlib inline')
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# In[6]:


def identity_block(X, f, filters, stage, block):
    
        
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# In[7]:


def convolutional_block(X, f, filters, stage, block, s = 2):
        
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1,1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size = (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X


# In[8]:


def ResNet50(input_shape=(64, 64, 3), classes=9):

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2, 2), name="avg_pool")(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


# In[64]:


model = ResNet50(input_shape=(64, 64, 3), classes=len(cl_array))


# In[65]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[92]:


y = np.array(y.astype(int))
X = np.array(X.astype(int))


# In[94]:


X


# In[95]:


y


# In[67]:


X_train_orig, X_test_orig, Y_train, Y_test = train_test_split(X, y, test_size=0.02)

X_train = X_train_orig/255
X_test = X_test_orig/255


print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


# In[68]:


model.fit(X_train, Y_train, epochs=20, batch_size=64)


# In[69]:


preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[70]:


model.summary()


# In[71]:


model.save('demographic_resnet.h5')


# In[60]:


model = keras.models.load_model('demographic_resnet.h5')


# In[88]:


## giving a sample image to identify which race

test_image_path = 'faces/WF-241/CFD-WF-241-210-N.jpg'
if f.endswith(file_types):
    img = dlib.load_rgb_image(test_image_path)
    dets = detector(img, 1)
    if(len(dets) == 1):
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(predictor(img, detection))
        images = dlib.get_face_chips(img, faces, size=64)
        images = np.array(images)
        images = np.array(images.astype(int))
        y_pred_matrix = model.predict(images)
        y_pred_matrix = y_pred_matrix.tolist()
        maxi = np.amax(y_pred_matrix)
        matrix_list = list(y_pred_matrix)
        index_max = matrix_list[0].index(maxi) + 1
        y_pred = cl_array[index_max]
        print("The given person's image is " + race_map[y_pred])
    elif(len(dets) == 0):
        print('No faces were detected in the image')
    else:
        print('The photo contains multiple faces. The system currently does not support multiple faces')
else:
    print('The format not supported by this modal')


# In[89]:


## giving a sample image to identify which race

test_image_path = 'faces/BM-211/CFD-BM-211-174-N.jpg'
if f.endswith(file_types):
    img = dlib.load_rgb_image(test_image_path)
    dets = detector(img, 1)
    if(len(dets) == 1):
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(predictor(img, detection))
        images = dlib.get_face_chips(img, faces, size=64)
        images = np.array(images)
        images = np.array(images.astype(int))
        y_pred_matrix = model.predict(images)
        y_pred_matrix = y_pred_matrix.tolist()
        maxi = np.amax(y_pred_matrix)
        matrix_list = list(y_pred_matrix)
        index_max = matrix_list[0].index(maxi)
        y_pred = cl_array[index_max]
        print("The given person's image is " + race_map[y_pred])
    elif(len(dets) == 0):
        print('No faces were detected in the image')
    else:
        print('The photo contains multiple faces. The system currently does not support multiple faces')
else:
    print('The format not supported by this modal')


# In[91]:


## giving a sample image to identify which race

test_image_path = 'faces/AF-200/CFD-AF-200-228-N.jpg'
if f.endswith(file_types):
    img = dlib.load_rgb_image(test_image_path)
    dets = detector(img, 1)
    if(len(dets) == 1):
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(predictor(img, detection))
        images = dlib.get_face_chips(img, faces, size=64)
        images = np.array(images)
        images = np.array(images.astype(int))
        y_pred_matrix = model.predict(images)
        y_pred_matrix = y_pred_matrix.tolist()
        maxi = np.amax(y_pred_matrix)
        matrix_list = list(y_pred_matrix)
        index_max = matrix_list[0].index(maxi) + 1
        y_pred = cl_array[index_max]
        print("The given person's image is " + race_map[y_pred])
    elif(len(dets) == 0):
        print('No faces were detected in the image')
    else:
        print('The photo contains multiple faces. The system currently does not support multiple faces')
else:
    print('The format not supported by this modal')


# In[ ]:




