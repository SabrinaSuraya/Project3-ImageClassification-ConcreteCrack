# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:15:05 2022

@author: sabri
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers
from tensorflow.keras.models import Sequential
import datetime
import pathlib
import cv2


#1.Data Preparation
file_path= r"C:\Users\sabri\Documents\PYTHON\DL\Datasets\Concrete Crack Images for Classification"
data_dir= pathlib.Path(file_path)
#%%
image_count= len(list(data_dir.glob('*/*.jpg')))
print(image_count)


#%%
#Create dataset
SEED=12345
BATCH_SIZE=32
IMG_SIZE= (160, 160)

train_dataset= keras.utils.image_dataset_from_directory(
                                                        data_dir, 
                                                        validation_split= 0.3, 
                                                        subset='training', 
                                                        seed= SEED, 
                                                        shuffle= True,
                                                        image_size=IMG_SIZE, 
                                                        batch_size= BATCH_SIZE)

#%%
validation_dataset= keras.utils.image_dataset_from_directory(
                                                        data_dir, 
                                                        validation_split= 0.3, 
                                                        subset='validation', 
                                                        seed= SEED, 
                                                        shuffle= True,
                                                        image_size=IMG_SIZE, 
                                                        batch_size= BATCH_SIZE)

#%%
class_names= train_dataset.class_names
print(class_names)

#%%

plt.figure(figsize=(10,10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax= plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

#%%
'''
for image_batch, label_batch in train_dataset:
    print(image_batch.shape)
    print(label_batch.shape)
 '''   
#%%

AUTOTUNE= tf.data.AUTOTUNE

train_dataset= train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset= validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

#Data Preparation is DONE

#%%
#2. Create data augmentation pipeline
data_augmentation= keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))


for images, labels in train_dataset.take(1):
    first_image= images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image= data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')


#%%
#Create a layer for data preprocessing
preprocess_input= applications.mobilenet_v2.preprocess_input

#Create teh base model by using MobileNetV2
IMG_SHAPE= IMG_SIZE + (3,)
base_model= applications.MobileNetV2(input_shape= IMG_SHAPE, include_top= False, weights='imagenet')

#Apply layer freezing
for layer in base_model.layers[:100]:
    layer.trainable= False
    
base_model.summary()
#%%
#Create classification layer
nClass= len(class_names)

global_avg_pooling= layers.GlobalAveragePooling2D()
output_layer= layers.Dense(nClass, activation='softmax')

#%%
#Use Functioonal API to construct teh entire model
inputs= keras.Input(shape= IMG_SHAPE)
x= data_augmentation(inputs)
x= preprocess_input(x)
x= base_model(x)
x= global_avg_pooling(x)
x= keras.layers.Dropout(0.2)(x)
outputs= output_layer(x)

model= keras.Model(inputs= inputs, outputs= outputs)
model.summary()
#%%

#Compile teh model
optimizer= keras.optimizers.Adam(learning_rate= 0.0001)
loss= keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer= optimizer, loss= loss, metrics=['accuracy'])
#%%

#Perform model training
EPOCHS=10

base_log_path = r"C:\Users\sabri\Documents\PYTHON\DL\TensorBoard\tb_logs"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ '___Project3')
tb = keras.callbacks.TensorBoard(log_dir=log_path)

history = model.fit(train_dataset,validation_data=validation_dataset, epochs=EPOCHS,callbacks=[tb])
#%%
#Make prediction from 
#Display image
file_prediction_path= r"C:\Users\sabri\Documents\PYTHON\DL\Prediction\Concrete- crackORnot\Concrete.jpg"
predict_concrete= cv2.imread(file_prediction_path)
predict_concrete=  cv2.resize(predict_concrete, (256, 256))
cv2.imshow("Predict Concrete", predict_concrete)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%

img= tf.keras.utils.load_img(
                            file_prediction_path,
                            target_size= IMG_SIZE
                            )

img_array= tf.keras.utils.img_to_array(img)
img_array= tf.expand_dims(img_array, 0) #Create a batch

predictions= model.predict(img_array)
score= tf.nn.softmax(predictions[0])

print("This image most likely belongs to {} with a {:2f} percent confidence". format(class_names[np.argmax(score)], 100*np.max(score)))


















