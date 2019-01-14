
# coding: utf-8

# In[4]:


import os
import csv
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
get_ipython().run_line_magic('matplotlib', 'inline')

samples = []
with open('./data_forw/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_back/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_turn/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_turn1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data_recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

print("train.len ",len(train_samples))
print("valid.len ",len(validation_samples))
#print("train_iamge.len ",train_samples[0].shape)
#print("valid_image.len ",validation_samples[0].shape)

    


# In[6]:


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                name = batch_sample[1]
                left_image = cv2.imread(name)
                left_angle = float(batch_sample[3])+0.2
                images.append(left_image)
                angles.append(left_angle)
                
                name = batch_sample[2]
                right_image = cv2.imread(name)
                right_angle = float(batch_sample[3])+0.2
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#X_train_play, y_train_play=next(train_generator)
#print("train.shape ",X_train_play.shape)


#fig,axis=plt.subplots(1,3,figsize=(9,9))
#axis=axis.ravel()
#for i in range(3):
#    index=random.randint(0,len(X_train_play))
#    axis[i].imshow(X_train_play[index])


# In[ ]:



ch, row, col = 3, 80, 320  # Trimmed image format
from keras.models import Sequential,Model
from keras.layers import Cropping2D,Convolution2D,Flatten,Dense,Lambda,Dropout

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
          
          
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(3*train_samples), 
                    validation_data=validation_generator, nb_val_samples=len(3*validation_samples), nb_epoch=3)
model.save('model.h5')  

