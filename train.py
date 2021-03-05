import os, glob, pickle
import pandas as pd
import numpy as np
import cv2

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping, Callback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


X = []
y = []

for path in glob.glob('./data/*'):
    species = os.path.basename(path)
    for breed in glob.glob(path +'/*'):
        # pull a simplified name
        if species == 'cat':
            breed_name = breed.split('/')[-1]
            for file in glob.glob(breed + '/*.jpg')[:350]: #limit to 350 per cat breed
                file_name = os.path.basename(file)
                # load file
                try:
                    size = (160, 160)
                    img_arr = cv2.imread(file)
                    #convert to color array
                    img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                    #reshape to uniform size 160x160x3
                    new_array = cv2.resize(img_rgb, size)
                    X.append(new_array)
                    y.append(1)
                except:
                    print("Invalid Path")
        else:
            breed_name = breed.split('-')[-1]
            for file in glob.glob(breed + '/*.jpg')[:150]: #limit to 150 per dog breed
                file_name = os.path.basename(file)
                # load file
                try:
                    size = (160, 160)
                    img_arr = cv2.imread(file)
                    #convert to color array
                    img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                    #reshape to uniform size 160x160x3
                    new_array = cv2.resize(img_rgb, size)
                    X.append(new_array)
                    y.append(0)
                except:
                    print("Invalid Path")

X = np.array(X)
y = np.array(y)
X = X.astype('float32')
y = utils.to_categorical(y, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9)

cnn = Sequential()
# Starting Layer
cnn.add(Conv2D(filters = 128,
               kernel_size = (3, 3),
               activation = 'relu',
               input_shape = (160, 160, 3)))
cnn.add(MaxPooling2D(pool_size = (2,2)))
#second convolutional layer
cnn.add(Conv2D(filters=64,            
                     kernel_size=(3, 3),        
                     activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

#third convolutional layer
cnn.add(Conv2D(filters=128,            
                     kernel_size=(3, 3),        
                     activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

#fourth convolutional layer
cnn.add(Conv2D(filters=32,            
                     kernel_size=(3, 3),        
                     activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

#fifth convolutional layer
cnn.add(Conv2D(filters=32,            
                     kernel_size=(3, 3),        
                     activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))


#flatten the metrics to fit into the Dense layers 
cnn.add(Flatten())
cnn.add(Dense(32, activation = 'relu'))
cnn.add(Dropout(0.3))
cnn.add(Dense(256, activation = 'relu'))
cnn.add(Dropout(0.3))
cnn.add(Dense(128, activation = 'relu'))
cnn.add(Dropout(0.3))
cnn.add(Dense(64, activation = 'relu'))
cnn.add(Dropout(0.3))
cnn.add(Dense(32, activation = 'relu'))
cnn.add(Dense(2, activation = 'sigmoid'))
cnn.compile(loss = 'binary_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])
early_stop = EarlyStopping(patience = 5, restore_best_weights = True)

res = cnn.fit(X_train, y_train,
             batch_size = 64,
             validation_data = (X_test, y_test),
             epochs = 30,
             callbacks = [early_stop],
             verbose = 1)

cnn.save('./model')