# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.utils import np_utils

BBOX_PATH = './boundbox.csv'
IMAGE_PATH = './images'
ANNO_PATH = './annotations'
XML_PATH = '/xmls'
TRIMAP_PATH = '/trimaps'

HEAD_PATH = './heads'
BBOX_VALUES = ['xmin', 'xmax', 'ymin', 'ymax']

# Read bounding box csv
bbx_df = pd.read_csv(BBOX_PATH)

x = []
breeds = []
family = []
def crop_head(path, bbx_df):
    for index,p in enumerate(bbx_df['filename']):
        print(index, ": ", p)
        image = cv2.imread(os.path.join(path, p))
        info = bbx_df[bbx_df['filename']==p][BBOX_VALUES].iloc[0]
        head_roi = image[info['ymin']:info['ymax'],info['xmin']:info['xmax']]
        cv2.imwrite(os.path.join(HEAD_PATH, p), head_roi)
        
        
def input_image(path, x, breeds, family):
    #for index,p in enumerate(os.listdir(path)):
    for index,p in enumerate(bbx_df['filename']):
        print(index, ': ', p)
        im = cv2.imread(os.path.join(path, p))
        brd = p.split('_')[0]
        if brd[0].isupper():
            family.append(1)
        else:
            family.append(0)
        breeds.append(brd)
        image = cv2.resize(im, (128,128))
        x.append(image)
        
        
input_image(HEAD_PATH, x, breeds, family)
        
x = np.array(x).reshape(-1,128,128,3)
x = x/255.
family = np.array(family)
breeds = np.array(breeds)
encoder = LabelEncoder()
encoder.fit(family) #asdasd
encoded_brd = encoder.transform(family) #asdasd
dummy_brd = np_utils.to_categorical(encoded_brd)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, dummy_brd, test_size=0.2, random_state=157)

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=Xtrain.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(2, activation='sigmoid')) #35

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(Xtrain, Ytrain, epochs=10, batch_size=40, verbose=1, validation_split=0.2)

predictions = model.predict(Xtest)
pred = np.around(predictions)

print('Accuracy on test set: ', accuracy_score(Ytest, pred))