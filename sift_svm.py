# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
import time
import progressbar
from sklearn.externals import joblib
#from keras.utils import np_utils

def get_histogram (dictionary, desc):
    f = np.zeros(dictionary.n_clusters)
    pred = dictionary.predict(desc)
    for p in pred:
        f[p] += 1
    return f

VOCA_SIZE = 1000
HESS_THRES = 400
prev_time = time.time()
IMAGE_PATH = './images'
HEAD_PATH = './heads'

surf = cv2.xfeatures2d.SURF_create()
image_dict = joblib.load('./models/image_dict_1000.dat')
head_dict = joblib.load('./models/head_dict_1000.dat')
print('Loaded dictionaries')

features = []
labels = []
names = []

print('Calculating histograms')
num_files = len(os.listdir(HEAD_PATH))
bar = progressbar.ProgressBar(maxval=num_files).start()
for i, img in enumerate(os.listdir(HEAD_PATH)):
    bar.update(i)
    image = cv2.imread(os.path.join(IMAGE_PATH, img), cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    kp, d = surf.detectAndCompute(image, None)
    if d is None:
        continue
    img_f = get_histogram(image_dict, d)
        
    image = cv2.imread(os.path.join(HEAD_PATH, img), cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    kp, d = surf.detectAndCompute(image, None)
    if d is None:
        continue
    head_f = get_histogram(head_dict, d)
        
    features.append(np.concatenate((img_f, head_f), axis=None))
    labels.append(img.split('_')[0])
    if img[0].isupper():
        names.append('cat')
    else:
        names.append('dog')
    
bar.finish()
print('Got all histograms. Time: ', str(time.time()-prev_time))
prev_time = time.time()

print('Training...')
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels) #!!!
#encoded_labels = encoder.fit_transform(names)
dummy_labels = encoded_labels
length = len(features)
features = np.array(features).reshape(length, VOCA_SIZE*2)
dummy_labels = np.array(dummy_labels)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, dummy_labels, test_size=0.2)

model = svm.SVC(gamma='scale', decision_function_shape='ovo')
model.fit(Xtrain, Ytrain)

print('SVM trained. Time: ', str(time.time()-prev_time))
prev_time = time.time()

joblib.dump(model, './models/svm_model_1000_label.dat')
predict = model.predict(Xtest)
print(accuracy_score(predict, Ytest))