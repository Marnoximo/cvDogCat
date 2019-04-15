# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import random
#from keras.utils import np_utils

def get_histogram (dictionary, desc):
    f = np.zeros(dictionary.n_clusters)
    pred = dictionary.predict(desc)
    for p in pred:
        f[p] += 1
    return f

VOCA_SIZE = 1000
HESS_THRES = 400
IMAGE_PATH = './images'
HEAD_PATH = './heads'

surf = cv2.xfeatures2d.SURF_create()
image_dict = joblib.load('./models/image_dict_1000.dat')
head_dict = joblib.load('./models/head_dict_1000.dat')
model = joblib.load('./models/svm_model_1000.dat')
encoder = joblib.load('./models/name_encoder.dat')
print('Loaded model')


####################################
while True:
    file_paths = os.listdir(HEAD_PATH)
    idx = random.randint(0, len(file_paths) - 1)
    image = cv2.imread(os.path.join(IMAGE_PATH, file_paths[idx]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    kp, d = surf.detectAndCompute(image, None)
    if d is None:
        continue
    img_f = get_histogram(image_dict, d)
    
    head = cv2.imread(os.path.join(HEAD_PATH, file_paths[idx]), cv2.IMREAD_GRAYSCALE)
    if head is None:
        continue
    kp, d = surf.detectAndCompute(head, None)
    if d is None:
        continue
    head_f = get_histogram(head_dict, d)
    
    f = np.concatenate((img_f, head_f), axis = None)
    if file_paths[idx][0].isupper():
        name = 'cat'
    else:
        name = 'dog'
    pred = model.predict([f])
    class_truth = encoder.transform
    print('True class is: ', name)
    print('Predict as: ', encoder.inverse_transform(pred))
    cv2.imshow('Image', image)
    key = cv2.waitKey(0)
    if key == ord('x'):
        break

cv2.destroyAllWindows()# -*- coding: utf-8 -*-

