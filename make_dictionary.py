# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import progressbar
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
import time
#from keras.utils import np_utils

VOCA_SIZE = 1000
IMAGE_PATH = 'images'
HEAD_PATH = 'heads'
HESS_THRES = 10

prev_time = time.time()
surf = cv2.xfeatures2d.SURF_create()
dictionary = MiniBatchKMeans(n_clusters=VOCA_SIZE, batch_size=500, init_size=2*VOCA_SIZE)

dsc = []
path = './'+HEAD_PATH
num_files = len(os.listdir(path))
bar = progressbar.ProgressBar(maxval=num_files).start()
for i, img in enumerate(os.listdir(path)):
    image = cv2.imread(os.path.join('./', IMAGE_PATH, img), cv2.IMREAD_GRAYSCALE)    #!!!
    if image is None:
        continue
    keypoints, descriptors = surf.detectAndCompute(image, None)
    if descriptors is None:
        continue
    dsc.extend(descriptors)
    bar.update(i)
bar.finish()

print('Got keypoints of all images. Time: ', str(time.time()-prev_time))
prev_time = time.time()

dictionary.fit(np.array(dsc))
filename = 'image_dict_' + str(VOCA_SIZE) + '.dat'  #!!!
joblib.dump(dictionary, os.path.join('./models/', filename))
print('Finished dictionary. Time: ', str(time.time()-prev_time))
