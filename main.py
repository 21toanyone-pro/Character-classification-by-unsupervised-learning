from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms

from sklearn.mixture import GaussianMixture


from PIL import Image, ImageFilter
from PIL import ImageFont 
from PIL import ImageDraw 

import os
import cv2
import glob
import re
import csv
import pandas as pd
import natsort
import shutil
import pickle
import signal
import sys

digits = load_digits()

path = './img/'
path_list = os.listdir(path)
path_list = natsort.natsorted(path_list)

listlist =[]
ac = listlist.append #이미지 경로
chars = pd.DataFrame([])
faces = []
    
for i in path_list: # 왕 이름
    pathA = path + str(i) +'/'
    pathA_list = os.listdir(pathA)
    pathA_list = natsort.natsorted(pathA_list)
    for j in pathA_list: # 권 수
        pathB = pathA + str(j) +'/'
        pathB_list = os.listdir(pathB)
        pathB_list = natsort.natsorted(pathB_list)
        for k in pathB_list: # 장 수
            pathB_l = str(pathB)+str(k)+'/'
            pathC_list = os.listdir(pathB_l)
            pathC_list = natsort.natsorted(pathC_list)
            for l in pathC_list:
                img = Image.open(pathB_l+l)
                ac(pathB_l+l)
                img = img.resize((32, 32))
                img = np.array(img)
                img = img / 255
                char = pd.Series(img.flatten(),name=l)
                chars = chars.append(char)

from sklearn.decomposition import PCA
chars_pca = PCA(n_components=100) # 몇개의 주성분을 쓸 것인지
chars_pca.fit(chars) 

components = chars_pca.transform(chars) # 특징행렬을 낮은 차원의 근사행렬로 변환
projected = chars_pca.inverse_transform(components) #변환된 근사행렬을 원래의 차원으로 복귀

def filename(frame_id):
        return "%08d.jpg" % frame_id

def save_img(y_pred):
    label_ids = np.unique(y_pred) # 라벨이 저장
    #num_unique_faces = len(np.where(label_ids > -1)[0]) # 라벨의 갯수

    for label_id in label_ids:
        dir_name = "./clust/" % label_id
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name+'/')     
        indexes = np.where(y_pred == label_id)[0]
        for i in indexes:
            filename = "ID%d" % label_id + "-" + str(i)+'.jpg'
            pathname = os.path.join(dir_name, filename)
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            shutil.copy(listlist[i], pathname)

model_G = GaussianMixture(n_components=30,covariance_type='full',random_state=0, tol=0.1,max_iter=1000)
model_G.fit(components)
labels =model_G.predict(components)

save_img(labels)

