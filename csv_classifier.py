import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import sys
import time
import tensorflow as tf
from tensorflow import keras
import scipy.misc as sm
import scipy
from PIL import Image
import csv

#将前面处理好的CSV格式的数据集转换成图像数据集
datasets_path = 'data/'
train_csv = os.path.join(datasets_path, 'train.csv')
val_csv = os.path.join(datasets_path, 'val.csv')
#test_csv = os.path.join(datasets_path, 'test.csv')
train_set = os.path.join(datasets_path, 'train')
val_set = os.path.join(datasets_path, 'val')
#test_set = os.path.join(datasets_path, 'test')
for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv)]:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num = 1
    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        #读取标签和图像数据并储存成48*48的矩阵
        for i, (label, pixel) in enumerate(csvr):
            pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
            subfolder = os.path.join(save_path, label)
            #print(label)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            #将这个NumPy数组转换为PIL的图像格式。然后，.convert('L')将图像转换为灰度图像
            im = Image.fromarray(pixel).convert('L')
            image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
            #print(image_name)
            im.save(image_name)