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
import argparse
import shutil
from model import get_model
from utils import move_logs_and_model
from data_augmentation import get_data_generators

# 显存按需分配，即 TensorFlow 运行时会根据需要逐步分配显存，避免一次性占用所有显存。
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# 定义一些全局变量
class_names=['anger','disgust','fear','happy','sad','surpriseds','normal']
num_classes = 7  



# 读取参数
parser = argparse.ArgumentParser(description='Training parameters')

parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--height', type=int, default=48, help='image height')
parser.add_argument('--width', type=int, default=48, help='image width')
parser.add_argument('--channels', type=int, default=1, help='Number of image channels')
parser.add_argument('--train_dir', type=str, default='data/train', help='Training data directory')
parser.add_argument('--valid_dir', type=str, default='data/val', help='Validation data directory')
parser.add_argument('--test_dir', type=str, default='data/test', help='Testing data directory')
args = parser.parse_args()

train_dir = args.train_dir
valid_dir = args.valid_dir
test_dir = args.test_dir

channels = args.channels
height = args.height
width = args.width

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size

train_generator, valid_generator, test_generator = get_data_generators(train_dir, valid_dir, test_dir, height, width, batch_size)

train_num=train_generator.samples
valid_num=valid_generator.samples
print(train_num,valid_num)
print(train_generator[0][0].shape)