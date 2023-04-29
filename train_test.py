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

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7" 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# 定义一些全局变量
class_names=['anger','disgust','fear','happy','sad','surpriseds','normal']
num_classes = 7  



# 读取参数
parser = argparse.ArgumentParser(description='Training parameters')

parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
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

# train_num=train_generator.samples
# valid_num=valid_generator.samples
# print(train_num,valid_num)

model = get_model(width, height, num_classes)

train_generator, valid_generator, test_generator = get_data_generators(train_dir, valid_dir, test_dir, height, width, batch_size)

train_num=train_generator.samples
valid_num=valid_generator.samples
print(train_num,valid_num)


# 定义回调函数
# 创建一个回调函数 checkpoint_cb，用于在训练期间监控验证集的准确率，并在每次验证集准确率提高时保存最好的模型参数。
checkpoint_path = "best_model.h5"
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    checkpoint_path, 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False, 
    mode='max')

# 定义了一个 TensorBoard 回调函数，用于在训练过程中将模型的训练情况写入到 TensorBoard 中，方便进行可视化分析。
tensorboard_cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# 创建了一个CSVLogger对象，用于将模型在训练过程中的一些指标记录到一个CSV文件中，比如训练集和验证集的损失函数值和准确率。
csvlog_cb = keras.callbacks.CSVLogger('training_logs.csv')

history = model.fit(
    train_generator,
    steps_per_epoch=train_num//batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_num//batch_size,
    callbacks=[tensorboard_cb, csvlog_cb, checkpoint_cb])

# 在测试数据集上评估模型
loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)

# 打印测试准确率
print('Test accuracy:', accuracy)

#调用函数将模型和log文件存到model_logs文件夹对应的以准确率命名的子文件夹下
move_logs_and_model(round(accuracy, 3))