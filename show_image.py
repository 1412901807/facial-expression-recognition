import os
import numpy as np
from tensorflow import keras
import argparse
from data_augmentation import get_data_generators
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
# 获取数据生成器
train_generator, _, _ = get_data_generators(train_dir, valid_dir, test_dir, height, width, batch_size)
save_path = './image'

# 保存前10个批次中的图像
for i in range(train_generator.samples//batch_size):
    # 获取一个批次的图像和标签
    images, labels = next(train_generator)
    # 遍历批次中的每张图像
    for j in range(len(images)):
        # 将图像像素值从[0, 1]范围变换为[0, 255]范围
        img = (images[j] * 255).astype(np.uint8)
        # 获取图像对应的标签
        label = np.argmax(labels[j])
        # 构造保存文件名，格式为 label_index.jpg
        filename = str(label) + '_' + str(i * train_generator.batch_size + j) + '.jpg'
        # 保存图像到文件夹中
        filepath = os.path.join(save_path, filename)
        keras.preprocessing.image.save_img(filepath, img)