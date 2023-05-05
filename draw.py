import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import itertools

def draw_metrics(train_loss,train_acc,val_loss,val_acc,path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot(val_acc, label='Validation Accuracy')

    # 标记最终数据值
    plt.annotate('{:.3f}'.format(train_loss[-1]), xy=(len(train_loss)-1, train_loss[-1]), xytext=(-20, 10), textcoords='offset points', fontsize=10, color='blue')
    plt.annotate('{:.3f}%'.format(train_acc[-1]), xy=(len(train_acc)-1, train_acc[-1]), xytext=(-20, 10), textcoords='offset points', fontsize=10, color='orange')
    plt.annotate('{:.3f}'.format(val_loss[-1]), xy=(len(val_loss)-1, val_loss[-1]*100), xytext=(-20, 10), textcoords='offset points', fontsize=10, color='green')
    plt.annotate('{:.3f}%'.format(val_acc[-1]), xy=(len(val_acc)-1, val_acc[-1]*100), xytext=(-20, 10), textcoords='offset points', fontsize=10, color='red')

    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(path)
    plt.close()

def draw_confusion_matrix(cm, path,classes=['anger','disgust','fear','happy','sad','surprised','normal'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)
    plt.close()

def draw_hist(name, class_names=['anger','disgust','fear','happy','sad','surprised','normal']):

    # 文件夹路径
    folder_path = os.path.join('data',name)

    # 获取子文件夹名称列表
    subfolder_names = os.listdir(folder_path)
    subfolder_names = sorted(subfolder_names)
    print(subfolder_names)
    # 统计每个子文件夹中图片的数量
    image_counts = {}
    for subfolder_name in subfolder_names:
        # 构造完整的文件夹路径
        subfolder_path = os.path.join(folder_path, subfolder_name)
        # 获取该子文件夹中所有图片文件的名称列表
        image_names = os.listdir(subfolder_path)
        # 统计图片数量
        image_counts[subfolder_name] = len(image_names)

    # 绘制直方图
    plt.bar(range(len(image_counts)), list(image_counts.values()), align='center')
    plt.xticks(range(len(class_names)), class_names)
    #plt.xticks(range(len(image_counts)), list(image_counts.keys()))
    plt.title(name)
    plt.savefig('pic/'+ name + '_count_hist.png')
    plt.close()

def draw_data_augmentation():
    path_train = 'data/train'
    path_valid = 'data/val'
    transforms_train = transforms.Compose([
        transforms.Grayscale(),#使用ImageFolder默认扩展为三通道，重新变回去就行
        transforms.RandomHorizontalFlip(),#随机翻转
        transforms.ColorJitter(brightness=0.5, contrast=0.5),#随机调整亮度和对比度
        transforms.ToTensor()
    ])
    transforms_vaild = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    data_train = torchvision.datasets.ImageFolder(root=path_train,transform=transforms_train)
    data_vaild = torchvision.datasets.ImageFolder(root=path_valid,transform=transforms_vaild)

    for i in range(1,16+1):
        plt.subplot(4, 4, i)
        # 每次调用都会进行一次数据增强
        plt.imshow(data_train[1][0].squeeze(),cmap='Greys_r')
        plt.axis('off')
    plt.savefig('pic/data_augmentation')
    plt.close()

if __name__ == '__main__':
    draw_hist('train')
    draw_hist('val')
    draw_data_augmentation()