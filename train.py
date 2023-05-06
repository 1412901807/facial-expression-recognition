import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
import matplotlib.pyplot as plt
from model import vggnet,resnet
from data_augmentation import get_data_generators
from sklearn.metrics import confusion_matrix
from draw import draw_confusion_matrix,draw_metrics
import argparse

# 指定gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

def train(model, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        print('epoch%d train loss: %.3f train acc: %.3f%%' % (epoch + 1, train_loss, train_acc*100))
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        

        # 验证
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            valid_loss = valid_loss / len(valid_loader)
            valid_acc = correct / total 
            print('[%d] valid loss: %.3f valid acc: %.3f%%' % (epoch + 1, valid_loss, valid_acc*100))
            valid_loss_history.append(valid_loss)
            valid_acc_history.append(valid_acc)

    return train_loss_history, train_acc_history, valid_loss_history, valid_acc_history

def test(model, test_loader,device):
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true += labels.cpu().numpy().tolist()
            y_pred += predicted.cpu().numpy().tolist()
    print('Accuracy of the network on the test images: %.3f %%' % (correct / total * 100))
    return 1.0*correct/total,confusion_matrix(y_true, y_pred)


if __name__ == '__main__':
    path_train = 'data/train'
    path_valid = 'data/val'
    
    parser = argparse.ArgumentParser(description='Facial Expression Recognition')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate ')
    parser.add_argument('--model', type=str, default="vggnet", metavar='MODEL', help='choice the model(default: vggnet)')
    
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    MODEL = args.model

    train_loader,valid_loader = get_data_generators(path_train,path_valid,BATCH_SIZE)

    if torch.cuda.is_available():
        # 使用GPU训练
        device = torch.device("cuda")
    else:
        # 使用CPU训练
        device = torch.device("cpu")
    #选择模型
    if MODEL == "vggnet":
        model = vggnet()
    elif MODEL == "resnet":
        model = resnet()

    # 将模型移动到相应的设备上
    model = model.to(device)
    print(model)
    # input_shape = (1,48,48)
    # summary(model,input_shape)

    # 训练和测试
    train_loss, train_acc, val_loss, val_acc = train(model, train_loader,device)
    accuracy, cm = test(model, valid_loader,device)

    # 创建文件夹，保存模型，曲线图和混淆矩阵
    folder_path = os.path.join("model",MODEL,'%.3f'%accuracy)
    print(folder_path)
    os.makedirs(folder_path)

    # 保存模型参数
    torch.save(model.state_dict(), os.path.join(folder_path,MODEL+'.h5'))

    # 绘制曲线图
    draw_metrics(train_loss,train_acc,val_loss,val_acc,os.path.join(folder_path,MODEL+'_metrics.jpg'))

    # 绘制混淆矩阵图
    draw_confusion_matrix(cm,os.path.join(folder_path,MODEL+'_confusion.jpg'))





