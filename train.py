import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import get_model
from data_augmentation import get_data_generators

os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 0.001

def train(model, train_loader,device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # 移动到gpu上
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            #print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

def test(model, test_loader,device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return correct/total*1.0


if __name__ == '__main__':
    path_train = 'data/train'
    path_valid = 'data/val'
    
    train_loader,valid_loader = get_data_generators(path_train,path_valid,BATCH_SIZE)

    if torch.cuda.is_available():
        # 使用GPU训练
        device = torch.device("cuda")
        print("gpu")
    else:
        # 使用CPU训练
        device = torch.device("cpu")
        print("cpu")

    # 将模型移动到相应的设备上
    model = get_model()
    model = model.to(device)
    print(model)

    # 训练和测试
    train(model, train_loader,device)
    accuracy = test(model, valid_loader,device)

    # 保存模型参数
    filename = 'model_{}.pth'.format(accuracy)
    torch.save(model.state_dict(), filename)
