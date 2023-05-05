import torch
import torchvision
import torchvision.transforms as transforms

def get_data_generators(path_train, path_vaild,batch_size):
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
    data_vaild = torchvision.datasets.ImageFolder(root=path_vaild,transform=transforms_vaild)

    train_loader = torch.utils.data.DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=data_vaild,batch_size=batch_size,shuffle=False)
    
    return train_loader,valid_loader
