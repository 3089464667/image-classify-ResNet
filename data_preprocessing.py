import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_data(batch_size=32, val_split=0.1):
    # 定义训练集的数据转换（包含图像增强）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),     # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 定义测试集的数据转换（不包含图像增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    train_size = int((1 - val_split) * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    valset.dataset.transform = test_transform
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    # 加载测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

def get_class_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_preprocess_data(batch_size=32, val_split=0.1):
    trainloader, valloader, testloader = load_data(batch_size, val_split)
    
    # 获取一个批次的数据来确定输入形状
    example_data, _ = next(iter(trainloader))
    input_shape = example_data.shape[1:]  # (C, H, W)
    
    return trainloader, valloader, testloader, input_shape