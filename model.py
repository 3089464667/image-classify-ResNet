import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes=10):
    # 使用预训练的ResNet18模型
    model = models.resnet18(pretrained=True)
    
    # 修改最后一层以适应CIFAR-10的类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model