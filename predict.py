import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import create_model

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    return testloader

def predict_on_gpu():
    testloader = load_test_data()

    model = create_model().to(device)
    model.load_state_dict(torch.load('cifar10_classification_model.pth'))
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"测试集准确率: {accuracy:.4f}")

    # 定义类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 显示一些预测结果示例
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = testloader.dataset[i][0].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # 反归一化
        plt.imshow(img)
        predicted_label = class_names[all_preds[i]]
        true_label = class_names[all_labels[i]]
        color = 'blue' if predicted_label == true_label else 'red'
        plt.xlabel(f"{predicted_label} ({true_label})", color=color)
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()

if __name__ == '__main__':
    predict_on_gpu()