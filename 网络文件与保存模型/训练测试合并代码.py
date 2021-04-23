import os
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

#运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))

#数据集正则化
normalize = transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform = transforms.Compose([
            transforms.RandomResizedCrop(200),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])

#加载数据集
train_dataset = datasets.ImageFolder('./MyDataset/train', transform=transform)
test_dataset = datasets.ImageFolder('./MyDataset/test', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size = 4, shuffle = True,num_workers = 0)
test_loader = DataLoader(dataset=test_dataset, batch_size = 4, shuffle = True,num_workers = 0)

#卷积网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*47*47, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 6)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16 * 47 * 47)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

#去除类名
classes = train_dataset.classes

#实例化
net = Net()

#用显卡
net = net.to(device)

#优化器和误差函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
cirterion = nn.CrossEntropyLoss()

epochs = 100

for epoch in range(epochs):
    running_loss = 0.0
    
    for i ,data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 100 == 99:
           print('[%d %5d] loss: %.3f' % (epoch + 1,i + 1,running_loss / 2000))
           running_loss = 0.0
            
print('finished training!') 

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader: 
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        outputs = net(images)
        _, predicted = torch.max(outputs.data,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print('Accuracy of the network on test images: %d %%' % (100 * correct / total))