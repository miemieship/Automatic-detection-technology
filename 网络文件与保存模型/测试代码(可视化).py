import os
import torch
import torchvision
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

#图像数据可视化
def imshow(inp, title=None,text_truth=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.4, 0.4, 0.4])
    std = np.array([0.2, 0.2, 0.2])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    if text_truth is not None:
        plt.text(-100, 230, text_truth,fontdict={'size':'11.5'})
    plt.pause(0.001)
    
    
    
#数据集正则化
normalize = transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform = transforms.Compose([
            transforms.RandomResizedCrop(200),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])

#加载数据集
train_dataset = datasets.ImageFolder('MyDataset/train', transform=transform)
test_dataset = datasets.ImageFolder('MyDataset/test', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size = 4, shuffle = True,num_workers = 0)
test_loader = DataLoader(dataset=test_dataset, batch_size = 4, shuffle = True,num_workers = 0)

#卷积网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*47*47, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 6)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16 * 47 * 47)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x

classes = train_dataset.classes

net = Net().cuda()
net.load_state_dict(torch.load('MyProject.pt'))

correct = 0
total = 0
i=1
j=0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        out = torchvision.utils.make_grid(images)
        outputs = net(images)
        _, predicted = torch.max(outputs.data,1)
        pre=predicted.cpu().numpy()
        lab=labels.cpu().numpy()
        plt.axis('off')
        if i % 50== 0:
            imshow(out.cpu(), title=('Pred',''.join('%5s' % [classes[x] for x in pre])),text_truth=('Truth',''.join('%5s' % [classes[x] for x in lab] )))
            #plt.text(0,-0.2,('Truth',''.join('%5s' % [classes[x] for x in lab] )))
        i=i+1
        j=j+1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
torch.save(net.state_dict(), 'MyProject.pt')