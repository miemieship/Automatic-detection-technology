import os
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
'''
Accuracy of the network on test images: 90 %
mean=[0.665, 0.665, 0.665], std=[0.25, 0.25, 0.25]
40 五数据增强与dropout

'''
#运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))

#数据集正则化
normalize = transforms.Normalize(mean=[0.665, 0.665, 0.665], std=[0.25, 0.25, 0.25])
transform = transforms.Compose([
            transforms.RandomResizedCrop(200),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-45,45)),
            #transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.ToTensor(),
            normalize
            ])


#加载数据集
train_dataset = datasets.ImageFolder('MyDataset/train', transform=transform)
test_dataset = datasets.ImageFolder('MyDataset/test', transform=transform)

#划分训练集和验证集
valid_size = 0.2
num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx =indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#数据集
train_loader = DataLoader(dataset=train_dataset, batch_size = 5, sampler = train_sampler,num_workers = 0)
valid_loader = DataLoader(dataset=train_dataset, batch_size = 5, sampler = valid_sampler,num_workers = 0)
test_loader = DataLoader(dataset=test_dataset, batch_size = 5, shuffle = True,num_workers = 0)

#卷积网络
class EffNet(nn.Module):

    def __init__(self, nb_classes=6, include_top=True, weights=None):
        super(EffNet, self).__init__()
        
        self.block1 = self.make_layers(32, 64)
        self.block2 = self.make_layers(64, 128)
        self.block3 = self.make_layers(128, 256)
        self.linear = nn.Linear(in_features = 256*25*25, out_features = 6,bias=False)#batch_size
        self.include_top = include_top
        self.weights = weights

    def make_layers(self, ch_in, ch_out):
        layers = [
            nn.Conv2d(3, ch_in, kernel_size=(1,1), stride=(1,1), bias=False, padding=0, dilation=(1,1)) if ch_in ==32 else nn.Conv2d(ch_in, ch_in, kernel_size=(1,1),stride=(1,1), bias=False, padding=0, dilation=(1,1)) ,
            self.make_post(ch_in),
            
            # 2维深度卷积，用的一个1x3的空间可分离卷积
            nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(1, 3),stride=(1,1), padding=(0,1), bias=False, dilation=(1,1)),
            self.make_post(ch_in),
            
            #最大池化
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            
            # 2维深度卷积，用的一个3x1的空间可分离卷积
            nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(3, 1), stride=(1,1), padding=(1,0), bias=False, dilation=(1,1)),
            self.make_post(ch_in),
            
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 2), stride=(1, 2), bias=False, padding=(0,0), dilation=(1,1)),
            self.make_post(ch_out),
        ]
        return nn.Sequential(*layers)

    def make_post(self, ch_in):
        layers = [
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(ch_in, momentum=0.99)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        #x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.block2(x)
        x = self.block3(x)
        if self.include_top:

            x = x.view(-1,256*25*25)
            #print(x.shape)
            x = self.linear(x)
        return x
        
classes = train_dataset.classes

net = EffNet()

net = net.to(device)

#优化器和误差函数
cirterion = nn.CrossEntropyLoss()

#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochs = 35 #here！

optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, 10, 0.6)  


valid_loss_min = np.Inf

epoch_list=[]
trainloss_list=[]
validloss_list=[]
x_ticks = np.linspace(1,epochs,10)
y_ticks = np.linspace(0,1,10)

for epoch in range(1,epochs+1):
    train_loss = 0.0
    valid_loss = 0.0
    
    scheduler.step()
    
    #训练
    net.train()
    for i ,data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    #验证
    net.eval()
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        output = net(inputs)
        loss = cirterion(output, labels)
        valid_loss += loss.item()
        
    #计算平均损失
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    
    #收集训练效果以供画图
    epoch_list.append(epoch)
    trainloss_list.append(train_loss)
    validloss_list.append(valid_loss)

    #显示损失函数
    print('Epoch: {} \tTrain Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    
    #保存最优化模型
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,valid_loss))
        torch.save(net.state_dict(), 'MyProject.pt')
        valid_loss_min = valid_loss
        
plt.figure()
plt.plot(epoch_list,trainloss_list,label='train')
plt.plot(epoch_list,validloss_list,color='red',linewidth=1.0,linestyle='--',label='valid')
plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0, epochs)
plt.ylim(-0.1, 1.1)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.show()