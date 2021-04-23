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
normalize = transforms.Normalize(mean=[0.665, 0.665, 0.665], std=[0.25, 0.25, 0.25])
transform = transforms.Compose([
            transforms.RandomResizedCrop(200),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-45,45)),
            #transforms.ColorJitter(brightness=0, contrast=0.3, hue=0),
            transforms.ToTensor(),
            normalize
            ])

#加载数据集
train_dataset = datasets.ImageFolder('MyDataset/train', transform=transform)
test_dataset = datasets.ImageFolder('MyDataset/test', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size = 5, shuffle = True,num_workers = 0)
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

net = EffNet().cuda()
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
#torch.save(net.state_dict(), 'MyProject.pt')