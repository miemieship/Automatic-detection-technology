import sys
import cv2 as cv

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from demo01 import Ui_MainWindow

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
from PIL import Image

#运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))

#数据集正则化
normalize = transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(200),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)), #添加这行
            normalize
            ])

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
net = Net().cuda()
net.load_state_dict(torch.load('MyProject.pt'))
net = net.to(device)
torch.no_grad()
classes = ['C1_inclusion','C2_patches','C3_crazing','C4_pitted','C5_rolled-in','C6_scratches']

#GUI界面设置
class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
    def btnReadImage_Clicked(self):
        '''
        从本地读取图片
        '''
        # 打开文件选取对话框
        global filename
        filename,  _ = QFileDialog.getOpenFileName(self, '打开图片')
        self.captured = cv.imread(filename)
        # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
        self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)

        rows, cols, channels = self.captured.shape
        bytesPerLine = channels * cols
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelCapture.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelCapture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
    def btnIdentify_Clicked(self):
        '''
        识别
        '''
        img = Image.open(filename)
        img_ = transform(img).unsqueeze(0)
        img_ = img_.to(device)
        outputs = net(img_)
        _, predicted = torch.max(outputs,1)
        pre=predicted.cpu().numpy()
        self.resultDisplay.setText(''.join('%5s' % classes[x] for x in pre))
        '''
        #加载数据集
        filename='MyDataset/test\C1_inclusion\In_1.bmp'
        image=Image.open(filename).convert('RGB') #读取图像，转换为三维矩阵
        classes = ['C1_inclusion','C2_patches','C3_crazing','C4_pitted','C5_rolled-in','C6_scratches']
        with torch.no_grad():
              outputs = net(image)
              _, predicted = torch.max(outputs.data,1)
              pre=predicted.cpu().numpy()
              lab=labels.cpu().numpy()
              self.resultDisplay.setText(''.join('%5s' % classes[x] for x in pre))
            #  QImg = QImage(images, cols, rows, bytesPerLine, QImage.Format_RGB888)
              #self.labelCapture.setPixmap(QPixmap.fromImage(QImg).scaled(
            #  self.labelCapture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        '''
        

    
    
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())
    #python -m PyQt5.uic.pyuic demo01.ui -o demo01.py
    #pyinstaller -F --clean --distpath shark final01.py
