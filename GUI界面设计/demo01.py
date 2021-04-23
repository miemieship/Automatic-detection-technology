# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo01.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(562, 696)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnReadImage = QtWidgets.QPushButton(self.centralwidget)
        self.btnReadImage.setGeometry(QtCore.QRect(170, 470, 221, 41))
        self.btnReadImage.setObjectName("btnReadImage")
        self.labelCapture = QtWidgets.QLabel(self.centralwidget)
        self.labelCapture.setGeometry(QtCore.QRect(90, 40, 400, 400))
        self.labelCapture.setObjectName("labelCapture")
        self.btnIdentify = QtWidgets.QPushButton(self.centralwidget)
        self.btnIdentify.setGeometry(QtCore.QRect(170, 530, 221, 41))
        self.btnIdentify.setObjectName("btnIdentify")
        self.resultDisplay = QtWidgets.QLabel(self.centralwidget)
        self.resultDisplay.setGeometry(QtCore.QRect(170, 590, 221, 31))
        self.resultDisplay.setText("")
        self.resultDisplay.setWordWrap(False)
        self.resultDisplay.setObjectName("resultDisplay")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 562, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.btnReadImage.clicked.connect(MainWindow.btnReadImage_Clicked)
        self.btnIdentify.clicked.connect(MainWindow.btnIdentify_Clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnReadImage.setText(_translate("MainWindow", "打开"))
        self.labelCapture.setText(_translate("MainWindow", "捕获图"))
        self.btnIdentify.setText(_translate("MainWindow", "识别"))
