# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_windowyAxcEv.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from src.qlabel import Label


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1299, 681)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet(u"")
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName(u"actionOpen")
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName(u"actionSave")
        self.actionSave_As = QAction(MainWindow)
        self.actionSave_As.setObjectName(u"actionSave_As")
        self.actionIncrease_Size = QAction(MainWindow)
        self.actionIncrease_Size.setObjectName(u"actionIncrease_Size")
        self.actionDecrease_Size = QAction(MainWindow)
        self.actionDecrease_Size.setObjectName(u"actionDecrease_Size")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"")
        self.horizontalLayout_6 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.original = Label(self.centralwidget)
        self.original.setObjectName(u"original")
        sizePolicy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.original.sizePolicy().hasHeightForWidth())
        self.original.setSizePolicy(sizePolicy)
        self.original.setScaledContents(True)

        self.verticalLayout_2.addWidget(self.original)

        self.label_8 = QLabel(self.centralwidget)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_8)

        self.verticalLayout_2.setStretch(0, 5)

        self.verticalLayout_5.addLayout(self.verticalLayout_2)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.original_2 = Label(self.centralwidget)
        self.original_2.setObjectName(u"original_2")
        sizePolicy.setHeightForWidth(self.original_2.sizePolicy().hasHeightForWidth())
        self.original_2.setSizePolicy(sizePolicy)
        self.original_2.setScaledContents(True)

        self.verticalLayout_4.addWidget(self.original_2)

        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.label_10)

        self.verticalLayout_4.setStretch(0, 5)

        self.verticalLayout_5.addLayout(self.verticalLayout_4)


        self.gridLayout.addLayout(self.verticalLayout_5, 0, 1, 1, 1)

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_2, 2, 1, 1, 1)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label, 2, 2, 1, 1)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.pushButton_5 = QPushButton(self.centralwidget)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setStyleSheet(u"background-color: rgb(158, 90, 214);")

        self.horizontalLayout_5.addWidget(self.pushButton_5)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_3)

        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setStyleSheet(u"background-color: rgb(114, 159, 207);")

        self.horizontalLayout_5.addWidget(self.pushButton_2)

        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setStyleSheet(u"background-color: rgb(0, 223, 255);")

        self.horizontalLayout_5.addWidget(self.pushButton)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_5)

        self.pushButton_4 = QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName(u"pushButton_4")
        palette = QPalette()
        brush = QBrush(QColor(0, 170, 0, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush)
        palette.setBrush(QPalette.Active, QPalette.BrightText, brush)
        palette.setBrush(QPalette.Active, QPalette.Base, brush)
        palette.setBrush(QPalette.Active, QPalette.Window, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush)
        brush1 = QBrush(QColor(255, 51, 51, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.BrightText, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush)
        palette.setBrush(QPalette.Disabled, QPalette.BrightText, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush)
        self.pushButton_4.setPalette(palette)
        self.pushButton_4.setLayoutDirection(Qt.LeftToRight)
        self.pushButton_4.setStyleSheet(u"background-color: rgb(0, 170, 0)")

        self.horizontalLayout_5.addWidget(self.pushButton_4)

        self.pushButton_3 = QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setStyleSheet(u"background-color: rgb(239, 41, 41);")

        self.horizontalLayout_5.addWidget(self.pushButton_3)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)

        self.horizontalLayout_5.setStretch(0, 5)
        self.horizontalLayout_5.setStretch(1, 4)
        self.horizontalLayout_5.setStretch(2, 4)
        self.horizontalLayout_5.setStretch(3, 2)
        self.horizontalLayout_5.setStretch(5, 2)
        self.horizontalLayout_5.setStretch(6, 1)

        self.gridLayout_2.addLayout(self.horizontalLayout_5, 3, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.label_3)

        self.comboBox = QComboBox(self.centralwidget)
        self.comboBox.setObjectName(u"comboBox")

        self.horizontalLayout.addWidget(self.comboBox)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)

        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.normalAug = QRadioButton(self.centralwidget) #QCheckBox
        self.normalAug.setObjectName(u"normalAug")
        self.normalAug.setEnabled(True)
        self.compoundAug = QRadioButton(self.centralwidget) #QCheckBox
        self.compoundAug.setObjectName(u"compoundAug")
        self.compoundAug.setEnabled(True)
        self.sequentialAug = QRadioButton(self.centralwidget)
        self.sequentialAug.setObjectName(u"sequentialAug")
        self.sequentialAug.setEnabled(True)

        self.horizontalLayout_2.addWidget(self.normalAug)
        self.horizontalLayout_2.addWidget(self.compoundAug)
        self.horizontalLayout_2.addWidget(self.sequentialAug)

        self.progressBar = QProgressBar(self.centralwidget)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setEnabled(True)
        self.progressBar.setMaximum(5)
        self.progressBar.setValue(0)

        self.horizontalLayout_2.addWidget(self.progressBar)


        self.gridLayout_2.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)


        self.gridLayout.addLayout(self.gridLayout_2, 3, 2, 1, 1)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 1, 1, 1, 1)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_2, 1, 2, 1, 1)

        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.preview = Label(self.centralwidget)
        self.preview.setObjectName(u"preview")
        sizePolicy.setHeightForWidth(self.preview.sizePolicy().hasHeightForWidth())
        self.preview.setSizePolicy(sizePolicy)
        self.preview.setScaledContents(True)

        self.verticalLayout_3.addWidget(self.preview)

        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_9)

        self.verticalLayout_3.setStretch(0, 5)

        self.verticalLayout_7.addLayout(self.verticalLayout_3)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.preview_2 = Label(self.centralwidget)
        self.preview_2.setObjectName(u"preview_2")
        sizePolicy.setHeightForWidth(self.preview_2.sizePolicy().hasHeightForWidth())
        self.preview_2.setSizePolicy(sizePolicy)
        self.preview_2.setScaledContents(True)

        self.verticalLayout_6.addWidget(self.preview_2)

        self.label_11 = QLabel(self.centralwidget)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setAlignment(Qt.AlignCenter)

        self.verticalLayout_6.addWidget(self.label_11)

        self.verticalLayout_6.setStretch(0, 5)

        self.verticalLayout_7.addLayout(self.verticalLayout_6)


        self.gridLayout.addLayout(self.verticalLayout_7, 0, 2, 1, 1)

        self.progressBar_2 = QProgressBar(self.centralwidget)
        self.progressBar_2.setObjectName(u"progressBar_2")
        self.progressBar_2.setValue(0)

        self.gridLayout.addWidget(self.progressBar_2, 2, 0, 1, 1)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout_13 = QVBoxLayout()
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.listAugs = QListWidget(self.centralwidget)
        self.listAugs.setObjectName(u"listAugs")

        self.verticalLayout_13.addWidget(self.listAugs)


        self.horizontalLayout_4.addLayout(self.verticalLayout_13)

        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.upListAug = QPushButton(self.centralwidget)
        self.upListAug.setObjectName(u"upListAug")

        self.verticalLayout_11.addWidget(self.upListAug)

        self.downListAug = QPushButton(self.centralwidget)
        self.downListAug.setObjectName(u"downListAug")

        self.verticalLayout_11.addWidget(self.downListAug)

        self.deleteListAug = QPushButton(self.centralwidget)
        self.deleteListAug.setObjectName(u"deleteListAug")

        self.verticalLayout_11.addWidget(self.deleteListAug)


        self.horizontalLayout_4.addLayout(self.verticalLayout_11)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.addAug = QPushButton(self.centralwidget)
        self.addAug.setObjectName(u"addAug")

        self.verticalLayout_10.addWidget(self.addAug)

        self.loadAug = QPushButton(self.centralwidget)
        self.loadAug.setObjectName(u"loadAug")

        self.verticalLayout_10.addWidget(self.loadAug)

        self.saveAug = QPushButton(self.centralwidget)
        self.saveAug.setObjectName(u"saveAug")

        self.verticalLayout_10.addWidget(self.saveAug)


        self.horizontalLayout_3.addLayout(self.verticalLayout_10)

        self.line_3 = QFrame(self.centralwidget)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.VLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_3.addWidget(self.line_3)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)


        self.verticalLayout.addLayout(self.horizontalLayout_4)


        self.gridLayout.addLayout(self.verticalLayout, 3, 1, 1, 1)

        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        font = QFont()
        font.setPointSize(10)
        self.tabWidget.setFont(font)
        self.tabWidget.setStyleSheet(u"QTabBar {font: 8pt \"MS Shell Dlg 2\"};")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayout_8 = QVBoxLayout(self.tab)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.uncompressed = QListWidget(self.tab)
        self.uncompressed.setObjectName(u"uncompressed")

        self.verticalLayout_8.addWidget(self.uncompressed)

        self.augmented = QListWidget(self.tab)
        self.augmented.setObjectName(u"augmented")

        self.verticalLayout_8.addWidget(self.augmented)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.horizontalLayout_8 = QHBoxLayout(self.tab_2)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.fileList = QListWidget(self.tab_2)
        self.fileList.setObjectName(u"fileList")

        self.horizontalLayout_8.addWidget(self.fileList)

        self.tabWidget.addTab(self.tab_2, "")

        self.gridLayout.addWidget(self.tabWidget, 0, 0, 2, 1)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 6)
        self.gridLayout.setColumnStretch(2, 6)

        self.horizontalLayout_6.addLayout(self.gridLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1299, 26))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setObjectName(u"menuEdit")
        self.menuFont_Size = QMenu(self.menuEdit)
        self.menuFont_Size.setObjectName(u"menuFont_Size")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuEdit.addAction(self.menuFont_Size.menuAction())
        self.menuFont_Size.addAction(self.actionIncrease_Size)
        self.menuFont_Size.addAction(self.actionDecrease_Size)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionOpen.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.actionSave.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.actionSave_As.setText(QCoreApplication.translate("MainWindow", u"Save As", None))
        self.actionIncrease_Size.setText(QCoreApplication.translate("MainWindow", u"Increase Size", None))
        self.actionDecrease_Size.setText(QCoreApplication.translate("MainWindow", u"Decrease Size", None))
        self.original.setText("")
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Original Image", None))
        self.original_2.setText("")
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Uncompressed Detections", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Augmentation Generator", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Models", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"Tune Parameters", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Generate Images", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Run Preview", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"Default", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Select a model:", None))
        self.normalAug.setText(QCoreApplication.translate("MainWindow", u"Normal Augmentations", None))
        self.compoundAug.setText(QCoreApplication.translate("MainWindow", u"Compound Augmentations", None))
        self.sequentialAug.setText(QCoreApplication.translate("MainWindow", u"Sequential Augmentations", None))
        self.preview.setText("")
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Augmented Image", None))
        self.preview_2.setText("")
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Augmented Detections", None))
        self.upListAug.setText(QCoreApplication.translate("MainWindow", u"^", None))
        self.downListAug.setText(QCoreApplication.translate("MainWindow", u"v", None))
        self.deleteListAug.setText(QCoreApplication.translate("MainWindow", u"x", None))
        self.addAug.setText(QCoreApplication.translate("MainWindow", u"Add Augmentations", None))
        self.loadAug.setText(QCoreApplication.translate("MainWindow", u"Load Augmentations", None))
        self.saveAug.setText(QCoreApplication.translate("MainWindow", u"Save Augmentations", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"Items", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Images", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuEdit.setTitle(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.menuFont_Size.setTitle(QCoreApplication.translate("MainWindow", u"Font Size", None))
    # retranslateUi

