from PyQt5.QtWidgets import QDialog, QDesktopWidget
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QIcon
from src.mplwidget import MplWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget

from superqt import QLabeledDoubleRangeSlider

import sys
import cv2
import numpy as np

from src.transforms import AugmentationPipeline, augList
from src.utils.images import convert_cvimg_to_qimg

class TuningPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi('./src/qt_designer_file/tuningPanel.ui', self)
        self.ui.actionOpen.triggered.connect(self.openNewImage)
        self.imagePath = None
        self.matrixArgs = [[1,2,3,4,5],
		                    [1,2,3,4,5],
							[1,2,3,4,5],
							[1,2,3,4,5],
							[1,2,3,4,5]]
        self.genImages()
        self.placeSliders()
		
        #self.blurSlider.sliderReleased.connect(self.changeMatrixArgs)
		
    def openNewImage(self):
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", filter="Image files (*.jpg *.png)")
        self.imagePath = filePath[0]
        #print(self.imagePath)
        self.genImages()
		
    def genImages(self):
        if self.imagePath is None:
            self.imagePath = 'C:/Users/ajcmo/Desktop/WORK/Noisey-image/imgs/default_imgs/tank_iso.jpg'
        img = cv2.imread(self.imagePath)
        qt_img = convert_cvimg_to_qimg(img)
        #print(self.ui.width())
        pm = QPixmap(qt_img)
        pm = pm.scaled(self.ui.width() / 7, self.ui.height() / 6, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.ui.original.setPixmap(pm)
        #self.ui.original.resize(pm.width(),pm.height())
		
        matrix_id = [self.ui.aug_11, self.ui.aug_12, self.ui.aug_13, self.ui.aug_14, self.ui.aug_15,
					self.ui.aug_21, self.ui.aug_22, self.ui.aug_23, self.ui.aug_24, self.ui.aug_25,
					self.ui.aug_31, self.ui.aug_32, self.ui.aug_33, self.ui.aug_34, self.ui.aug_35,
					self.ui.aug_41, self.ui.aug_42, self.ui.aug_43, self.ui.aug_44, self.ui.aug_45,
					self.ui.aug_51, self.ui.aug_52, self.ui.aug_53, self.ui.aug_54, self.ui.aug_55]
		
		# Setting up aug pipeline for tuning (6 augs)
        tuningAug = AugmentationPipeline(augList)
        tuningAug.clear()
        tuningAug.append("Size")
        tuningAug.append("Gaussian Blur")
        tuningAug.append("Salt and Pepper")
        tuningAug.append("Contrast")
        tuningAug.append("Intensity")
        tuningAug.append("JPG Compression")
		
        #print(self.matrixArgs)
		
        cnt = 0
        #aug_img = img
        for i in range(5):
            for j in range(5):
                aug_img = img
                for aug in tuningAug:
                    if (aug.title == "Size"):
                        aug_img = aug(aug_img, aug.args[i])
                    #else:
                        #aug_img = aug(aug_img, self.matrixArgs[i][j]) # PROBLEM HERE???
                    elif (aug.title == "Gaussian Blur"):
                        aug_img = aug(aug_img, self.matrixArgs[0][j])
                    elif (aug.title == "Salt and Pepper"):
                        aug_img = aug(aug_img, self.matrixArgs[1][j])
                    elif (aug.title == "Contrast"):
                        aug_img = aug(aug_img, self.matrixArgs[2][j])
                    elif (aug.title == "Intensity"):
                        aug_img = aug(aug_img, self.matrixArgs[3][j])
                    elif (aug.title == "JPG Compression"):
                        aug_img = aug(aug_img, self.matrixArgs[4][j])
                aug_qt_img = convert_cvimg_to_qimg(aug_img)
                aug_pm = QPixmap(aug_qt_img)
                aug_pm = aug_pm.scaled(self.ui.width() / 7, self.ui.height() / 6, Qt.KeepAspectRatio, Qt.FastTransformation)
                matrix_id[cnt].setPixmap(aug_pm)
                cnt += 1
		
        #print(self.ui.horizontalSlider.TickPosition())
		
        
		
    def placeSliders(self):
		# Slider for Gaussian Blur 
        self.blurLabel = QLabel()
        self.blurLabel.setObjectName(u"blurLabel")
        self.blurLabel.setAlignment(Qt.AlignCenter)
        self.blurLabel.setText("Gaussian Blur")
        self.blurSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.blurSlider.setRange(1.0, 5.0)
        self.blurSlider.setValue((1.0, 2.0, 3.0, 4.0, 5.0))
        self.ui.sliderLayout.addWidget(self.blurLabel)
        self.ui.sliderLayout.addWidget(self.blurSlider)
        
		# Slider for Salt & Pepper 
        self.saltpepLabel = QLabel()
        self.saltpepLabel.setObjectName(u"blurLabel")
        self.saltpepLabel.setAlignment(Qt.AlignCenter)
        self.saltpepLabel.setText("Salt & Pepper")
        self.saltpepSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.saltpepSlider.setRange(1.0, 5.0)
        self.saltpepSlider.setValue((1.0, 2.0, 3.0, 4.0, 5.0))
        self.ui.sliderLayout.addWidget(self.saltpepLabel)
        self.ui.sliderLayout.addWidget(self.saltpepSlider)
		
		# Slider for Contrast
        self.contLabel = QLabel()
        self.contLabel.setObjectName(u"blurLabel")
        self.contLabel.setAlignment(Qt.AlignCenter)
        self.contLabel.setText("Contrast")
        self.contSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.contSlider.setRange(1.0, 5.0)
        self.contSlider.setValue((1.0, 2.0, 3.0, 4.0, 5.0))
        self.ui.sliderLayout.addWidget(self.contLabel)
        self.ui.sliderLayout.addWidget(self.contSlider)
		
		# Slider for Intensity
        self.intLabel = QLabel()
        self.intLabel.setObjectName(u"blurLabel")
        self.intLabel.setAlignment(Qt.AlignCenter)
        self.intLabel.setText("Intensity")
        self.intSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.intSlider.setRange(1.0, 5.0)
        self.intSlider.setValue((1.0, 2.0, 3.0, 4.0, 5.0))
        self.ui.sliderLayout.addWidget(self.intLabel)
        self.ui.sliderLayout.addWidget(self.intSlider)
		
		# Slider for JPG Compression
        self.compLabel = QLabel()
        self.compLabel.setObjectName(u"blurLabel")
        self.compLabel.setAlignment(Qt.AlignCenter)
        self.compLabel.setText("JPG Compression")
        self.compSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.compSlider.setRange(1.0, 5.0)
        self.compSlider.setValue((1.0, 2.0, 3.0, 4.0, 5.0))
        self.ui.sliderLayout.addWidget(self.compLabel)
        self.ui.sliderLayout.addWidget(self.compSlider)
		
		# Connect sliders to re-generate matrix
        self.blurSlider.valueChanged.connect(self.changeMatrixArgs)
        self.saltpepSlider.valueChanged.connect(self.changeMatrixArgs)
        self.contSlider.valueChanged.connect(self.changeMatrixArgs)
        self.intSlider.valueChanged.connect(self.changeMatrixArgs)
        self.compSlider.valueChanged.connect(self.changeMatrixArgs)
		
        self.ui.regenMatrix.clicked.connect(self.genImages)
		
    def changeMatrixArgs(self):
        self.matrixArgs[0] = self.blurSlider.value()
        self.matrixArgs[1] = self.saltpepSlider.value()
        self.matrixArgs[2] = self.contSlider.value()
        self.matrixArgs[3] = self.intSlider.value()
        self.matrixArgs[4] = self.compSlider.value()
        