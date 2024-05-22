from PyQt5.QtWidgets import QDialog, QDesktopWidget

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import * #Qt, QObject, QThread, pyqtSignal, QSize
from PyQt5.QtGui import * #QPixmap, QIcon, QFont
from src.mplwidget import MplWidget
from PyQt5.QtWidgets import * #QApplication, QMainWindow, QPushButton, QLabel, QCheckBox, QVBoxLayout, QWidget

from PyQt5 import uic
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
		
		# Create standard augmentation level matrix
        self.matrixArgs = [[7.5,10.7,15.0,21.4,30.0],
		                    [0.375,0.53,0.75,1.07,1.5],
							[0.5,1.0,1.5,2.0,2.5],
							[5.0,7.14,10.0,14.0,20.0],
							[80.0,82.5,85.0,87.5,90.0]]
		
		#[[10,15,20,25,30],
		#[0.3,0.6,0.9,1.2,1.5],
		#[0.5,1,1.5,1.2,1.5],
		#[7,8.5,10,11.5,13],
		#[80,82.5,85,87.5,90]]
        self.blurSlider = None
        self.genImages()
        self.placeSliders()
		
        for i in range(5):
            x_label = QLabel()
            x_label.setAlignment(Qt.AlignCenter)
            x_label.setText("Y%d%d%d%d%d" % (i+1, i+1, i+1, i+1, i+1))
            x_label.setFont(QFont('Arial', 10))
            x_label.setStyleSheet("font-weight: bold")
            self.x_label_row.addWidget(x_label)
			
            y_label = QLabel()
            y_label.setAlignment(Qt.AlignCenter)
            y_label.setText("SIZE = %d" % (i+1))
            y_label.setFont(QFont('Arial', 10))
            y_label.setStyleSheet("font-weight: bold")
            self.y_label_col.addWidget(y_label)
			
		
        #self.blurSlider.sliderReleased.connect(self.changeMatrixArgs)
		
	# Opens a new image from the file explorer
    def openNewImage(self):
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", filter="Image files (*.jpg *.png)")
        self.imagePath = filePath[0]
        #print(self.imagePath)
        self.clearRows()
        self.genImages()
        
	# Clears all rows in the augmentation matrix
    def clearRows(self):
        for i in reversed(range(self.ui.row1.count())): 
            self.ui.row1.itemAt(i).widget().setParent(None)
            self.ui.row2.itemAt(i).widget().setParent(None)
            self.ui.row3.itemAt(i).widget().setParent(None)
            self.ui.row4.itemAt(i).widget().setParent(None)
            self.ui.row5.itemAt(i).widget().setParent(None)
		
	# Places the sliders on the UI
    def placeSliders(self):
		# Slider for Gaussian Blur 
        self.blurLabel = QLabel() #QCheckBox()
        self.blurLabel.setObjectName(u"blurLabel")
        self.blurLabel.setAlignment(Qt.AlignCenter)
        #self.blurLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.blurLabel.setText("Gaussian Blur")
        self.blurSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.blurSlider.setRange(10.0, 60.0)
        self.blurSlider.setValue((7.5,10.7,15.0,21.4,30.0))
        self.blurSlider.setDecimals(2)
        self.ui.sliderLayout.addWidget(self.blurLabel)
        self.ui.sliderLayout.addWidget(self.blurSlider)
        
		# Slider for Salt & Pepper 
        self.saltpepLabel = QLabel() #QCheckBox()
        self.saltpepLabel.setObjectName(u"saltPepLabel")
        self.saltpepLabel.setAlignment(Qt.AlignCenter)
        self.saltpepLabel.setText("Salt & Pepper")
        self.saltpepSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.saltpepSlider.setRange(0.0, 4.0)
        self.saltpepSlider.setValue((0.375,0.53,0.75,1.07,1.5))
        self.saltpepSlider.setDecimals(2)
        self.ui.sliderLayout.addWidget(self.saltpepLabel)
        self.ui.sliderLayout.addWidget(self.saltpepSlider)
		
		# Slider for Contrast
        self.contLabel = QLabel() #QCheckBox()
        self.contLabel.setObjectName(u"contLabel")
        self.contLabel.setAlignment(Qt.AlignCenter)
        self.contLabel.setText("Contrast")
        self.contSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.contSlider.setRange(0.0, 10.0)
        self.contSlider.setValue((0.5,1.0,1.5,2.0,2.5))
        self.contSlider.setDecimals(2)
        self.ui.sliderLayout.addWidget(self.contLabel)
        self.ui.sliderLayout.addWidget(self.contSlider)
		
		# Slider for Intensity
        self.intLabel = QLabel() #QCheckBox()
        self.intLabel.setObjectName(u"intLabel")
        self.intLabel.setAlignment(Qt.AlignCenter)
        self.intLabel.setText("Intensity")
        self.intSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.intSlider.setRange(4.0, 16.0)
        self.intSlider.setValue((5.0,7.14,10.0,14.0,20.0))
        self.intSlider.setDecimals(2)
        self.ui.sliderLayout.addWidget(self.intLabel)
        self.ui.sliderLayout.addWidget(self.intSlider)
		
		# Slider for JPG Compression
        self.compLabel = QLabel() #QCheckBox()
        self.compLabel.setObjectName(u"compLabel")
        self.compLabel.setAlignment(Qt.AlignCenter)
        self.compLabel.setText("JPG Compression")
        self.compSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.compSlider.setRange(80.0, 100.0)
        self.compSlider.setValue((80.0,82.5,85.0,87.5,90.0))
        self.compSlider.setDecimals(2)
        self.ui.sliderLayout.addWidget(self.compLabel)
        self.ui.sliderLayout.addWidget(self.compSlider)
		
		# Connect sliders to rchange augmentation parameters
        self.blurSlider.valueChanged.connect(self.changeMatrixArgs)
        self.saltpepSlider.valueChanged.connect(self.changeMatrixArgs)
        self.contSlider.valueChanged.connect(self.changeMatrixArgs)
        self.intSlider.valueChanged.connect(self.changeMatrixArgs)
        self.compSlider.valueChanged.connect(self.changeMatrixArgs)
		
		# Connect button to re-generate matrix
        self.ui.regenMatrix.clicked.connect(self.genImages)
        self.ui.resetSlidersButton.clicked.connect(self.resetSliders)
		
	# Re-generates the augmentation matrix
    def genImages(self):
        if self.imagePath is None:
            self.imagePath = './imgs/default_imgs/tank_iso.jpg'
        img = cv2.imread(self.imagePath)
        qt_img = convert_cvimg_to_qimg(img)
        
        pm = QPixmap(qt_img)
        pm = pm.scaled(self.ui.width() / 7, self.ui.height() / 6, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.ui.original.setPixmap(pm)
        self.ui.original.resize(pm.width(),pm.height())
		
        #matrix_id = [self.ui.aug_11, self.ui.aug_12, self.ui.aug_13, self.ui.aug_14, self.ui.aug_15,
		#			self.ui.aug_21, self.ui.aug_22, self.ui.aug_23, self.ui.aug_24, self.ui.aug_25,
		#			self.ui.aug_31, self.ui.aug_32, self.ui.aug_33, self.ui.aug_34, self.ui.aug_35,
		#			self.ui.aug_41, self.ui.aug_42, self.ui.aug_43, self.ui.aug_44, self.ui.aug_45,
		#			self.ui.aug_51, self.ui.aug_52, self.ui.aug_53, self.ui.aug_54, self.ui.aug_55]
		
		# Setting up aug pipeline for tuning (6 augs)
        tuningAug = AugmentationPipeline(augList)
        tuningAug.clear()
        tuningAug.append("Size")
        tuningAug.append("Gaussian Blur")
        tuningAug.append("Salt and Pepper")
        tuningAug.append("Contrast")
        tuningAug.append("Intensity")
        tuningAug.append("JPG Compression")
		
        self.clearRows()
		
        #cnt = 0
        # Iterate through each level of the Size augmentation
        for i in range(5):
			# Iterate through every other augmentation
            for j in range(5):
                aug_img = img
				# Read in augmentation name, perform aug()
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
				# Convert CV2 image to QT pixmap, and set
                aug_qt_img = convert_cvimg_to_qimg(aug_img)
                aug_pm = QPixmap(aug_qt_img)
                aug_pm = aug_pm.scaled(self.ui.width() / 7, self.ui.height() / 6, Qt.KeepAspectRatio, Qt.FastTransformation)
                #matrix_id[cnt].setPixmap(aug_pm)
                #cnt += 1
                augImg = QLabel()
                augImg.setPixmap(aug_pm)
                if (i == 0):
                    self.ui.row1.addWidget(augImg)
                elif (i == 1):
                    self.ui.row2.addWidget(augImg)
                elif (i == 2):
                    self.ui.row3.addWidget(augImg)
                elif (i == 3):
                    self.ui.row4.addWidget(augImg)
                elif (i == 4):
                    self.ui.row5.addWidget(augImg)
		
	# Changes each row of the augmentation matrix to the current valeus from 
	# the corresponding slider 
    def changeMatrixArgs(self):
        self.matrixArgs[0] = self.blurSlider.value()
        self.matrixArgs[1] = self.saltpepSlider.value()
        self.matrixArgs[2] = self.contSlider.value()
        self.matrixArgs[3] = self.intSlider.value()
        self.matrixArgs[4] = self.compSlider.value()
		
    def resetSliders(self):
        self.blurSlider.setValue((7.5,10.7,15.0,21.4,30.0))
        self.saltpepSlider.setValue((0.375,0.53,0.75,1.07,1.5))
        self.contSlider.setValue((0.5,1.0,1.5,2.0,2.5))
        self.intSlider.setValue((5.0,7.14,10.0,14.0,20.0))
        self.compSlider.setValue((80.0,82.5,85.0,87.5,90.0))
        
        self.genImages()
		
    def genImages(self):
        #self.filepath = 'C:/Users/ajcmo/Desktop/WORK/Noisey-image/imgs/default_imgs/tank_iso.jpg'
        #print(self.ui.width())
        pm = QPixmap('C:/Users/ajcmo/Desktop/WORK/Noisey-image/imgs/default_imgs/tank_iso.jpg')
        pm = pm.scaled(self.ui.width() / 6, self.ui.height() / 6, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.ui.original.setPixmap(pm)
        #self.ui.original.resize(pm.width(),pm.height())
		
		# First row
        self.ui.aug_11.setPixmap(pm)
        self.ui.aug_12.setPixmap(pm)
        self.ui.aug_13.setPixmap(pm)
        self.ui.aug_14.setPixmap(pm)
        self.ui.aug_15.setPixmap(pm)
		
		# Second row
        self.ui.aug_21.setPixmap(pm)
        self.ui.aug_22.setPixmap(pm)
        self.ui.aug_23.setPixmap(pm)
        self.ui.aug_24.setPixmap(pm)
        self.ui.aug_25.setPixmap(pm)
		
		# Third row
        self.ui.aug_31.setPixmap(pm)
        self.ui.aug_32.setPixmap(pm)
        self.ui.aug_33.setPixmap(pm)
        self.ui.aug_34.setPixmap(pm)
        self.ui.aug_35.setPixmap(pm)
		
		# Fourth row
        self.ui.aug_41.setPixmap(pm)
        self.ui.aug_42.setPixmap(pm)
        self.ui.aug_43.setPixmap(pm)
        self.ui.aug_44.setPixmap(pm)
        self.ui.aug_45.setPixmap(pm)
		
		# Fifth row
        self.ui.aug_51.setPixmap(pm)
        self.ui.aug_52.setPixmap(pm)
        self.ui.aug_53.setPixmap(pm)
        self.ui.aug_54.setPixmap(pm)
        self.ui.aug_55.setPixmap(pm)
		
        #self.ui.aug_1_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        #self.ui.aug_1_slider.setRange(0.0, 10.0)
        #self.ui.aug_1_slider.setValue((2.0, 8.0))
        #self.ui.aug_1_slider.show()
