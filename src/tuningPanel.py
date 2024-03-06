from PyQt5.QtWidgets import QDialog, QDesktopWidget
from PyQt5 import uic
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QIcon
from src.mplwidget import MplWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget

from superqt import QLabeledDoubleRangeSlider

import sys
import cv2

from src.transforms import AugmentationPipeline, augList
from src.utils.images import convert_cvimg_to_qimg

class TuningPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi('./src/qt_designer_file/tuningPanel.ui', self)
        self.genImages()
		
    def genImages(self):
        filepath = 'C:/Users/ajcmo/Desktop/WORK/Noisey-image/imgs/default_imgs/tank_iso.jpg'
        img = cv2.imread(filepath)
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
		
        cnt = 0
        #aug_img = img
        for i in range(5):
            for j in range(5):
                aug_img = img
                for aug in tuningAug:
                    if (aug.title == "Size"):
                        aug_img = aug(aug_img, aug.args[i])
                    else:
                        aug_img = aug(aug_img, aug.args[j])
                aug_qt_img = convert_cvimg_to_qimg(aug_img)
                aug_pm = QPixmap(aug_qt_img)
                aug_pm = aug_pm.scaled(self.ui.width() / 7, self.ui.height() / 6, Qt.KeepAspectRatio, Qt.FastTransformation)
                matrix_id[cnt].setPixmap(aug_pm)
                cnt += 1
		
        #print(self.ui.horizontalSlider.TickPosition())
		
        self.ui.horizontalSlider = QLabeledDoubleRangeSlider()
        self.ui.horizontalSlider.setRange(0.0, 10.0)
        self.ui.horizontalSlider.setValue((2.0, 8.0))
        self.ui.horizontalSlider.show()
		
        #rangedSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        #rangedSlider.setRange(0.0, 10.0)
        #rangedSlider.setValue((2.0, 8.0))
        #rangedSlider.show()