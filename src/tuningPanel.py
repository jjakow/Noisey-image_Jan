from PyQt5.QtWidgets import QDialog, QDesktopWidget
from PyQt5 import uic
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QIcon
from src.mplwidget import MplWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget

from superqt import QLabeledDoubleRangeSlider

import sys

class TuningPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi('./src/qt_designer_file/tuningPanel.ui', self)
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
		
        self.ui.horizontalSlider = QLabeledDoubleRangeSlider()
        self.ui.horizontalSlider.setRange(0.0, 10.0)
        self.ui.horizontalSlider.setValue((2.0, 8.0))