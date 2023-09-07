# For Pyinstaller spash screen
try:
    import pyi_splash
    pyi_splash.update_text("Loading...")
except ImportError:
    pass

# System libs
import os
from pathlib import Path
import sys
from queue import Queue

# PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox
from src.window import Ui_MainWindow
from PyQt5.QtCore import Qt

import cv2

# import utilities:
from src.utils.images import convert_cvimg_to_qimg
from src.transforms import AugDialog, mainAug
from src.experimentDialog import ExperimentConfig, ExperimentDialog
from src import models
from src.utils.weights import Downloader
from src.dataParser import ReadYAMLProgressWindow, yamlWorker

CURRENT_PATH = str(Path(__file__).parent.absolute()) + '/'
TEMP_PATH = CURRENT_PATH + 'src/tmp_results/'
DEFAULT_PATH = CURRENT_PATH + 'imgs/default_imgs/'

class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal(tuple)
    progress = QtCore.pyqtSignal(int)

    def setup(self, files, model_type, listWidget):
        self.files = files
        self.listWidget = listWidget
        #assert model_type == 'segmentation' or model_type == 'yolov3', "Model Type %s is not a defined term!"%(model_type)
        self.model_type = model_type

    def run(self):
        model = models._registry[self.model_type]
        self.progress.emit(1)

        model.initialize()
        self.progress.emit(2)

        result = []
        for img in self.files:
            pred = model.run(img)
            temp = model.draw(pred, img)
            temp["pred"] = pred
            result.append(temp)
            self.progress.emit(3)

        result[0]['model'] = self.model_type
        self.progress.emit(4)
        model.deinitialize()

        self.finished.emit((result, self.listWidget))


class mainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.addWindow = AugDialog(self.ui.listAugs, self)
        self.addWindow.setModal(True)
        self.addWindow.demoAug()

        # Check status of configurations:
        weight_dict = {'mit_semseg':"ade20k-hrnetv2-c1", 'yolov3':"yolov3.weights", 'detr':"detr-r50-e632da11.pth", 'yolov4':"yolov4.weights", "yolov3-face":"yolov3-face_last.weights", "yolox":"yolox_m.pth"}
        self.labels = []

        if Downloader.check(weight_dict):
            self.downloadDialog = Downloader(weight_dict, self)
            self.downloadDialog.setModal(True)
            self.downloadDialog.show()

        self.ui.listAugs.setMaximumSize(400,100) # quickfix for sizing issue with layouts
        self.ui.deleteListAug.setMaximumWidth(30)
        self.ui.upListAug.setMaximumWidth(30)
        self.ui.downListAug.setMaximumWidth(30)

        self.ui.progressBar.hide()
        self.ui.progressBar_2.hide()

        self.ui.comboBox.addItems(list(models._registry.keys()))

        # Buttons
        self.ui.pushButton.clicked.connect(self.run_model)  
        self.ui.pushButton_2.clicked.connect(self.startExperiment)
        self.ui.pushButton_3.clicked.connect(self.close)
        self.ui.pushButton_4.clicked.connect(self.setToDefault)

        # Augmentation Generator:
        #self.ui.compoundAug.setChecked(True)
        self.ui.addAug.clicked.connect(self.addWindow.show)
        #self.ui.demoAug.clicked.connect(self.addWindow.demoAug)
        self.ui.loadAug.clicked.connect(self.addWindow.__loadFileDialog__)
        self.ui.saveAug.clicked.connect(self.addWindow.__saveFileDialog__)
        self.ui.deleteListAug.clicked.connect(self.addWindow.__deleteItem__)
        self.ui.downListAug.clicked.connect(self.addWindow.__moveDown__)
        self.ui.upListAug.clicked.connect(self.addWindow.__moveUp__)
        # self.ui.listAugs.itemChanged.connect(self.changePreviewImage)
        # access model of listwidget to detect changes
        self.addWindow.pipelineChanged.connect(self.changePreviewImage)
        #self.ui.runOnAug.stateChanged.connect(self.runAugOnImage)

        # Menubar buttons
        #self.ui.actionOpen.triggered.connect(lambda: self.open_file())
        self.ui.actionOpen.triggered.connect(self.parseData)
        self.ui.actionIncrease_Size.triggered.connect(self.increaseFont)
        self.ui.actionDecrease_Size.triggered.connect(self.decreaseFont)

        # Qlistwidget signals
        self.ui.augmented.currentItemChanged.connect(self.change_augmented_selection)
        self.ui.uncompressed.currentItemChanged.connect(self.change_uncompressed_selection)
        self.ui.fileList.currentItemChanged.connect(self.change_file_selection)

        # QActions
        # Default values (images, noise, etc.) are set up here:
        self.default_img()

        # Font
        font = self.font()
        font.setPointSize(10)
        self.ui.centralwidget.setFont(font)

        # Drag and drop
        self.ui.original.imageDropped.connect(self.open_file)

        self.ui.fileList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.fileList.customContextMenuRequested.connect(self.listwidgetmenu)
        self.ui.fileList.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.ui.listAugs.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.label_eval = None

        # yaml stuff:
        self.yamlThread = QThread()
        self.yamlProgress = ReadYAMLProgressWindow()
        self.yamlQueue = Queue()
        self.yamlWorker = yamlWorker(self.yamlQueue)

    def listwidgetmenu(self, position):
        """menu for right clicking in the file list widget"""
        right_menu = QtWidgets.QMenu(self.ui.fileList)
        remove_action = QtWidgets.QAction("Close", self, triggered = self.closeFile)

        right_menu.addAction(self.ui.actionOpen)

        if self.ui.fileList.itemAt(position):
            right_menu.addAction(remove_action)

        right_menu.exec_(self.ui.fileList.mapToGlobal(position))

    def closeFile(self):
        """Removes a file from the file list widget"""
        items = self.ui.fileList.selectedItems()

        for item in items:
            row = self.ui.fileList.row(item)
            self.ui.fileList.takeItem(row)
        
        if self.ui.fileList.count() == 0:
            self.emptyNest = True
            self.flipAllControls(False)
            self.ui.original.clear()
            self.ui.preview_2.clear()
            self.ui.original_2.clear()
            self.ui.preview.clear()

    def increaseFont(self):
        """Increses the size of font across the whole application"""
        self.ui.centralwidget.setFont(QtGui.QFont('Ubuntu', self.ui.centralwidget.fontInfo().pointSize() + 1))
        #print(self.ui.centralwidget.fontInfo().pointSize())

    def decreaseFont(self):
        """Decreses the size of font across the whole application"""
        self.ui.centralwidget.setFont(QtGui.QFont('Ubuntu', self.ui.centralwidget.fontInfo().pointSize() - 1))
        
    def apply_augmentations(self, img):
        for aug in mainAug:
            img = aug(img, example=True)
        return img

    def changePreviewImage(self, *kwargs):
        #print(kwargs)
        print("recreating noisey image")
        current_item = self.ui.fileList.currentItem()
        image = cv2.imread(current_item.data(QtCore.Qt.UserRole)['filePath'])
        #if image is not None:
        image = self.apply_augmentations(image)
        qt_img = convert_cvimg_to_qimg(image)
        self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(qt_img))


    def default_img(self):
        #print(CURRENT_PATH + "imgs/" + fileName)
        if(os.path.isdir(DEFAULT_PATH)):
            onlyfiles = [f for f in os.listdir(DEFAULT_PATH) if os.path.isfile(os.path.join(DEFAULT_PATH, f))]
    
        onlyfiles = [DEFAULT_PATH + s for s in onlyfiles]
        
        self.open_file(onlyfiles)
        

            # if(Path(file).stem == "original"):
            #     original = os.path.join(DEFAULT_PATH, file)
            #     self.open_file(original)

            # if(Path(file).stem == "segmentation"):
            #     print(Path(file).stem)
            #     segmentation = os.path.join(DEFAULT_PATH, file)
            #     qt_img = convert_cvimg_to_qimg(cv2.imread(segmentation))
            #     self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(qt_img))

            # if(Path(file).stem == "segmentation_overlay"):
            #     segmentation_overlay = os.path.join(DEFAULT_PATH, file)
            #     qt_img = convert_cvimg_to_qimg(cv2.imread(segmentation_overlay))
            #     self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(qt_img))

        self.changePreviewImage()
        self.ui.fileList.setCurrentRow(0)
        # self.run_model()

    def open_file(self, filePaths):
        if(isinstance(filePaths, list) == 0):
            filePaths = [filePaths]

        new_item = None

        for filePath in filePaths:
            fileName = os.path.basename(filePath)
            items = self.ui.fileList.findItems(fileName, QtCore.Qt.MatchExactly)
            if len(items) > 0:
                self.ui.statusbar.showMessage("File already opened", 3000)
                if(len(filePaths) == 1):
                    self.ui.fileList.setCurrentItem(items[0])
                continue

            new_item = QtWidgets.QListWidgetItem()
            new_item.setText(fileName)
            new_item.setData(QtCore.Qt.UserRole, {'filePath':filePath})
        
            self.ui.fileList.addItem(new_item)


        if(new_item is not None):
            self.ui.original.setPixmap(QtGui.QPixmap(filePath))
            self.ui.fileList.setCurrentItem(new_item)
            self.ui.original_2.clear()
            self.ui.preview_2.clear()
            
    def parseData(self):
        filePaths = QtWidgets.QFileDialog.getOpenFileNames(self, "Select image", filter="Image files (*.jpg *.png *.bmp *.yaml)")
        # print(filePaths[0])
        
        for filePath in filePaths[0]:

            if filePath.endswith(".yaml"):
                # create read_yaml progress:
                self.yamlProgress.show()

                # disable controls here:

                # run read_yaml on a worker thread:
                self.yamlWorker.filePath = filePath
                self.yamlWorker.moveToThread(self.yamlThread)
                self.yamlThread.started.connect(self.yamlWorker.run)
                self.yamlWorker.finished.connect(self.postParseData)
                self.yamlWorker.finished.connect(self.yamlWorker.deleteLater)
                self.yamlWorker.finished.connect(self.yamlThread.quit)
                self.yamlWorker.finished.connect(self.yamlThread.wait)

                self.yamlThread.start()
            else:
                new_item = None
                fileName = os.path.basename(filePath)
                items = self.ui.fileList.findItems(fileName, QtCore.Qt.MatchExactly)
                if len(items) > 0:
                    self.ui.statusbar.showMessage("File already opened", 3000)

                new_item = QtWidgets.QListWidgetItem()
                new_item.setText(fileName)
                new_item.setData(QtCore.Qt.UserRole, {'filePath':filePath})
                self.ui.fileList.addItem(new_item)

                if(new_item is not None):
                    self.ui.original.setPixmap(QtGui.QPixmap(filePath))
                    self.ui.fileList.setCurrentItem(new_item)
                    self.ui.original_2.clear()
                    self.ui.preview_2.clear()

    def postParseData(self):
        if self.yamlQueue.qsize() > 0:
            self.yamlProgress.hide()
            res = self.yamlQueue.get()
            filePaths, label = res
            self.labels, self.label_eval = label
            self.ui.fileList.clear()
            self.open_file(filePaths)
            self.yamlThread = QThread()
            self.yamlWorker = yamlWorker(self.yamlQueue)
        else:
            assert False
        return 0

    def postParseData(self):
        if self.yamlQueue.qsize() > 0:
            self.yamlProgress.hide()
            res = self.yamlQueue.get()
            filePaths, label = res
            self.labels, self.label_eval = label
            self.ui.fileList.clear()
            self.open_file(filePaths)
            self.yamlThread = QThread()
            self.yamlWorker = yamlWorker(self.yamlQueue)
        else:
            assert False
        return 0

    def reportProgress2(self, n):
        if(n == 3):
            self.ui.progressBar_2.setValue(self.ui.progressBar_2.value() + 1)

    def reportProgress(self, n):
        self.ui.progressBar.setValue(n)

    def flipAllControls(self, value):
        self.ui.addAug.setEnabled(value)
        self.ui.loadAug.setEnabled(value)
        self.ui.saveAug.setEnabled(value)
        #self.ui.demoAug.setEnabled(value)
        self.ui.comboBox.setEnabled(value)
        self.ui.pushButton.setEnabled(value)
        self.ui.pushButton_2.setEnabled(value)
        #self.ui.pushButton_3.setEnabled(value)
        self.ui.pushButton_4.setEnabled(value)
        self.ui.compoundAug.setEnabled(value)
        #self.ui.checkBox_2.setEnabled(value)
        self.ui.upListAug.setEnabled(value)
        self.ui.downListAug.setEnabled(value)
        self.ui.deleteListAug.setEnabled(value)
        self.ui.listAugs.setEnabled(value)
        return 0

    def change_file_selection(self, qListItem):
        if hasattr(self, 'emptyNest'):
            if self.emptyNest:
                self.flipAllControls(True)
                self.emptyNest = False

        if not qListItem is None:
            originalImg = cv2.imread(qListItem.data(QtCore.Qt.UserRole)['filePath'])

            self.ui.augmented.clear()
            self.ui.uncompressed.clear()
            self.ui.original_2.clear()
            self.ui.preview_2.clear()

            originalQtImg = convert_cvimg_to_qimg(originalImg)
            self.ui.original.setPixmap(QtGui.QPixmap.fromImage(originalQtImg))

            self.changePreviewImage()
        else:
            print("INFO: qListItem was None (was it cleared by YAML read function?)")

    def change_uncompressed_selection(self, current):
        if(current == None):
            return

        qListItem = self.ui.fileList.currentItem()
        data = qListItem.data(QtCore.Qt.UserRole)
        originalImg = cv2.imread(data['filePath'])

        dst = data.get('uncompressed_dst')
        
        if(dst is None ):
            return

        class_text = current.text().split(' (')[0]

        if(class_text == "all"):
            dst_qt = convert_cvimg_to_qimg(dst)
            self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(dst_qt))

        else:
            pred = qListItem.data(QtCore.Qt.UserRole)['uncompressed_pred']
            model = models._registry[data['model']]
            imgs = model.draw_single_class(pred, originalImg, class_text)
            qImg_overlay = convert_cvimg_to_qimg(imgs["overlay"])
            self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(qImg_overlay))

            if "segmentation" in imgs:
                qImg_segmentation= convert_cvimg_to_qimg(imgs["segmentation"])
                self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(qImg_segmentation))
        
    def change_augmented_selection(self, current):
        if(current == None):
            return

        qListItem = self.ui.fileList.currentItem()
        data = qListItem.data(QtCore.Qt.UserRole)
        originalImg = cv2.imread(data['filePath'])
        noseImg = self.apply_augmentations(originalImg)

        dst = data.get('augmented_dst')
        
        if(dst is None ):
            return

        class_text = current.text().split(' (')[0]

        if(class_text == "all"):
            dst_qt = convert_cvimg_to_qimg(dst)
            self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(dst_qt))

        else:
            pred = qListItem.data(QtCore.Qt.UserRole)['augmented_pred']
            model = models._registry[data['model']]
            imgs = model.draw_single_class(pred, noseImg, class_text)
            qImg_overlay = convert_cvimg_to_qimg(imgs["overlay"])
            self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(qImg_overlay))

            if "segmentation" in imgs:
                qImg_segmentation= convert_cvimg_to_qimg(imgs["segmentation"])
                self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(qImg_segmentation))

    
    def display_result(self, result):
        # Result[0]:
        # Result[1]: qListItem
        
        qListItem = result[1]
        model_results = result[0]

        self.ui.fileList.setCurrentItem(qListItem)

        data = qListItem.data(QtCore.Qt.UserRole)

        # Code to display the original image with detections
        self.ui.original_2.clear()
        uncompressed = model_results[0]
        data['uncompressed_pred'] = uncompressed['pred']
        data['uncompressed_dst'] = uncompressed['dst']
        data['model'] = uncompressed['model']
        self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(convert_cvimg_to_qimg(uncompressed['dst'])))

        names = uncompressed["listOfNames"]
        data['uncompressed_names'] = names
        cls_unc = uncompressed["classes"] # breaks at everything EXCEPT NAS (for now)
        #print(cls)
		
        for x in names:
            n = "{} ({})".format(x, cls_unc[x])
            i = QtWidgets.QListWidgetItem(n)
            i.setBackground(QtGui.QColor(names[x][0], names[x][1], names[x][2]))
            self.ui.uncompressed.addItem(i)

        augmented = model_results[1]
        data['augmented_pred'] = augmented['pred']
        data['augmented_dst'] = augmented['dst']
        self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(convert_cvimg_to_qimg(augmented['dst'])))

        names = augmented["listOfNames"]
        data['augmented_names'] = names
        cls_aug = augmented["classes"] # breaks at everything EXCEPT NAS (for now)
        for x in names:
            n = "{} ({})".format(x, cls_aug[x])
            i = QtWidgets.QListWidgetItem(n)
            i.setBackground(QtGui.QColor(names[x][0], names[x][1], names[x][2]))
            self.ui.augmented.addItem(i)

        qListItem.setData(QtCore.Qt.UserRole, data)
        self.ui.tabWidget.setCurrentIndex(0)
		

 
    def run_model(self):
        qListItem = self.ui.fileList.currentItem()
        img = cv2.imread(qListItem.data(QtCore.Qt.UserRole).get('filePath'))

        if img is None:
            self.ui.statusbar.showMessage("Import an image first!", 3000)
            return

        noiseImg = self.apply_augmentations(img.copy())

        self.ui.pushButton_2.setEnabled(False)

        self.ui.progressBar.show()
        self.ui.augmented.clear()
        self.ui.uncompressed.clear()
        self.ui.original_2.clear()
        self.ui.preview_2.clear()

        self.thread = QtCore.QThread()
        self.worker = Worker()

        #detectedNames = {"all": [255,255,255]}
        #display_sep = self.ui.checkBox_2.isChecked()
        comboModelType = self.ui.comboBox.currentText()

        self.worker.setup((img, noiseImg), comboModelType, qListItem)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.ui.progressBar.hide)
        self.worker.finished.connect(self.display_result)
        #if(comboModelType == "Semantic Segmentation"):
        self.worker.finished.connect(lambda: self.ui.pushButton_2.setEnabled(True))
        self.worker.progress.connect(self.reportProgress)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def startExperiment(self):
        # fill image paths with dummy inputs for now
        comboModelType = self.ui.comboBox.currentText()

        # check augmentation setup:
        if self.ui.compoundAug.isChecked():
            ret, msg = mainAug.checkArgs()
            if not ret:
                print(msg) # create dialog box saying something is wrong 
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setText(msg)
                msg_box.setWindowTitle("Compound Error")
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec_()
                return -1

        # initialize model (temp; move to thread worker):
        _model = models._registry[comboModelType]
        _model.initialize()
        
        # replace preset list with variable list:
        # assemble active image path list:
        lw = self.ui.fileList
        items = [lw.item(x) for x in range(lw.count())]
        imgPaths = []

        if self.label_eval == 'coco':
            # ask coco api:
            imgPaths = self.labels['coco'].getImgIds()
        else:
            for qListItem in items:
                file_path = qListItem.data(QtCore.Qt.UserRole).get('filePath')
                if(file_path is None):
                    self.ui.statusbar.showMessage("Import an image first!", 3000)
                    return -1
                imgPaths.append(file_path)

        config = ExperimentConfig(mainAug, self.ui.compoundAug.isChecked(), imgPaths, _model, comboModelType, labels=self.labels, labelType=self.label_eval)
        self.experiment = ExperimentDialog(config, self)
        #self.experiment.setModal(True)
        self.experiment.show()
        self.experiment.startExperiment()
        
    
    def setToDefault(self):
        # mainAug.load('default_aug.txt')
        self.addWindow.demoAug()
        # self.addWindow.__applyConfig__()
        # self.addWindow.__updateViewer__()
        # self.addWindow.__reloadAugs__()
        #self.ui.checkBox_2.setChecked(False)
        self.ui.compoundAug.setChecked(False)
        self.ui.comboBox.setCurrentIndex(0)
    
        self.default_img()
        

if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    window = mainWindow()
    window.show()
    window.showMaximized()

    if 'pyi_splash' in sys.modules:
        pyi_splash.close()

    app.exec_()