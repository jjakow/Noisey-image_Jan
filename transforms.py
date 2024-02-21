from PyQt5.QtCore import pyqtSignal, Qt, QSize
from PyQt5.QtGui import QFont, QPalette
from src.utils.qt5extra import CheckState

from PyQt5.QtWidgets import QDialog, QFileDialog, QListWidgetItem, QMessageBox
from PyQt5 import uic
import cv2
import numpy as np
from src.utils import images
from src.transform_funcs import *
import src.transform_funcs as trans

augList = {
    "Size": {"function": sizescale, "default": [1, 2, 3, 4, 5], "example": 1, "limits": trans.__sizescaleCheck__},
    "Gaussian Blur": {"function": gaussian_blur, "default": [1, 2, 3, 4, 5], "example":1, "limits":trans.__gaussianBlurCheck__},
    "Salt and Pepper": {"function": saltAndPapper_noise, "default": [1, 2, 3, 4, 5], "example":2, "limits":trans.__saltPepperCheck__},
    "Contrast": {"function": contrast, "default": [1, 2, 3, 4, 5], "example":1, "limits":trans.__contrastCheck__},
	"Intensity": {"function": dim_intensity, "default": [1, 2, 3, 4, 5], "example":1, "limits":trans.__intensityCheck__},
    "JPG Compression": {"function": jpg_compression, "default": [1, 2, 3, 4, 5], "example": 1, "limits":trans.__jpgCompressionCheck__},

    " ": {"function": passthrough, "default": [], "example": [], "limits": None, "line": "Other Augmentations"},
	#"Size": {"function": sizescale, "default": [0, 1, 2, 3, 4, 5], "example": 1, "limits": trans.__sizescaleCheck__},
    "Flip Axis": {"function": flipAxis, "default": [-1, 0, 1], "example": -1, "limits":trans.__flipAxisCheck__},
    "Barrel": {"function": barrel, "default": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01], "example":0.005, "limits":trans.__barrelCheck__},
    "Fisheye": {"function": fisheye, "default": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "example":0.4, "limits":trans.__fishEyeCheck__},
    "Simple Mosaic": {"function": alternate_mosaic, "default":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "example":2, "limits":trans.__altMosaicCheck__}, # 1x1 - 5x5
    # ==============================
	#" ": {"function": trans.passthrough, "default":[], "example":[],"limits":None, "line": "Blurring Methods"},
    #"Gaussian Blur": {"function": gaussian_blur, "default": [3, 13, 23, 33, 43, 53, 63, 73, 83], "example":33, "limits":trans.__gaussianBlurCheck__},
    #"Intensity": {"function": dim_intensity, "default": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], "example":0.5, "limits":trans.__intensityCheck__},
    "Rain": {"function": rain, "default": [0,1,2], "example":1, "limits":trans.__rainCheck__},
    "Black and White": {"function": black_white, "default":[0,1,2], "example":0, "limits":trans.__blackWhiteCheck__}, 
    "Saturation" : {"function": saturation, "default":[50], "example":50, "limits":trans.__saturationCheck__},
    # ==============================
	#"  ": {"function": trans.passthrough, "default":[], "example":[],"limits":None, "line": "Noising Methods"},
    "Gaussian Noise": {"function": gaussian_noise, "default": [1,10,15,20,25,30,35,40,45,50,55,60], "example":25, "limits":trans.__gaussianNoiseCheck__},
    #"Salt and Pepper": {"function": saltAndPapper_noise, "default": [x/100 for x in range(12)], "example":0.25, "limits":trans.__saltPepperCheck__},
	"Speckle Noise": {"function": speckle_noise, "default": [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2], "example":1.5, "limits":trans.__speckleNoiseCheck__},
	# ==============================
	#"   ": {"function": trans.passthrough, "default":[], "example":[],"limits":None, "line": "Compression Methods"},
    "Bilinear Resizing": {"function": bilinear, "default": [10,20,30,40,50,60,70,80,90,95], "example":25, "limits":trans.__bilinearCheck__},
    #"JPEG Compression": {"function": jpeg_comp, "default": [100,90,80,70,60,50,40,30,20,10], "example":20, "limits":trans.__JPEGCheck__},
    "WebP Compression": {"function": webp_transform, "default": [10,25,50,75,100], "example":10, "limits":trans.__WEBPCheck__},
    "Compressive Autoencoder": {"function": cae, "default": [140,148,156,164,172,180,188,196], "example":172, "limits":trans.__compressiveAutoCheck__},
    "Image H264 Compression": {"function": ffmpeg_h264_to_tmp_video, "default":[1, 2, 3, 4, 5], "example":3, "limits":trans.__h264Check__},
    #"Image H264": {"function": ffmpeg_h264_to_tmp_video, "default":[0,10,20,30,40,50,60,70,80,90,100], "example":60, "limits":trans.__h264Check__},
    "Image H265": {"function": ffmpeg_h265_to_tmp_video, "default":[0,5,10,15,20,25,30,35,40,45,50], "example":45, "limits":trans.__h265Check__}
	
    #"Flip Axis": {"function": flipAxis, "default": [-1], "example": -1, "limits":trans.__flipAxisCheck__},
    # "Simple Mosaic": {"function": simple_mosaic, "default":[], "example":[], "limits":trans.__simpleMosaicCheck__},
    #"Sharpen": {"function": sharpen, "default": [5,6,7,8,9,10,11,12], "example":9, "limits":trans.__sharpenCheck__},
    #"Rotation": {"function": rotation, "default": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], "example":60, "limits":trans.__rotationCheck__},
    #"Color Inversion": {"function": invert, "default": [1], "example":1, "limits":trans.__invertCheck__},
    #"Pincushion": {"function": pincushion, "default": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01], "example":0.005, "limits":trans.__pincushionCheck__},
}

# horizontal line seperator:

class Augmentation:
    """
    Creates and Add Augmentations
    """
    def __init__(self, aug, original_position, args, limit=None, verbose=True) -> None:
        self.__title__ = aug[0]
        self.__run__ = aug[1]
        self.__checked__ = False
        self.__position__ = original_position
        self.__args__ = args[0] # list of values
        self.__example__ = args[1]
        self.__verbose__ = verbose
        self.__limitFunc__ = limit

    @property
    def title(self):
        return self.__title__

    @property
    def enabled(self):
        return self.__checked__

    @property
    def position(self):
        return self.__position__
    
    @property
    def args(self):
        return self.__args__

    @property
    def exampleParam(self):
        return self.__example__

    def setExampleParam(self, value):
        self.__example__ = value

    def __call__(self, image, request_param=None, example=False):
        if example:
            _param = [self.__example__]
            if self.__verbose__:
                if example and not request_param is None:
                    print("WARNING: Request param is ignored since example is set")
        else:
            if not request_param is None:
                _param = [request_param]
                if not request_param in self.__args__:
                    if self.__verbose__:
                        print("WARNING: Requested params not in set arguments. Set verbose to false to dismiss")
            else:
                if self.__verbose__: print("WARNING: No request given. Example is false, so returning example value")
                _param = [self.__example__] 
        return self.__run__(image, *_param)

    def setParam(self, args):
        self.__args__ = args

    def valid(self, param_list):
        if self.__limitFunc__:
            for param in param_list:
                ret = self.__limitFunc__(param)
                if not ret:
                    return False
        return True

class AugmentationPipeline():
    def __init__(self, augList:dict) -> None:
        self.__list__ = augList
        self.__keys__ = []
        self.__augList__ = []
        self.__index__ = 0
        self.__pipeline__ = []
        self.__line_pos__ = []
        self.__line_text__ = []
        self.__wrapper__()

    def __wrapper__(self):
        aug_pos = 0
        for pos, item in enumerate(self.__list__.items()):
            #print(pos)
            if not "line" in item[1]:
                _item = Augmentation( (item[0], item[1]["function"]), aug_pos, args=(item[1]["default"], item[1]["example"]), verbose=False, limit=item[1]['limits'])
                self.__augList__.append(_item)
                self.__keys__.append(item[0])
                aug_pos += 1
            else:
                self.__line_pos__.append(pos)
                self.__line_text__.append(item[1]['line'])
            #print(self.__line_pos__)
            #print(self.__line_text__)
            #print("=====")

    def __len__(self):
        return len(self.__pipeline__)

    def __iter__(self):
        return (self.__pipeline__[x] for x in range(len(self.__pipeline__)))

    def __getitem__(self, key):
        return self.__pipeline__[key]

    def __next__(self):
        self.__index__ += 1
        try:
            return self.__pipeline__[self.__index__-1]
        except IndexError:
            self.__index__ = 0
            raise StopIteration

    def __repr__(self) -> str:
        _out = ''
        for pipe in self.__pipeline__:
            _out += '%s - %s\n'%(pipe.title, pipe.position)
        return _out

    def exists(self, title):
        # make more efficient later:
        for item in self.__pipeline__:
            if title == item.title:
                return True
        return False

    def index(self, title):
        for i, item in enumerate(self.__pipeline__):
            if title == item.title:
                return i
        return -1

    def append(self, aug_title, param=None, example=None):
        augIndex = self.__keys__.index(aug_title)
        augItem = self.__augList__[augIndex]
        if not param is None: 
            augItem.setParam(param)
        if not example is None: augItem.setExampleParam(example)
        self.__pipeline__.append(augItem)

    def remove(self, aug_title):
        augIndex = self.__keys__.index(aug_title)
        print(augIndex, self.__keys__, aug_title)
        for i, aug in enumerate(self.__pipeline__):
            print(aug.title, aug.position)
            if aug.position == augIndex:
                self.__pipeline__.remove(aug)
                break


    def clear(self):
        self.__pipeline__.clear()
        self.__index__ = 0
        return 0

    def load(self, filename):
        # check if filename is a .txt:
        with open(filename, 'r') as f:
            content = list(map(str.strip, f.readlines()))
        oldLen = len(mainAug)
        
        # create list of errors which may arise
        errs = []
        # format: [title,# of parameters,*parameters,1,example]
        for _content in content:
            _content = _content.split(',')
            name = _content[0]
            nargs = int(_content[1])
            params = []

            try:
                for i in range(nargs):
                    params.append( float(_content[i+2]) )
            except ValueError as e:
                print(e)
                errs.append('Parameters listed for %s contains items that are not numbers'%(name))
                break

            params = list(params)
            _example_buffer_loc = nargs+3 #+2 and +1 to get to 1
            
            try:
                example = float(_content[_example_buffer_loc])
            except ValueError:
                errs.append('Example parameter listed in %s is not a valid number'%(name))
                break

            if name in augList:
                _aug  = Augmentation([name, augList[name]['function']], list(augList.keys()).index(name), [params, example], verbose=False, limit=augList[name]['limits'])
                if not _aug.valid(params):
                    errs.append('Stated parameter list in %s contains values out of range'%(name))
                    break

                if not _aug.valid([example]):
                    errs.append('Stated example parameter in %s contains values out of range'%(name))
                    break

                self.__pipeline__.append(_aug)
            else:
                errs.append("Augmentation name %s is not recognized!"%(name))

        if len(errs) == 0:
            # valid file so remove old augs:
            self.__pipeline__ = self.__pipeline__[oldLen:]
        else:
            # reset to old augs:
            self.__pipeline__ = self.__pipeline__[:oldLen]

        return errs

    def save(self, filename):
        if '.txt' not in filename[0]:
            _filename = "%s.txt"%(filename[0])
        else:
            _filename = filename[0]
        
        with open(_filename, 'w') as f:
            for aug in self.__pipeline__:
                aug_title = aug.__title__
                parameters = [str(i) for i in aug.__args__]
                para_out = ','.join(parameters)
                para_length = len(parameters)
                str_out = "%s,%i,%s,1,%f\n"%(aug_title, para_length, para_out,aug.exampleParam)
                f.write(str_out)

    def checkArgs(self):
        maxLen = 0
        for aug in self.__pipeline__:
            if maxLen == 0:
                maxLen = len(aug.args)
            if maxLen != len(aug.args):
                return False, "Compounding augmentations require equal number of parameters for each active parameter. %s has mismatch of %i parameters"%(aug.title, len(aug.args))
        return True, ""

    next = __next__ # python 2

class AugDialog(QDialog):
    pipelineChanged = pyqtSignal(object)

    def __init__(self, listViewer, parent):
        # Config tells what noises are active, what the parameters are
        super(AugDialog, self).__init__(parent)
        self.__viewer__ = listViewer # outside of the Augmentation Dialog UI
        self.lastRow = 0
        uic.loadUi('./src/qt_designer_file/dialogAug.ui', self)
        #self.listWidget.setStyleSheet( "QListWidget::item { border-bottom: 1px solid black; }" )
        self.__loadAugs__()
        self.__loadEvents__()
        self.defaultImage = 'imgs/default_imgs/100FACES.jpg'
        self.__loadInitialImage__()
        self.__loadExample__()
        self.savedAugPath = './src/data/saved_augs'
        self.__applyConfig__()
        self.hideCheckBox()
        _btn1, _btn2 = self.buttonBox.buttons() # ok, cancel
        _btn1.clicked.connect(self.__applySelection__)
        _btn2.clicked.connect(self.close)
        # example rerun:
        self.runExample.clicked.connect(lambda: self.__loadAugSelection__(self.listWidget.currentItem()))
        self.listWidget.setStyleSheet( "QListWidget::item { color: rgb(0,0,0) }" );

    def __loadEvents__(self):
        self.listWidget.itemClicked.connect(self.__loadAugSelection__)

    def __augError__(self, errs: list):
        errsStr = ''
        if len(errs) == 1:
            errsStr += errs[0]
        else:
            for i in errs:
                if i == errs[len(errs) - 1] and len(errs) != 1:
                    errsStr += ('and ' + i)
                else: errsStr += (i + ', ')

        errorBox = QMessageBox()
        errorBox.setWindowTitle("Error")
        errorBox.setIcon(QMessageBox.Critical)
        errorBox.setText("Incorrect parameters! Check the following augmentations: \n" + errsStr + ".")
        x = errorBox.exec_()

    def __loadAugs__(self):
        j = 0
        _title_font = QFont()
        _title_font.setBold(True)
        
        for i, aug in enumerate(mainAug.__augList__):
            if i in mainAug.__line_pos__:
                _line = QListWidgetItem()
                _line.setFlags(Qt.NoItemFlags)
                _line.setSizeHint(QSize(-1, 20))
                _line.setFont(_title_font)
                _line.setText(mainAug.__line_text__[j])
                self.listWidget.addItem(_line)
                j += 1

            _item = QListWidgetItem()
            _item.setText(aug.title)
            _item.setCheckState(CheckState.Unchecked)
            _item.setData(Qt.UserRole, [aug, "", ""]) # aug, parameters, example
            self.listWidget.addItem(_item)
    
    def __reloadAugs__(self):
        for i in range(self.listWidget.count()):
            _payload = self.listWidget.item(i).data(Qt.UserRole)
            if not _payload is None:
                strArgs = [str(k) for k in augList[self.listWidget.item(i).text()]["default"]]
                parameters = ",".join(strArgs)
                if self.listWidget.item(i).text() != " " and self.listWidget.item(i).text() != "  " and self.listWidget.item(i).text() != "   ":
                    _payload[1] = parameters
                    _payload[2] = str(augList[self.listWidget.item(i).text()]["example"])
                self.listWidget.item(i).setData(Qt.UserRole, _payload)
                if i == (self.listWidget.currentRow()):
                    self.noiseRange.setText(parameters)
                    self.exampleLine.setText(str(augList[self.listWidget.item(i).text()]["example"]))
    
    def __loadInitialImage__(self):
        self._img = cv2.imread(self.defaultImage)
        h,w,_ = self._img.shape
        new_h = 500
        new_w = int((new_h/h)*w)
        self._img = cv2.resize(self._img, (new_w, new_h))

    def __loadExample__(self):
        # Assuming default image:
        _copy = np.copy(self._img)
        qtImage = images.convertCV2QT(_copy, 1000, 500)
        self.previewImage.setPixmap(qtImage)
        self.__loadAugSelection__(self.listWidget.itemAt(0,0))
        self.listWidget.setCurrentItem(self.listWidget.itemAt(0,0))

    def __loadAugSelection__(self, aug):
        # update old active aug:
        errs = []
        _payload = self.listWidget.item(self.lastRow).data(Qt.UserRole)
        _payload[1] = self.noiseRange.text()
        _payload[2] = self.exampleLine.text()

        self.listWidget.item(self.lastRow).setData(Qt.UserRole, _payload)
        
        # change GUI when item is clicked
        currentItem = aug.text()
        if currentItem != "" and currentItem in mainAug.__keys__:
            _payload = aug.data(Qt.UserRole)
            augIndex = mainAug.__keys__.index(currentItem)
            augItem = mainAug.__augList__[augIndex]
            augIndex2 = self.listWidget.row(aug)
            
            if _payload[1] == '': 
                strArgs = [ str(i) for i in augItem.args]
                parameters = ",".join(strArgs)
            else: parameters = _payload[1]
            
            if _payload[2] == '':
                example = augItem.exampleParam
            else: example = _payload[2]

            # GUI range controls:
            self.noiseRange.setText(parameters)
            self.exampleLine.setText(str(example))

            try:
                newExampleValue = float(example)
                if not augItem.valid([newExampleValue]):
                    errs.append("Example value not in the valid range")
                augItem.setExampleParam(newExampleValue)
            except ValueError:
                errs.append('Example value not a number')

            print(errs)
            if len(errs) != 0:
                self.__augError__(errs)
            else:
                _copy = np.copy(self._img)
                _copy = augItem(_copy, example=True)
                qtImage = images.convertCV2QT(_copy, 1000, 500)
                self.previewImage.setPixmap(qtImage)
                self.lastRow = augIndex2
        
        if currentItem == "Size":
            self.info_label.setText("Size reduces the image dimension scale by 50% per each augmentation level.")
		
        if currentItem == "Intensity":
            self.info_label.setText("Dims the intensity of the image by the given factor/range of factor.")
        
        if currentItem == "Gaussian Noise":
            self.info_label.setText("Gaussian Noise is a statistical noise having a probability density function equal to normal distribution with a given standard deviation and the mean of 2, also known as Gaussian Distribution. Random Gaussian function is added to Image function to generate this noise.")
        
        if currentItem == "Gaussian Blur":
            self.info_label.setText("It blurs the image using the kernel of the given size. (A kernel, in this context, is a small matrix which is combined with the image using a mathematical technique: convolution). In a Gaussian blur, the pixels nearest the center of the kernel are given more weight than those far away from the center. This averaging is done on a channel-by-channel basis, and the average channel values become the new value for the pixel in the filtered image.")
        
        if currentItem == "JPEG Compression":
            self.info_label.setText("The JPEG compression is a block based compression. The data reduction is done by the subsampling of the color information, the quantization of the DCT-coefficients and the Huffman-Coding (reorder and coding). The user can control the amount of image quality loss due to the data reduction by setting (or chose presets).")
        
        if currentItem == "Normal Compression":
            self.info_label.setText("It resizes an image, scale it along each axis (height and width), considering the specified scale factors")
        
        if currentItem == "Salt and Pepper":
            self.info_label.setText("Add salt and pepper noise to image (ie. (makes some of the pixels completply black or white) with given probability of the noise (prob)")
        
        if currentItem == "Flip Axis":
            self.info_label.setText("Based on the mode, it flips around the axis. If mode > 0, Flips along vertical axis. If mode = 0,  Flips along horizontal axis. If mode < 0, Flips along both axes")
        
        if currentItem == "Fisheye":
            self.info_label.setText("Fisheye transformation distorts the image pixels weighted by the euclidean distance from the given center of the transformation. the distortion factor controls the amount of distortion used for the transformation.")
        
        if currentItem == "Barrel":
            self.info_label.setText("Barrel Distortion folds the image inwards and introduces black regions on the outside where image information is missing. The given distortion factor controls the amount of distortion used for the transformation.")
        
        if currentItem == "Simple Mosaic":
            self.info_label.setText("Simple mosaic combines 4 training images into one in certain ratios")
        
        if currentItem == "Black and White":
            self.info_label.setText("Black and White noise changes the image into the corresponding gray scale image.")
        
        if currentItem == "Speckle Noise":
            self.info_label.setText("Speckle is a granular noise that inherently exists in an image and degrades its quality. It can be generated by multiplying the noise values with different pixels of an image. Noise function has a probability density function equal to normal distribution with a given standard deviation and the mean of 2.")
        
        if currentItem == "Saturation":
            self.info_label.setText("Saturation impacts the color intensity of the image, making it more vivid or muted depending on the value.")

        if currentItem == "Alternate Mosaic":
            self.info_label.setText("Alternate Mosaic slice the given image into n by n parts and generated the new image by shuffling those slices.")
        
        if currentItem == "Image H264":
            self.info_label.setText("Apply H264 compression on a single image (parameter is managed by a constant quantization parameter).")

    # change GUI to match mainAug
    def __applyConfig__(self):
        # update config given:
        for i in range(self.listWidget.count()):
            if not i in mainAug.__line_pos__:
                listItem = self.listWidget.item(i)
                listItem.setCheckState(CheckState.Unchecked)
        else:
            for aug in mainAug:
                itemPos = aug.position
                listItem = self.listWidget.item(itemPos)
                listItem.setCheckState(CheckState.Checked)

    def show(self):
        self.__applyConfig__()
        return super().show()

    def closeEvent(self, event):
        for i in range(self.listWidget.count()):
            listItem = self.listWidget.item(i)
            payload = listItem.data(Qt.UserRole)
            if not payload is None: 
                payload[1] = ''; payload[2] = ''
                self.listWidget.item(i).setData(Qt.UserRole, payload)
        event.accept()

    # change mainAug to match selected items from GUI:
    def __applySelection__(self):
        # update the active item:
        cr = self.listWidget.currentRow()
        _payload = self.listWidget.item(cr).data(Qt.UserRole)
        _payload[1] = self.noiseRange.text()
        _payload[2] = self.exampleLine.text()
        self.listWidget.item(cr).setData(Qt.UserRole, _payload)
        errs = []

        # get checks from listWidget:
        for i in range(self.listWidget.count()):
            listItem = self.listWidget.item(i)
            augIndex = -1
            for pos, item in enumerate(mainAug.__augList__):
                if listItem.text() == item.title:
                    augIndex = pos
                    break 

            # parse the list items:
            _payload = listItem.data(Qt.UserRole)
            if not _payload is None:
                _noiseRange = _payload[1]
                _example = _payload[2]

                if _noiseRange != '':
                    try:
                        _param = [float(j) for j in _noiseRange.split(',')]
                    except ValueError:
                        errs.append("%s - One value is not a number"%listItem.text())
                        continue
                    
                    if not mainAug.__augList__[augIndex].valid(_param) and listItem.checkState() == Qt.Checked:
                        #print("RAHH!!! WRONG PARAMETER!!")
                        errs.append("%s - Incorrect range"%listItem.text())
                        continue
                else: _param = None

                if _example != '': 
                    _example = float(_payload[-1])
                else: _example = None

                itemIndex = mainAug.index(listItem.text())

                if listItem.checkState() and itemIndex == -1:
                    mainAug.append(listItem.text(), param=_param, example=_example)
                elif listItem.checkState() and itemIndex != -1:
                    if not _param is None: mainAug.__pipeline__[itemIndex].setParam(_param)
                    if not _example is None: mainAug.__pipeline__[itemIndex].setExampleParam(_example)
                elif not listItem.checkState(): # make more efficient later
                    for item in mainAug.__pipeline__:
                        if item.title == listItem.text():
                            mainAug.remove(listItem.text())
                            break
        
        if len(errs) == 0:
            self.__updateViewer__()
        else:
            self.__augError__(errs)

    def __updateViewer__(self):
        # add listviewer:
        self.__viewer__.clear()
        for item in mainAug:
            self.__viewer__.addItem(item.title)
        self.pipelineChanged.emit(None)

    def __loadFileDialog__(self):
        _file = QFileDialog.getOpenFileName(self, "Load in Augmentation", self.savedAugPath, '*.txt')
        if _file[0] != '':
            errs = mainAug.load(_file[0])
            if len(errs) != 0:
                self.__augError__(errs)
            else:
                self.__applyConfig__() # change GUI
                self.__updateViewer__()

    def __saveFileDialog__(self):
        save_path = QFileDialog.getSaveFileName(self, 'Save Current Augmentation', self.savedAugPath, '*.txt')
        mainAug.save(save_path)

    def __deleteItem__(self):
        selected_item = self.__viewer__.currentItem()
        if selected_item is not None:
            for item in mainAug:
                if item.title == selected_item.text():
                    mainAug.remove(item.title)
                    self.__updateViewer__()
                    return 0

    def __moveDown__(self):
        selected_idx = self.__viewer__.currentRow()
        if selected_idx != -1:
            if selected_idx != len(mainAug)-1:
                #print("running!")
                item = self.__viewer__.takeItem(selected_idx)
                #print(item)
                #self.__viewer__.removeItemWidget(item)
                mainAug.__pipeline__.insert(selected_idx+1, mainAug.__pipeline__.pop(selected_idx))

                self.__viewer__.insertItem(selected_idx+1, item)
                self.__updateViewer__()
                self.__viewer__.setCurrentRow(selected_idx+1)

    def __moveUp__(self):
        selected_idx = self.__viewer__.currentRow()
        if selected_idx != -1:
            if selected_idx != 0:
                #print("running!")
                item = self.__viewer__.takeItem(selected_idx)
                #print(item)
                #self.__viewer__.removeItemWidget(item)
                mainAug.__pipeline__.insert(selected_idx-1, mainAug.__pipeline__.pop(selected_idx))

                self.__viewer__.insertItem(selected_idx-1, item)
                self.__updateViewer__()
                self.__viewer__.setCurrentRow(selected_idx-1)

    def demoAug(self):
        mainAug.clear()
        # mainAug.append('Gaussian Blur')
        # mainAug.append('Gaussian Noise')
        #mainAug.append('JPEG Compression')
        mainAug.append('Salt and Pepper')
        self.__updateViewer__()
    
    def augParameterError(self, errs: list):
        print("failed to save noise value and/or example of augmentation(s)")
        errsStr = ''
        for i in errs:
            if i == errs[len(errs) - 1] and len(errs) != 1:
                errsStr += ('and ' + i)
            elif len(errs) == 1: errsStr += i
            else: errsStr += (i + ', ')

        errorBox = QMessageBox()
        errorBox.setWindowTitle("Error")
        errorBox.setIcon(QMessageBox.Critical)
        errorBox.setText("At least one of the augmentations has an illegal character in its parameters. Only numbers and commas should be used for a noise range, and only numbers should be used for an example. Check the following augmentations: " + errsStr + ".")
        x = errorBox.exec_()
    
    def hideCheckBox(self):
        for i in range(self.listWidget.count()):
            #print(i)
            if self.listWidget.item(i).text() == " " or self.listWidget.item(i).text() == "  " or self.listWidget.item(i).text() == "   ":
                row = i
                self.listWidget.item(i).setFlags(Qt.NoItemFlags)

# Augmentation holder:
mainAug = AugmentationPipeline(augList)