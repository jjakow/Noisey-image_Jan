import abc
import os
from pathlib import Path
from pyexpat import model
from types import SimpleNamespace
import os, csv, torch
# import scipy.io
import numpy as np
import cv2


# if __name__ == '__main__':
#     import sys
#     sys.path.append('./')

# move imports into their own __init__.py file

# import seg network here:
# from src.predict_img import new_visualize_result, process_img, predict_img, load_model_from_cfg, visualize_result, transparent_overlays
# from src.mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion

# import yolov3 stuff:
import src.obj_detector.detect as detect
from src.obj_detector.models import load_model
from src.obj_detector.utils.utils import load_classes

# mtcnn:
from src.mtcnn import detector as mtcnn_detector
from src.mtcnn import visualization_utils as mtcnn_utils
from PIL import Image

# import mAP eval:
from src.evaluators.map_metric.lib.BoundingBoxes import BoundingBox
from src.evaluators.map_metric.lib import BoundingBoxes
from src.evaluators.map_metric.lib.Evaluator import *
from src.evaluators.map_metric.lib.utils import BBFormat

# import efficientNetV2
# from src.effdet import create_model
# from timm.utils import AverageMeter
# from torchvision import transforms

# import yolov4 stuff:
from src.yolov4.utils.torch_utils import select_device
from src.yolov4.models.models import Darknet
from src.yolov4.models.models import load_darknet_weights
import src.yolov4.utils.datasets as yolov4_datasets
from src.yolov4.utils.general import non_max_suppression, scale_coords

# import yolov3-ultralytics here:
# from src.yolov3 import detect
from src.yolov3.utils.plots import Annotator, Colors
from src.yolov3.utils.augmentations import letterbox
from src.yolov3.models.common import DetectMultiBackend
import yaml

# import compressive autoendcoding here:
from src.cae.src.data_loader import preprocess_single as cae_preprocess_single
from src.cae.src.models.cae_32x32x32_zero_pad_bin import CAE

# YOLOX imports:
from src.yolox.yolox.data.datasets import COCO_CLASSES
from src.yolox.yolox.data.data_augment import ValTransform
from src.yolox.yolox.exp import get_exp
from src.yolox.yolox.utils import fuse_model, get_model_info, postprocess, vis
from src.yolox.yolox.utils.visualize import _COLORS
from src.yolox.exps.default.yolox_m import Exp

# YoloNAS imports
import super_gradients
from super_gradients.training import models
from super_gradients.common.object_names import Models

currPath = str(Path(__file__).parent.absolute()) + '/'

class Model(abc.ABC):
    """
    Creates and adds models. 
    Requirment: The network needs to be fitted in four main funtions: run, initialize, deinitialize, and draw.   
    """
    def __init__(self, *network_config) -> None:
        self.complexOutput = False
        self.isCOCO91 = False
    
    @abc.abstractclassmethod
    def run(self, input):
        raise NotImplementedError

    @abc.abstractclassmethod
    def initialize(self, *kwargs):
        raise NotImplementedError

    @abc.abstractclassmethod
    def deinitialize(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def draw(self, pred):
        raise NotImplementedError

    @abc.abstractclassmethod
    def draw_single_class(self, pred, img, selected_class):
        raise NotImplementedError

    @abc.abstractclassmethod
    def report_accuracy(self):
        raise NotImplementedError

    @abc.abstractproperty
    def outputFormat(self):
        raise NotImplementedError


    def __call__(self):
        pred = self.run()
        return pred

class MTCNN(Model):
    def __init__(self, *network_config) -> None:
        super().__init__(*network_config)
        self.network_config = network_config
        self.complexOutput = False
        self.isCOCO91 = False
    
    def run(self, input):
        _img = Image.fromarray(input)
        detections, _ = mtcnn_detector.detect_faces(self.model, _img)
        cls_matrix = np.zeros((detections.shape[0],1))
        detections = np.concatenate((detections, cls_matrix), axis=1)
        return detections

    def initialize(self):
        self.model = mtcnn_detector.load_model(*self.network_config)

    def deinitialize(self):
        return -1

    def draw(self, pred, img):
        _img = Image.fromarray(img)
        _img = mtcnn_utils.show_bboxes(_img, pred)
        img = np.array(_img)
        return {"dst": img, "listOfNames": {"all": [255,255,255], "face": [255,0,0]}}

    def draw_single_class(self, pred, img, selected_class):
        img = self.draw(pred, img)["dst"]
        return {"overlay": img}

    def report_accuracy(self, pred:list, gt:list, evalType='voc'):
        """Function takes in prediction boxes and ground truth boxes and
        returns the mean average precision (mAP) @ IOU 0.5 under VOC2007 criteria (default).
        Args:
            pred (list): A list of BoundingBox objects representing each detection from method
            gt (list): A list of BoundingBox objects representing each object in the ground truth
        Returns:
            mAP: a number representing the mAP over all classes for a single image.
        """        
        if len(pred) == 0: return 0

        allBoundingBoxes = BoundingBoxes()
        evaluator = Evaluator()

        # loop through gt:
        for _gt in gt:
            assert type(_gt) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_gt)))
            allBoundingBoxes.addBoundingBox(_gt)

        for _pred in pred:
            assert type(_pred) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_pred)))
            allBoundingBoxes.addBoundingBox(_pred)

        #for box in allBoundingBoxes:
        #    print(box.getAbsoluteBoundingBox(format=BBFormat.XYWH), box.getBBType()) 
        if evalType == 'voc':
            metrics = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
        elif evalType == 'coco':
            assert False
        else: assert False, "evalType %s not supported"%(evalType) 
        return metrics[0]['AP']

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

# class Segmentation(Model):
#     """
#     Segmentation Model that inherits the Model class
#     It specifies its four main functions: run, initialize, deinitialize, and draw. 
#     """
#     def __init__(self, *network_config) -> None:
#         super().__init__(network_config)
        
#         self.complexOutput = True
#         self.isCOCO91 = False
#         self.cfg, self.colors = network_config
#         #self.cfg = str(Path(__file__).parent.absolute()) + "/config/ade20k-hrnetv2.yaml"
#         # colors
#         #self.colors = scipy.io.loadmat(str(Path(__file__).parent.absolute()) + '/data/color150.mat')['colors']
#         self.names = {}
#         self.complexOutput = True # output is a large matrix. Saving output is a little different than object detector

#         with open(str(Path(__file__).parent.absolute()) + '/data/object150_info.csv') as f:
#             reader = csv.reader(f)
#             next(reader)
#             for row in reader:
#                 self.names[int(row[0])] = row[5].split(";")[0]

#     def run(self, input):
#         if torch.cuda.is_available():
#             self.segmentation_module.cuda()
#         else:
#             self.segmentation_module.cpu()

#         img_original, singleton_batch, output_size = process_img(frame = input)

#         try:
#             # predict
#             img_original, singleton_batch, output_size = process_img(frame = input)
#             pred = predict_img(self.segmentation_module, singleton_batch, output_size)
#         except:
#             self.segmentation_module.cpu()

#             print("Using cpu")

#             # predict
#             img_original, singleton_batch, output_size = process_img(frame = input, cpu = 1)
#             pred = predict_img(self.segmentation_module, singleton_batch, output_size)
#         return pred

#     def initialize(self, *kwargs):
#         # Network Builders
#         print("parsing {}".format(self.cfg))
#         self.segmentation_module = load_model_from_cfg(self.cfg)
        
#         self.segmentation_module.eval()
#         return 0

#     def deinitialize(self):
#         return -1

#     def draw(self, pred, img):
#         detectedNames = {"all": [255,255,255]}
#         pred_color, org_pred_split = visualize_result(img, pred, self.colors)

#         #color_palette = get_color_palette(pred, org_pred_split.shape[0], self.names, self.colors, detectedNames)

#         # transparent pred on org
#         dst = transparent_overlays(img, pred_color, alpha=0.6)

#         return {"dst": dst, 
#         "segmentation": pred_color, 
#         "listOfNames":detectedNames
#                 }

#     def draw_single_class(self, pred, img, selected_class):
#         imgs = new_visualize_result(pred, img, selected_class)
#         return {"segmentation": imgs[0], "overlay": imgs[1]}

#     def report_accuracy(self, pred, pred_truth):
#         acc_meter = AverageMeter()
#         intersection_meter = AverageMeter()
#         union_meter = AverageMeter()

#         acc, pix = accuracy(pred, pred_truth)
#         intersection, union = intersectionAndUnion(pred, pred_truth, 150)
#         acc_meter.update(acc, pix)
#         intersection_meter.update(intersection)
#         union_meter.update(union)
        
#         class_ious = {}
#         iou = intersection_meter.sum / (union_meter.sum + 1e-10)
#         for i, _iou in enumerate(iou):
#             class_ious[i] = _iou
#         return iou.mean(), acc_meter.average(), class_ious

#     def outputFormat(self):
#         return "{}" # hex based output?

#     def calculateRatios(self, dets):
#         values, counts = np.unique(dets, return_counts=True)
#         total_idx = [i for i in range(150)]
#         for idx in total_idx:
#             if not idx in values:
#                 counts = np.insert(counts, idx, 0)
#         return counts

class YOLOv3(Model):
    """
    YOLO Model that inherits the Model class
    It specifies its four main functions: run, initialize, deinitialize, and draw. 
    """
    def __init__(self, *network_config) -> None:
        super(YOLOv3, self).__init__()
        # network_config: CLASSES, CFG, WEIGHTS
        self.CLASSES, self.CFG, self.WEIGHTS = network_config
        self.isCOCO91 = False
        # self.CLASSES = os.path.join(currPath, 'obj_detector/cfg', 'coco.names')
        # self.CFG = os.path.join(currPath, 'obj_detector/cfg', 'yolov3.cfg')
        # self.WEIGHTS = os.path.join(currPath,'obj_detector/weights','yolov3.weights')
        print(self.CLASSES, self.CFG, self.WEIGHTS)
        self.classes = load_classes(self.CLASSES)
        self.img_size = 416
        self.conf_thres = 0.5
        self.nms_thres = 0.5

    def run(self, input):
        pred = detect.detect_image(self.yolo, input, img_size=self.img_size, conf_thres=self.conf_thres, nms_thres=self.nms_thres)
        return pred #[x1,y1,x2,y2,conf,class] <--- box

    def initialize(self, *kwargs):
        self.yolo = load_model(self.CFG, self.WEIGHTS)
        return 0
    
    def deinitialize(self):
        return -1
    
    def draw(self, pred, img):
        np_img, detectedNames = detect._draw_and_return_output_image(img, pred, 416, self.classes)
        return {"dst": np_img,
                "listOfNames":detectedNames}

    def draw_single_class(self, pred, img, selected_class):
        np_img = detect._draw_and_return_output_image_single_class(img, pred, selected_class, self.classes)
        return {"overlay": np_img}

    def report_accuracy(self, pred:list, gt:list, evalType='voc'):
        """Function takes in prediction boxes and ground truth boxes and
        returns the mean average precision (mAP) @ IOU 0.5 under VOC2007 criteria (default).
        Args:
            pred (list): A list of BoundingBox objects representing each detection from method
            gt (list): A list of BoundingBox objects representing each object in the ground truth
        Returns:
            mAP: a number representing the mAP over all classes for a single image.
        """        
        if len(pred) == 0: return 0

        allBoundingBoxes = BoundingBoxes()
        evaluator = Evaluator()

        # loop through gt:
        for _gt in gt:
            assert type(_gt) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_gt)))
            allBoundingBoxes.addBoundingBox(_gt)

        for _pred in pred:
            assert type(_pred) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_pred)))
            allBoundingBoxes.addBoundingBox(_pred)

        image = np.zeros((1400,1607,3), dtype=np.uint8)
        out_image = allBoundingBoxes.drawAllBoundingBoxes(image, '100faces')
        cv2.imwrite('test.png', out_image)

        #for box in allBoundingBoxes:
        #    print(box.getAbsoluteBoundingBox(format=BBFormat.XYWH), box.getBBType()) 
        if evalType == 'voc':
            metrics = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
            print(metrics)
        elif evalType == 'coco':
            assert False
        else: assert False, "evalType %s not supported"%(evalType) 
        return metrics[0]['AP']

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

class EfficientDetV2(Model):
    '''
    augmentation values
    GN: 1,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105
    '''
    def __init__(self, *network_config) -> None:
        super(EfficientDetV2, self).__init__()

        # network_config: CLASSES, CFG, WEIGHTS
        self.isCOCO91 = True
        self.CLASSES, self.CFG = network_config
        self.numClasses = len(self.CLASSES)
        print(self.CLASSES, self.CFG)
        self.conf_thres = 0.25
        self.classes = load_classes(self.CLASSES)
        self.inputTrans = {
            'efficientdetv2_dt': (768, 768),
            'efficientdet_d1': (640, 640),
            'tf_efficientdet_d1': (640, 640),
            'efficientdet_d2': (768, 768),
            'tf_efficientdetv2_ds': (1024, 1024),
            'efficientdetv2_dt': (768, 768),
            'tf_efficientdet_d7x': (1536, 1536),
            'tf_efficientdet_d4': (1024, 1024),
            'efficientdet_d0': (512, 512),
            'tf_efficientdet_d0': (512, 512),
            'tf_efficientdet_d0_ap': (512, 512)
        }

    def initialize(self, *kwargs):
        self.bench = create_model(
            self.CFG,
            bench_task='predict',
            #num_classes=len(self.CLASSES),
            pretrained=True,
        )
        self.bench.eval()

    def run(self, input):
        with torch.no_grad():
            # transform image and predict
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=self.inputTrans[self.CFG]),])
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            scores = self.bench(self.transforms(input).unsqueeze(0))

            # resize to match original image
            scores = scores[0].detach().numpy()
            scores[:, 0] = scores[:, 0] / self.inputTrans[self.CFG][1] * input.shape[1]
            scores[:, 1] = scores[:, 1] / self.inputTrans[self.CFG][0] * input.shape[0]
            scores[:, 2] = scores[:, 2] / self.inputTrans[self.CFG][1] * input.shape[1]
            scores[:, 3] = scores[:, 3] / self.inputTrans[self.CFG][0] * input.shape[0]
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu()
            return scores[np.where(scores[:,4] > self.conf_thres)]

    def deinitialize(self):
        return -1

    def draw(self, pred, img):
        np_img, detectedNames = detect._draw_and_return_output_image(img, pred, 416, self.classes)
        return {"dst": np_img,
                "listOfNames":detectedNames}

    def draw_single_class(self, pred, img, selected_class):
        np_img = detect._draw_and_return_output_image_single_class(img, pred, selected_class, self.classes)
        return {"overlay": np_img}

    def report_accuracy(self, pred, pred_truth):
        acc_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()

        acc, pix = accuracy(pred, pred_truth)
        intersection, union = intersectionAndUnion(pred, pred_truth, 150)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        
        class_ious = {}
        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        for i, _iou in enumerate(iou):
            class_ious[i] = _iou
        return iou.mean(), acc_meter.average(), class_ious
      
    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

# class DETR(Model):
#     """
#     DETR Model that inherits the Model class
#     It specifies its four main functions: run, initialize, deinitialize, and draw.
#     """
#     def __init__(self, *network_config) -> None:
#         super(DETR, self).__init__(network_config)
#         self.CLASSES, self.WEIGHTS = network_config[0], network_config[1]
#         print(self.CLASSES, self.WEIGHTS)
#         self.classes = load_classes(self.CLASSES)
#         self.conf_thres = 0.25
#         self.isCOCO91 = True
    
#     def initialize(self, *kwargs):
#         self.model = DETRdemo(num_classes=len(self.classes))
#         self.model.load_state_dict(torch.load(self.WEIGHTS, map_location=torch.device('cpu')))
#         self.model.eval()
#         if torch.cuda.is_available():
#             self.model.cuda()
#             self.on_gpu = True
#         else:
#             self.model.cpu()
#             self.on_gpu = False
#         self.transform = transforms.Compose([
#             transforms.Resize(800, max_size=1333),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         return 0

#     def run(self, input):
#         with torch.no_grad():
#             img = Image.fromarray(input)
#             #if self.on_gpu: img = img.cuda()
#             scores, boxes = self.model.detect(img, self.model, self.transform, threshold=self.conf_thres)
#             _confidences, _classes = torch.max(scores, axis=1)
#             _cls_conf = torch.cat((torch.unsqueeze(_confidences, axis=0), torch.unsqueeze(_classes, axis=0)))
#             _cls_conf = torch.transpose(_cls_conf, 0,1)
#             pred = torch.cat((boxes, _cls_conf), axis=1) #[x1,y1,x2,y2,conf,class] <--- box
#             pred = pred.cpu()
#             return pred

#     def deinitialize(self):
#         return -1

#     def draw(self, pred, img):
#         np_img, detectedNames = detect._draw_and_return_output_image(img, pred, 416, self.classes)
#         return {"dst": np_img,
#                 "listOfNames":detectedNames}

#     def draw_single_class(self, pred, img, selected_class):
#         np_img = detect._draw_and_return_output_image_single_class(img, pred, selected_class, self.classes)
#         return {"overlay": np_img}

#     def report_accuracy(self, pred:list, gt:list, evalType='voc'):
#         """Function takes in prediction boxes and ground truth boxes and
#         returns the mean average precision (mAP) @ IOU 0.5 under VOC2007 criteria (default).
#         Args:
#             pred (list): A list of BoundingBox objects representing each detection from method
#             gt (list): A list of BoundingBox objects representing each object in the ground truth
#         Returns:
#             mAP: a number representing the mAP over all classes for a single image.
#         """        
#         if len(pred) == 0: return 0

#         allBoundingBoxes = BoundingBoxes()
#         evaluator = Evaluator()

#         # loop through gt:
#         for _gt in gt:
#             assert type(_gt) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_gt)))
#             allBoundingBoxes.addBoundingBox(_gt)

#         for _pred in pred:
#             assert type(_pred) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_pred)))
#             allBoundingBoxes.addBoundingBox(_pred)

#         image = np.zeros((1400,1607,3), dtype=np.uint8)
#         out_image = allBoundingBoxes.drawAllBoundingBoxes(image, '100faces')
#         cv2.imwrite('test.png', out_image)

#         #for box in allBoundingBoxes:
#         #    print(box.getAbsoluteBoundingBox(format=BBFormat.XYWH), box.getBBType()) 
#         if evalType == 'voc':
#             metrics = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
#             print(metrics)
#         elif evalType == 'coco':
#             assert False
#         else: assert False, "evalType %s not supported"%(evalType) 
#         return metrics[0]['AP']

#     def outputFormat(self):
#         return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"


class YOLOv4(Model):
    """
    YOLOv4 Model that inherits the Model class
    It specifies its four main functions: run, initialize, deinitialize, and draw. 
    """
    def __init__(self, *network_config) -> None:
        super(YOLOv4, self).__init__()
        # network_config: CLASSES, CFG, WEIGHTS
        self.isCOCO91 = False
        self.CLASSES, self.CFG, self.WEIGHTS = network_config
        self.img_size = (416,416)
        print(self.CLASSES, self.CFG, self.WEIGHTS)
        self.classes = load_classes(self.CLASSES)
        self.conf_thres = 0.25

    def run(self, input):
        im0 = np.copy(input)
        img = yolov4_datasets.letterbox(input, self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        if torch.cuda.is_available():
            img = torch.from_numpy(img).cuda()
        else:
            img = torch.from_numpy(img).cpu()
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.yolo(img, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, 0.6, classes=None, agnostic=False)
        #return pred #[x1,y1,x2,y2,conf,class] <--- box
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        if torch.cuda.is_available():
            pred[0] = pred[0].cpu()
        pred = pred[0].detach().numpy()
        #print(self.yolo.training)
        return pred

    def initialize(self, *kwargs):
        self.yolo = Darknet(self.CFG, self.img_size)
        if torch.cuda.is_available():
            self.yolo = self.yolo.cuda()
        else:
            self.yolo = self.yolo.cpu()
        load_darknet_weights(self.yolo, self.WEIGHTS)
        self.yolo.eval()
    
    def deinitialize(self):
        return -1
    
    def draw(self, pred, img):
        np_img, detectedNames = detect._draw_and_return_output_image(img, pred, 416, self.classes)
        return {"dst": np_img,
                "listOfNames":detectedNames}

    def draw_single_class(self, pred, img, selected_class):
        np_img = detect._draw_and_return_output_image_single_class(img, pred, selected_class, self.classes)
        return {"overlay": np_img}

    def report_accuracy(self, pred:list, gt:list, evalType='voc'):
        """Function takes in prediction boxes and ground truth boxes and
        returns the mean average precision (mAP) @ IOU 0.5 under VOC2007 criteria (default).
        Args:
            pred (list): A list of BoundingBox objects representing each detection from method
            gt (list): A list of BoundingBox objects representing each object in the ground truth
        Returns:
            mAP: a number representing the mAP over all classes for a single image.
        """        
        if len(pred) == 0: return 0

        allBoundingBoxes = BoundingBoxes()
        evaluator = Evaluator()

        # loop through gt:
        for _gt in gt:
            assert type(_gt) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_gt)))
            allBoundingBoxes.addBoundingBox(_gt)

        for _pred in pred:
            assert type(_pred) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_pred)))
            allBoundingBoxes.addBoundingBox(_pred)

        #for box in allBoundingBoxes:
        #    print(box.getAbsoluteBoundingBox(format=BBFormat.XYWH), box.getBBType()) 
        if evalType == 'voc':
            metrics = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
            print(metrics)
        elif evalType == 'coco':
            assert False
        else: assert False, "evalType %s not supported"%(evalType) 
        return metrics[0]['AP']

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

class YOLOv3_Ultralytics(Model):
    def __init__(self, *network_config) -> None:
        super().__init__(*network_config)
        _yaml, self.weight = network_config
        self.isCOCO91 = False
        with open(_yaml, 'r') as stream:
            self.YAML = yaml.safe_load(stream)

        self.img_size = (416,416)
        if torch.cuda.is_available():
            self.device = select_device('0')
        else:
            self.device = select_device('cpu')
        self.conf_thres = 0.5
        self.iou_thres = 0.6
        self.max_det = 1000
        self.img_size = 416
        
        self.hide_conf = True
        self.hide_labels = False
        self.colors = Colors()  # create instance for 'from utils.plots import colors'

        # Stuff for COCO
        self.predictions = []

    def initialize(self, *kwargs):
        self.model = DetectMultiBackend(self.weight, device=self.device, dnn=False)
        self.names = self.model.names

    def run(self, input):
        with torch.no_grad():
            imageShape = input.shape
            gn = torch.tensor(imageShape)[[1, 0, 1, 0]]  # normalization gain whwh
            im = letterbox(input, self.img_size, stride=self.model.stride, auto=False)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(self.device)
            im = torch.unsqueeze(im, axis=0)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            pred = self.model(im, augment=False, visualize=False)
            pred = pred.cpu()
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False)[0]
            ###
            #print(pred)
			###
            if pred.shape[0] > 0:
                # Rescale boxes from img_size to im0 size
                pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], imageShape).round()
                #self.predictions = pred
                print(pred)
                return pred
            else:
                return []

    def draw(self, preds, im0, class_filter=None):
        labels = {"all":[255,255,255]}
        if len(preds) > 0:
            names = self.names
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0  # for save_crop
            annotator = Annotator(im0, line_width=2)
            for *xyxy, conf, cls in reversed(preds):
                c = int(cls)  # integer class
                label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                _color = self.colors(c, True)
                if not label in labels:
                    _c = list(_color)
                    labels[label] = [_c[2], _c[1], _c[0]]
                if class_filter:
                    if class_filter == label:
                        annotator.box_label(xyxy, label, color=_color)
                else:
                    annotator.box_label(xyxy, label, color=_color)
                # Stream results
            im0 = annotator.result()
        #cv2.imshow('test', im0)
        #cv2.imwrite('test.png', im0)
        #cv2.waitKey(-1)
        return {"dst": im0,
                "listOfNames":labels}

    def deinitialize(self):
        del self.model

    def draw_single_class(self, preds, im0, selected_class):
        res = self.draw(preds, im0, class_filter=selected_class)
        return {"overlay": res["dst"]}

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

    def testCOCO(self, pred):
        
        pass
      
    def report_accuracy(self, pred:list, gt:list, evalType='voc'):
        """Function takes in prediction boxes and ground truth boxes and
        returns the mean average precision (mAP) @ IOU 0.5 under VOC2007 criteria (default).
        Args:
            pred (list): A list of BoundingBox objects representing each detection from method
            gt (list): A list of BoundingBox objects representing each object in the ground truth
        Returns:
            mAP: a number representing the mAP over all classes for a single image.
        """        
        if len(pred) == 0: return 0

        allBoundingBoxes = BoundingBoxes()
        evaluator = Evaluator()

        # loop through gt:
        for _gt in gt:
            assert type(_gt) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_gt)))
            allBoundingBoxes.addBoundingBox(_gt)

        for _pred in pred:
            assert type(_pred) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_pred)))
            allBoundingBoxes.addBoundingBox(_pred)
            print("prediction", _pred)

        #for box in allBoundingBoxes:
        #    print(box.getAbsoluteBoundingBox(format=BBFormat.XYWH), box.getBBType()) 
        if evalType == 'voc':
            metrics = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
            print('Printing metrics for YOLOv3_Ultralytics', metrics)

        elif evalType == 'coco':
            assert False
        else: assert False, "evalType %s not supported"%(evalType) 
        return metrics[0]['AP']

class YOLOX(Model):
    def __init__(self, *network_config) -> None:   
        
        exp_file, name, self.weight, self.classes = network_config
        self.complexOutput = False
        self.isCOCO91 = False
        self.exp = Exp()
        self.test_size = self.exp.test_size
        self.preproc = ValTransform(legacy=False)
        self.num_classes = self.exp.num_classes
        #self.confthre = self.exp.test_conf
        self.conf_thres = 0.5
        self.nmsthre = self.exp.nmsthre
        if torch.cuda.is_available():
            self.device = "gpu"
        else:
            self.device = "cpu"

    def run(self, img):
        
        imgShape = img.shape
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])

        with torch.no_grad():
            img, _ = self.preproc(img, None, self.test_size)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.float()
            if self.device == "gpu":
                img = img.cuda()
                self.model.cuda()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.conf_thres,
                self.nmsthre, class_agnostic=True
            )

            outputs = outputs[0]

            if outputs is None:
                return torch.Tensor([])

            outputs = outputs.cpu()
            bboxes = outputs[:, 0:4]
            # preprocessing: resize
            bboxes /= ratio
            #bboxes[:, [0,2]] *= imgShape[1]
            #bboxes[:, [1,3]] *= imgShape[0]
            cls = outputs[:, 6]
            cls = torch.unsqueeze(cls, axis=1)
            scores = outputs[:, 4] * outputs[:, 5]
            scores = torch.unsqueeze(scores, axis=1)
            outputs = torch.cat((bboxes, scores, cls),axis=-1)
            outputs = outputs.cpu()
            return outputs

    def initialize(self):
        self.model = self.exp.get_model()
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(self.weight, map_location=torch.device('cpu'))["model"])
        else:
            self.model.load_state_dict(torch.load(self.weight)["model"])
        self.model.eval()
        
    def deinitialize(self):
        del self.model

    def draw(self,  preds, im0, class_filter=None):
        

        labels = {"all":[255,255,255]}

        for i in range(preds.shape[0]):
            bboxes = preds[i,:4].int().tolist()
            cls_id = preds[i,5].int()
            score = preds[i,4]
            label = self.classes[cls_id]

            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(label, score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            if class_filter:
                if class_filter != label:
                    continue

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            im0 = cv2.rectangle(im0, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), color, thickness=2)
            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            im0 = cv2.rectangle(
                im0,
                (bboxes[0], bboxes[1] + 1),
                (bboxes[0] + txt_size[0] + 1, bboxes[1] + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            im0 = cv2.putText(im0, text, (bboxes[0], bboxes[1] + txt_size[1]), font, 0.4, txt_color, thickness=1)
            if not label in labels:
                labels[label] = [color[2], color[1], color[0]]

        return {"dst":im0, "listOfNames":labels}

    def draw_single_class(self, preds, im0, selected_class):
        res = self.draw(preds, im0, class_filter=selected_class)
        return {"overlay": res["dst"]}

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

    def report_accuracy(self, pred:list, gt:list, evalType='voc'):
        """Function takes in prediction boxes and ground truth boxes and
        returns the mean average precision (mAP) @ IOU 0.5 under VOC2007 criteria (default).
        Args:
            pred (list): A list of BoundingBox objects representing each detection from method
            gt (list): A list of BoundingBox objects representing each object in the ground truth
        Returns:
            mAP: a number representing the mAP over all classes for a single image.
        """        
        if len(pred) == 0: return 0

        allBoundingBoxes = BoundingBoxes()
        evaluator = Evaluator()

        # loop through gt:
        for _gt in gt:
            assert type(_gt) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_gt)))
            allBoundingBoxes.addBoundingBox(_gt)

        for _pred in pred:
            assert type(_pred) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_pred)))
            allBoundingBoxes.addBoundingBox(_pred)

        #for box in allBoundingBoxes:
        #    print(box.getAbsoluteBoundingBox(format=BBFormat.XYWH), box.getBBType()) 
        if evalType == 'voc':
            metrics = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
            print(metrics)
        elif evalType == 'coco':
            assert False
        else: assert False, "evalType %s not supported"%(evalType) 
        return metrics[0]['AP']

class CompressiveAE(Model):

    def __init__(self, *network_config) -> None:
        super().__init__(*network_config)
        self.checkpoint = network_config[0]
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.ns = SimpleNamespace(device=device, resize=False)
        self.initialize()

    def run(self, image, patch_size, size=None):
        with torch.no_grad():
            # Run the detect.py script
            image_shape = image.shape
            img, patches, pad_img, pad, pwh = cae_preprocess_single(image, patch_size)

            if self.ns.device == "cuda":
                patches = patches.cuda()

            #out = T.zeros(6, 10, 3, 128, 128)
            ps = patches.shape
            out = torch.zeros(ps[1], ps[2], ps[0], ps[3], ps[4])

            for i in range(pad[1]):
                for j in range(pad[0]):
                    x = patches[:, i, j, :, :]
                    x = torch.unsqueeze(x, axis=0)
                    if self.ns.resize:
                        x = torch.nn.functional.interpolate(x, size=(128,128), mode='bilinear')
                    if self.ns.device == "cuda":
                        x.cuda()
                    y = self.model(x)
                    if self.ns.resize:
                        y = torch.nn.functional.interpolate(y, size=(patch_size,patch_size), mode='bilinear')
                    y = y[0]
                    out[i, j] = y.data
            
            # save output
            out = np.transpose(out, (0, 3, 1, 4, 2))
            out = np.reshape(out, (pad_img[0], pad_img[1], 3))
            out *= 255.0
            out = out.clamp(0,255.0)
            out = out.cpu().numpy()
            out = out.astype(np.uint8)
            #out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out = out[pwh[0][1]:image.shape[0]+pwh[0][1], pwh[0][0]:image.shape[1]+pwh[0][0],:]
        return out
    
    def initialize(self):
        # Intialize the configuraiton file and img
        self.model = CAE(self.ns, check_size=False)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=torch.device(self.ns.device)))
        self.model.eval()
        if self.ns.device == "cuda":
            self.model.cuda()

    def deinitialize(self):
        pass
    
    def draw(self):
        return -1

    def draw_single_class(self):
        return -1

    def outputFormat(self):
        return -1
    
    def report_accuracy(self):
        return -1

class DETR(Model):
    def __init__(self, *network_config) -> None:
        #sys.path.append( os.path.join(os.getcwd(),'src/detr'))
        import src.detr.datasets.transforms as T
        
        super().__init__(*network_config)
        file_names = network_config[0]
        self.classes = self.read_name_file(file_names)
        self.isCOCO91 = True
        self.device = 'cpu'
        if torch.cuda.is_available(): self.device = 'cuda'

        self.args = SimpleNamespace(
            dataset_file='coco',
            device=self.device,
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            num_queries=100,
            pre_norm=False,
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=2048,
            position_embedding="sine",
            dilation=False,
            backbone='resnet50',
            frozen_weights=False,
            clip_max_norm=0.1,
            lr_drop=200,
            epochs=300,
            weight_decay=1e-4,
            batch_size=1,
            lr_backbone=1e-5,
            lr=1e-4,
            masks=False,
            aux_loss=False,
            set_cost_class=1,
            set_cost_bbox=5,
            set_cost_giou=2,
            mask_loss_coef=1,
            dice_loss_coef=1,
            bbox_loss_coef=5,
            giou_loss_coef=2,
            eos_coef=0.1,
        )
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.complexOutput = False
        self.conf_thres = 0.5

    def read_name_file(self, filepath):
        with open(filepath, 'r') as f:
            contents = list(map(str.strip, f.readlines()))
        return contents

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        scale_factor = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        if self.device == 'cuda':
            scale_factor = scale_factor.cuda()
        b = b * scale_factor
        return b

    def run(self, img):
        
        import time
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        img, _ = self.transform(im, {})
        img = img.unsqueeze(0)

        img_w, img_h = im.size
        target_tensor = torch.Tensor([[img_w, img_h]])
        if self.device == 'cuda': target_tensor = target_tensor.cuda()
        # propagate through the model
        with torch.no_grad():

            if self.device == 'gpu':
                img = img.cuda()

            outputs = self.model(img)
            #results = self.postprocess['bbox'](outputs, target_tensor)
            #img_w, img_h = im.size
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > self.conf_thres
            # convert boxes from [0; 1] to image scales
            bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
            probs = probas[keep,:]
            clses = torch.unsqueeze(torch.argmax(probs, axis=1), axis=1)
            probs = probas[keep, torch.argmax(probs, axis=1)]
            probs = torch.unsqueeze(probs, axis=1)
            preds = torch.cat((bboxes_scaled, probs, clses), axis=1)
            preds = preds.cpu()
        return preds

    def initialize(self):
        from src.detr.models.detr import build

        if self.device == 'cpu':
            model_weights = torch.load("./src/detr/detr-r50-e632da11.pth", map_location=torch.device('cpu'))
        else:
            model_weights = torch.load("./src/detr/detr-r50-e632da11.pth", map_location=torch.device('cuda'))

        tuple_model = build(self.args)
        self.model = tuple_model[0]
        self.model.load_state_dict(model_weights['model'])   
        self.model.eval()
        self.postprocess = tuple_model[-1]
        print(self.postprocess)

    def deinitialize(self):
        del self.model
        return 0

    def draw(self, preds, im0, class_filter=None):
        
        labels = {"all":[255,255,255]}
        for i in range(preds.shape[0]):
            bboxes = preds[i,:4].int().tolist()
            #bboxes = preds[i,:4].int().tolist()
            cls_id = preds[i,5].int()
            score = preds[i,4]
            label = self.classes[cls_id]
            
            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(label, score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            if class_filter:
                if class_filter != label:
                    continue

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            im0 = cv2.rectangle(im0, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), color, thickness=2)
            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            im0 = cv2.rectangle(
                im0,
                (bboxes[0], bboxes[1] + 1),
                (bboxes[0] + txt_size[0] + 1, bboxes[1] + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            im0 = cv2.putText(im0, text, (bboxes[0], bboxes[1] + txt_size[1]), font, 0.4, txt_color, thickness=1)
            if not label in labels:
                labels[label] = [color[2], color[1], color[0]]

        return {"dst":im0, "listOfNames":labels}

    def draw_single_class(self, preds, im0, selected_class):
        res = self.draw(preds, im0, class_filter=selected_class)
        return {"overlay": res["dst"]}

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

    def report_accuracy(self, pred, gt, evalType='voc'):
        """Function takes in prediction boxes and ground truth boxes and
        returns the mean average precision (mAP) @ IOU 0.5 under VOC2007 criteria (default).
        Args:
            pred (list): A list of BoundingBox objects representing each detection from method
            gt (list): A list of BoundingBox objects representing each object in the ground truth
        Returns:
            mAP: a number representing the mAP over all classes for a single image.
        """        
        if len(pred) == 0: return 0

        allBoundingBoxes = BoundingBoxes()
        evaluator = Evaluator()

        # loop through gt:
        for _gt in gt:
            assert type(_gt) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_gt)))
            allBoundingBoxes.addBoundingBox(_gt)

        for _pred in pred:
            assert type(_pred) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_pred)))
            allBoundingBoxes.addBoundingBox(_pred)

        #for box in allBoundingBoxes:
        #    print(box.getAbsoluteBoundingBox(format=BBFormat.XYWH), box.getBBType()) 
        if evalType == 'voc':
            metrics = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
            print(metrics)
        elif evalType == 'coco':
            assert False
        else: assert False, "evalType %s not supported"%(evalType) 
        return metrics[0]['AP']

class YOLO_NAS(Model):
	# Based off of YoloNAS quickstart, will need to be changed for long-term custom training
    def __init__(self, *network_config) -> None:
        super().__init__(*network_config)
        self.weight, _yaml = network_config
        self.isCOCO91 = False
        with open(_yaml, 'r') as stream:
            self.YAML = yaml.safe_load(stream)
        self.img_size = (416, 416)
        self.conf_thres = 0.5
        self.iou_thres = 0.6
        self.max_det = 1000
        self.img_size = 416
		
        self.hide_conf = True
        self.hide_labels = False
        self.colors = Colors()
    
    def run(self, input):
        """
        pred = self.net.predict(input)
        return pred
        """
		
        self.pred = self.net.predict(input, conf=0.5)
        self.pred_objects = list(self.pred._images_prediction_lst)[0]
		
        print("==================================")
        #print(pred_objects.prediction.bboxes_xyxy[0])
        #print(pred_objects.prediction.bboxes_xyxy[0][0]) # x
        #print(pred_objects.prediction.bboxes_xyxy[0][1]) # y
        #print(pred_objects.prediction.bboxes_xyxy[0][2] - pred_objects.prediction.bboxes_xyxy[0][0]) # w
        #print(pred_objects.prediction.bboxes_xyxy[0][3] - pred_objects.prediction.bboxes_xyxy[0][0]) # h
        #print(pred_objects.prediction.confidence[0]) # conf
        #print(pred_objects.prediction.labels[0].astype(int)) # label
		
        num_dets = len(self.pred_objects.prediction.confidence)
        detections = []
        det = []
        for d in range(num_dets):
            #for i in range(6):
                #print("{} | {} | {} | {} | {} | {}\n".format(pred_objects.prediction.bboxes_xyxy[i][0], pred_objects.prediction.bboxes_xyxy[i][1], pred_objects.prediction.bboxes_xyxy[i][2], pred_objects.prediction.bboxes_xyxy[i][3], pred_objects.prediction.confidence[i], pred_objects.prediction.labels[i].astype(int)))
            det.append(self.pred_objects.prediction.bboxes_xyxy[d][0])
            det.append(self.pred_objects.prediction.bboxes_xyxy[d][1])
            det.append(self.pred_objects.prediction.bboxes_xyxy[d][2])
            det.append(self.pred_objects.prediction.bboxes_xyxy[d][3])
            det.append(self.pred_objects.prediction.confidence[d])
            det.append(self.pred_objects.prediction.labels[d].astype(int))
            detections.append(det)
            det = []
        #print(detections)
        
        #print(type(pred)) #ImagesDetectionPrediction
        #print(type(pred_objects)) #ImageDetectionPrediction
        
		#return pred_objects # return ImageDetectionPrediction converted to list(?)
        return detections
		
        """
        FIELDS:
            - image
            - prediction
                - bboxes_xyxy
                - confidence
                - labels
        """
		
        # self.bboxes = pred.prediction.bboxes_xyxy
		# self.confidence = pred.prediciton.confidence
        # self.labels = pred.prediction.labels
        # self.int_labels = self.labels.astype(int)
        # self.class_names = pred.class_names
        # self.pred_classes = [self.class_names[i] for i in self.int_labels]

    def initialize(self, *kwargs):
        self.net = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")
        return 0

    def deinitialize(self):
        return self.net

    def draw(self, pred, img, class_filter=None):
        labels = {"all": [255,255,255]}
	
        new_img = np.copy(img)
		
        self.bboxes = self.pred_objects.prediction.bboxes_xyxy
        self.confidence = self.pred_objects.prediction.confidence
        self.labels = self.pred_objects.prediction.labels
        self.int_labels = self.labels.astype(int)
        self.class_names = self.pred_objects.class_names
        self.pred_classes = [self.class_names[i] for i in self.int_labels]
		
        print(len(self.confidence))
        if len(self.confidence) == 0:
            print(0)
        else:
            print(sum(self.confidence) / len(self.confidence))
        """
        annotator = Annotator(new_img, line_width=2)
        for xyxy, conf, cls in zip(self.bboxes, self.confidence, self.labels):
            c = cls.astype(int).item()
            label = None if self.hide_labels else (self.class_names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
            _color = self.colors(c, True)
            if not label in labels:
                _c = list(_color)
                labels[label] = [_c[2], _c[1], _c[0]]
            if class_filter:
                if class_filter == label:
                    annotator.box_label(xyxy, label, color=_color)
            else:
                annotator.box_label(xyxy, label, color=_color)
        new_img = annotator.result()
        """
        #print(pred)
        if len(pred) > 0:
            annotator = Annotator(new_img, line_width=2)
            for *xyxy, conf, cls in reversed(pred):
                c = int(cls)
                label = None if self.hide_labels else (self.class_names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                _color = self.colors(c, True)
                if not label in labels:
                    _c = list(_color)
                    labels[label] = [_c[2], _c[1], _c[0]]
                if class_filter:
                    if class_filter == label:
                        annotator.box_label(xyxy, label, color=_color)
                else:
                    annotator.box_label(xyxy, label, color=_color)
                new_img = annotator.result()
		
        return {"dst": new_img,
                "listOfNames": labels}

    def draw_single_class(self, pred, img, selected_class):
        res = self.draw(pred, img, class_filter=selected_class)
        return {"overlay": res["dst"]}

    def report_accuracy(self, pred, gt, evalType='voc'):
        if len(pred) == 0:
            return 0
			
        allBoundingBoxes = BoundingBoxes()
        evaluator = Evaluator()
		
        for _gt in gt:
            assert type(_gt) == BoundingBox, "_gt is not BoundingBox type. Instead is %s" % (str(type(_gt)))
            allBoundingBoxes.addBoundingBox(_gt)
			
        for _pred in pred:
            assert type(_pred) == BoundingBox, "_pred is not BoundingBox type. Instead is %s" % (str(type(_gt)))
            allBoundingBoxes.addBoundingBox(_pred)
			
        if evalType == 'voc':
            metrics = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
            print('Printing metrics for YOLO-NAS', metrics)
			
        elif evalType == 'coco':
            assert False
        else:
            assert False, "evalType %s is not supported" % (evalType)
			
        return metrics[0]['AP']
	
    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

    def __call__(self):
        pred = self.run()
        return pred

_registry = {
    'Face Detection (YOLOv3)': YOLOv3(
         os.path.join(currPath, 'obj_detector/cfg', 'face.names'),
         os.path.join(currPath, 'obj_detector/cfg', 'yolov3-face.cfg'),
         os.path.join(currPath,'obj_detector/weights','yolov3-face_last.weights')
    ),
    'Face Detection (MTCNN)': MTCNN(
        os.path.join(currPath, 'mtcnn/weights', 'pnet.npy'),
        os.path.join(currPath, 'mtcnn/weights', 'rnet.npy'),
        os.path.join(currPath,'mtcnn/weights','onet.npy')
    ),
    # 'Semantic Segmentation': Segmentation(
    #     str(Path(__file__).parent.absolute()) + "/mit_semseg/config/ade20k-hrnetv2.yaml",
    #     scipy.io.loadmat(str(Path(__file__).parent.absolute()) + '/data/color150.mat')['colors']
    # ),
    'Object Detection (YOLOv3)': YOLOv3(
        os.path.join(currPath, 'obj_detector', 'cfg', 'coco.names'),
        os.path.join(currPath, 'obj_detector', 'cfg', 'yolov3.cfg'),
        os.path.join(currPath,'obj_detector', 'weights', 'yolov3.weights')
    ),
    # 'Object Detection (EfficientDetV2)': EfficientDetV2(
    #     os.path.join(currPath, 'detr', 'datasets', 'coco.names'),
    #     'efficientdetv2_dt'
    # ),
    'Object Detection (DETR)': DETR(
        os.path.join(currPath, 'detr', 'datasets', 'coco.names'),
        #os.path.join(currPath, 'detr', 'weights', 'detr.weights')
    ),
    'Object Detection (YOLOv4)': YOLOv4(
        os.path.join(currPath, 'yolov4', 'data', 'coco.names'),
        os.path.join(currPath, 'yolov4', 'cfg', 'yolov4.cfg'),
        os.path.join(currPath,'yolov4', 'weights', 'yolov4.weights')
    ),
    'Object Detection (YOLOv3-Ultra)': YOLOv3_Ultralytics(
        os.path.join(currPath, 'yolov3', 'models', 'yolov3.yaml'),
        os.path.join(currPath, 'yolov3', 'yolov3.pt')
    ),
    'Object Detection (YOLOX)': YOLOX(
        os.path.join(currPath, "yolox/exps/default/yolox_m.py"),
        "yolo-m",
        os.path.join(currPath, "yolox/weights/yolox_m.pth"),
        COCO_CLASSES
    ),
    'Object Detection (YOLO_NAS)': YOLO_NAS(
        os.path.join(currPath, 'yolonas', 'yolo_nas_s.pt'),
        os.path.join(currPath, 'yolonas', 'yolo_nas_s_arch_params.yaml')
    #   os.path.join(currPath, "HERE")
    )
}


if __name__ == '__main__':
    import cv2
    model = _registry['Object Detection (YOLOv4)']
    model.initialize()
    img = cv2.imread('imgs/default_imgs/original.png')
    assert type(img) != None # image path wrong
    preds = model.run(img)
    print(preds) # right format if last is 0<x<1 and first 4 are large numbers