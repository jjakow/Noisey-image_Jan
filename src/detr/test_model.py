import os
import sys
from types import SimpleNamespace
import torch
from models.detr import build
from PIL import Image
import requests
import torchvision.transforms as T
torch.set_grad_enabled(False);
import matplotlib.pyplot as plt

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    print("Testing")
    model_weights = torch.load("detr-r50-dc5-f0fb7ef5.pth")
    #backbone_weights = torch.load("resnet50-0676ba61.pth")

    # COCO classes
    CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
    ]

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    args = SimpleNamespace(
        dataset_file='coco',
        device='cuda:0',
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=100,
        pre_norm=False,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        position_embedding="sine",
        dilation=True,
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
        aux_loss=True,
        set_cost_class=1,
        set_cost_bbox=5,
        set_cost_giou=2,
        mask_loss_coef=1,
        dice_loss_coef=1,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        eos_coef=0.1,
    )

    tuple_model = build(args)
    '''
    print(model[0])
    print()
    print(model[1])
    print()
    print(model[2])
    '''
    model = tuple_model[0]
    model.load_state_dict(model_weights['model'])
    #model.load_state_dict(backbone_weights)

    #print([i for i in model.keys()])
    #print([i for i in weights.keys()])

    #model = torch.hub.load('facebookresearch/detr', 'detr_resnet50_dc5', pretrained=False)
    #print(model)

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    plot_results(im, probas[keep], bboxes_scaled)