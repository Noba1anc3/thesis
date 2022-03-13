import imp
import numpy
import torch
import os
import sys
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision
sys.path.insert(0, '/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/')
import cv2 as cv
from PIL import Image
a = torch.load("/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_dev_layoutlm-base-uncased_512_tok_img_row")
b = torch.load("/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_train_layoutlm-base-uncased_512")
image = cv.cvtColor(numpy.asarray(b[0].resized_image),cv.COLOR_RGB2BGR)
cv.imshow('1', image)
cv.waitKey(0)

image = ToTensor()(image).unsqueeze(0)
model = torchvision.models.resnet101(pretrained=True)
backbone = nn.Sequential(*(list(model.children())[:-3]))
feature_maps = backbone(image)


import cv2 as cv
image = cv.imread("/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/FUNSD/training_data/images/00040534.png")

print(1)