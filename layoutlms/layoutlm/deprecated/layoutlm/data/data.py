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

a = torch.load("/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_train_layoutlm-base-uncased_512_tok_img_rowIDs")

for i, item in enumerate(a):
    senIDs = [0]
    actual_bboxes = [(i, x, item.senIDs[i]) for i, x in enumerate(item.boxes)]
    print(actual_bboxes)
    break
    # for j, row_line in enumerate(item.row_lines):
    #     if j == 0: continue
    #     if row_line in [[0,0,0,0], [1000,1000,1000,1000]]: senIDs.append(0)
    #     else:
    #         if row_line == item.row_lines[j-1]:
    #             senIDs.append(senIDs[j-1])
    #         else:
    #             senIDs.append(senIDs[j-1]+1)
    # a[i].senIDs = senIDs

# image = ToTensor()(image).unsqueeze(0)
# model = torchvision.models.resnet101(pretrained=True)
# backbone = nn.Sequential(*(list(model.children())[:-3]))
# feature_maps = backbone(image)

# torch.save(a, "/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_dev_layoutlm-base-uncased_512_tok_img_rowIDs")