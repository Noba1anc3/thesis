import imp
from re import I
from turtle import left, up
import numpy
import torch
import os
import copy
import sys
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision
sys.path.insert(0, '/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/')
import cv2 as cv
from PIL import Image

a = torch.load("/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_train_layoutlm-base-uncased_512_tok_img_rowIDs_knn")

folder_path = '/home/dreamaker/thesis/thesis/SG_Dataset/test_tok/image/'
def overlap(box1, box2):
    if box1[0] > box2[0] and box1[0] < box2[2]: return True
    if box1[2] > box2[0] and box1[2] < box2[2]: return True
    return False


def get_box(box, page_size): 
    new_width = page_size[0]*1200/page_size[1]
    boxx = [box[0], box[1], box[2], box[3]]
    boxx[0], boxx[2] = int(boxx[0] * page_size[0] / 1000),\
                        int(boxx[2] * page_size[0] / 1000),
    boxx[0], boxx[2] = int(boxx[0] * new_width / page_size[0]),\
                        int(boxx[2] * new_width / page_size[0]),

    boxx[1], boxx[3] = int(boxx[1] * page_size[1] / 1000),\
                        int(boxx[3] * page_size[1] / 1000),
    boxx[1], boxx[3] = int(boxx[1] * 1200 / page_size[1] ),\
                        int(boxx[3] * 1200 / page_size[1] ),
    return boxx

# for item in a:
#     img = cv.imread(os.path.join(folder_path, item.file_name))
#     img = cv.resize(img, (int(1200*img.shape[1]/img.shape[0]), 1200))
#     for i in range(512):
#         if item.senIDs[i] != 0 and item.label_ids[i] != 0 and item.label_ids[i] != -100:
#             k_boxes = item.knn[i]
#             for i, k_box in enumerate(k_boxes):
#                 kbox = get_box(item.row_lines[k_box], item.page_size)
#                 cv.rectangle(img, (kbox[0], kbox[1]), (kbox[2], kbox[3]),
#                     (140,80,20), thickness=2)
#             box = item.row_lines[i]
#             bbox = get_box(box, item.page_size)
#             cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
#                     (100,13,20), thickness=2)
#             # cv.putText(img, 'center', (bbox[0], bbox[1]),
#             #     cv.FONT_HERSHEY_COMPLEX, 0.4, color=(20,130,110))
#             cv.imshow('1', img)
#             cv.waitKey(0)
            


def cal_dist(box1, box2):
    center1 = (box1[0] + box1[2], box1[1] + box1[3])
    center2 = (box2[0] + box2[2], box2[1] + box2[3])
    dist = (center1[0] - center2[0])**2 + (center1[1] - center2[1])**2
    return dist

for i, item in enumerate(a):
    print(i, len(a))


    knn = []
    k_list = [0 for _ in range(10)]
    for u in range(512):
        knn.append(copy.copy(k_list))
    a[i].knn = knn

    for j, box in enumerate(item.row_lines):
        dists = {}
        curset = set()
        if item.senIDs[j] == 0: continue
        for k, obox in enumerate(item.row_lines):
            if item.senIDs[k] in curset: continue
            if item.senIDs[k] == 0 or item.senIDs[k] == item.senIDs[j]: continue
            dists[k] = cal_dist(box, obox)
            curset.add(item.senIDs[k])
        for l, (key, _) in enumerate(sorted(dists.items(), key=lambda item: item[1])):
            if l == 10: break
            a[i].knn[j][l] = key
        if a[i].knn[j][0] == 0: break

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

torch.save(a, "/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_train_layoutlm-base-uncased_512_tok_img_rowIDs_knn")