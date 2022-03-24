import imp
from turtle import left, up
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
#         if item.senIDs[i] != 0:
#             if item.up_id[i] != 0 and item.senIDs[item.up_id[i]] != 0:
#                 up_box = item.up_box[i]
#                 uup_box = get_box(up_box, item.page_size)
#                 cv.rectangle(img, (uup_box[0],uup_box[1]), (uup_box[2], uup_box[3]),
#                     (100,100,100),thickness=3)
#                 cv.putText(img, 'left', (uup_box[0], uup_box[1]),
#                     cv.FONT_HERSHEY_COMPLEX, 0.4, color=(100,100,100))
#             if item.down_id[i] != 0 and item.senIDs[item.down_id[i]] != 0:
#                 down_box = item.down_box[i]
#                 ddown_box = get_box(down_box, item.page_size)
#                 cv.rectangle(img, (ddown_box[0], ddown_box[1]), (ddown_box[2], ddown_box[3]),
#                     (156,120,40),thickness=3)
#                 cv.putText(img, 'left', (ddown_box[0], ddown_box[1]), 
#                     cv.FONT_HERSHEY_COMPLEX, 0.4, color=(156,120,40))
#             if item.left_id[i] != 0 and item.senIDs[item.left_id[i]] != 0:
#                 left_box = item.left_box[i]
#                 lleft_box = get_box(left_box, item.page_size)
#                 cv.rectangle(img, (lleft_box[0], lleft_box[1]), (lleft_box[2], lleft_box[3]),
#                     (10,45,70), thickness=3)
#                 cv.putText(img, 'left', (lleft_box[0], lleft_box[1]), 
#                     cv.FONT_HERSHEY_COMPLEX, 0.4, color=(10,45,70))
#             if item.right_id[i] != 0 and item.senIDs[item.right_id[i]] != 0:
#                 right_box = item.right_box[i]
#                 rright_box = get_box(right_box, item.page_size)
#                 cv.rectangle(img, (rright_box[0], rright_box[1]), (rright_box[2], rright_box[3]),
#                     (201,10,170),thickness=3)
#                 cv.putText(img, 'right', (rright_box[0], rright_box[1]), 
#                     cv.FONT_HERSHEY_COMPLEX, 0.4, color=(201,10,170))
#             box = item.row_lines[i]
#             bbox = get_box(box, item.page_size)

#             cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
#                     (20,130,110),thickness=3)
#             cv.putText(img, 'center', (bbox[0], bbox[1]),
#                 cv.FONT_HERSHEY_COMPLEX, 0.4, color=(20,130,110))
#             cv.imshow('1', img)
#             cv.waitKey(0)


for i, item in enumerate(a):
    print(i, len(a))
    a[i].left_min = []
    a[i].left_id = []
    a[i].left_box = []

    a[i].right_min = []
    a[i].right_id = []
    a[i].right_box = []

    a[i].up_min = []
    a[i].up_id = []
    a[i].up_box = []

    a[i].down_min = []
    a[i].down_id = []
    a[i].down_box = []

    for j, curbox in enumerate(item.row_lines):
        if item.senIDs[j] == 0:
            a[i].left_min.append(0)
            a[i].left_id.append(0)
            a[i].left_box.append([0,0,0,0])
            a[i].right_min.append(0)
            a[i].right_id.append(0)
            a[i].right_box.append([0,0,0,0])
            a[i].up_min.append(0)
            a[i].up_id.append(0)
            a[i].up_box.append([0,0,0,0])
            a[i].down_min.append(0)
            a[i].down_id.append(0)
            a[i].down_box.append([0,0,0,0])
        else:
            curbox_height = curbox[3] - curbox[0]
            curbox_mid = ((curbox[0] + curbox[2])/2, (curbox[1] + curbox[3])/2)
            left_min, left_id, left_box = 10086, 0, [0,0,0,0]
            for k, box in enumerate(item.row_lines):
                if item.senIDs[k] == 0:
                    continue
                box_mid = ((box[0] + box[2])/2, (box[1] + box[3])/2)
                if box[2] < curbox[0] and abs(box_mid[1] - curbox_mid[1]) < curbox_height:
                    if curbox[0] - box[2] < left_min:
                        left_min = curbox[0] - box[2]
                        left_id = item.senIDs[k]
                        left_box = box
            if left_min == 10086: left_min = 0
            a[i].left_min.append(left_min)
            a[i].left_id.append(left_id)
            a[i].left_box.append(left_box)

            right_min, right_id, right_box = 10086, 0, [0,0,0,0]
            for k, box in enumerate(item.row_lines):
                if item.senIDs[k] == 0:
                    continue
                box_mid = ((box[0] + box[2])/2, (box[1] + box[3])/2)
                if box[0] > curbox[2] and abs(box_mid[1] - curbox_mid[1]) < curbox_height:
                    if box[0] - curbox[2] < right_min:
                        right_min = box[0] - curbox[2]
                        right_id = item.senIDs[k]
                        right_box = box
            if right_min == 10086: right_min = 0
            a[i].right_min.append(right_min)
            a[i].right_id.append(right_id)
            a[i].right_box.append(right_box)

            up_min, up_id, up_box = 10086, 0, [0,0,0,0]
            for k, box in enumerate(item.row_lines):
                if item.senIDs[k] == 0:
                    continue
                box_mid = ((box[0] + box[2])/2, (box[1] + box[3])/2)
                if box[3] < curbox[1] and overlap(curbox, box):
                    if curbox[1] - box[3] < up_min:
                        up_min = curbox[1] - box[3]
                        up_id = item.senIDs[k]
                        up_box = box
            if up_min == 10086: up_min = 0
            a[i].up_min.append(up_min)
            a[i].up_id.append(up_id)
            a[i].up_box.append(up_box)

            down_min, down_id, down_box = 10086, 0, [0,0,0,0]
            for k, box in enumerate(item.row_lines):
                if item.senIDs[k] == 0:
                    continue
                box_mid = ((box[0] + box[2])/2, (box[1] + box[3])/2)
                if box[1] > curbox[3] and overlap(curbox, box):
                    if box[1] - curbox[3] < down_min:
                        down_min = box[1] - curbox[3]
                        down_id = item.senIDs[k]
                        down_box = box
            if down_min == 10086: down_min = 0
            a[i].down_min.append(down_min)
            a[i].down_id.append(down_id)
            a[i].down_box.append(down_box)



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

torch.save(a, "/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_train_layoutlm-base-uncased_512_tok_img_rows_udlr")