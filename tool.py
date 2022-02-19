from cmath import nan
import imp
from operator import truediv
import os
import json
from shutil import copyfile
from turtle import width
import cv2 as cv
import numpy as np
from Direction_Classify.tools.infer.correct import rotate_bound

all_datas = "SG_Dataset"


train_img_path = os.path.join( all_datas, "train", "image")
train_jsn_path = os.path.join( all_datas, "train", "json")

test_img_path = os.path.join( all_datas, "test", "image")
test_jsn_path = os.path.join( all_datas, "test", "json")

sem_labels = ['ignore', 'INVConsignee', 'INVShipper', 'INVTotalGW', 
'INVCommodity.COO', 'INVNo', 'INVCurrency', 'INVPage', 'INVCommodity.Desc', 
'INVDate', 'INVTermType', 'INVCommodity.Total', 'INVCommodity.Qty', 
'INVTotalQty', 'INVTotal', 'INVCommodity.Price', 'INVCommodity.ItemNo', 
'INVCommodity.PartNumber', 'INVCommodity.HSCode', 'INVCommodity.Unit', 
'INVWtUnit', 'INVCommodity.GW', 'INVCommodity.BoxNumber', 'INVTotalNW', 'INVQtyUom']


def sem_colors():
    import random
    colors = []
    for i in range(25):
        colors.append((random.random()*255, random.random()*255,random.random()*255))
    return colors

colors = sem_colors()

def get_img(file_path):
    img = cv.imread(file_path)
    width_ratio = img.shape[1] / 600
    height_ratio = img.shape[0] / 800
    img = cv.resize(img, (600, 800))
    return img, width_ratio, height_ratio

def open_json(file_path):
    jsonfile = json.load(open(file_path))
    return jsonfile

# 基于宽高比从位置中计算两顶点坐标
def get_loc_with_ratio(loc, wr, hr):
    LU = (int((loc[0][0]+loc[3][0])/2/wr), int((loc[0][1]+loc[1][1])/2/hr))
    RD = (int((loc[2][0]+loc[1][0])/2/wr), int((loc[3][1]+loc[2][1])/2/hr))
    return LU, RD

def angle_cal(loc):
    import math
    x0 = loc[0][0]
    y0 = loc[0][1]
    x1 = loc[1][0]
    y1 = loc[1][1]
    x2 = loc[2][0]
    y2 = loc[2][1]
    x3 = loc[3][0]
    y3 = loc[3][1]

    up = down = 0

    if y0 > y1 and y3 > y2: up = True
    if y0 < y1 and y3 < y2: down = True

    theta = 0
    if up or down:
        delta_y = abs((y1 - y0 + y2 - y3)/2)
        delta_x = abs((x1 - x0 + x2 - x3)/2)
        tan_theta = delta_y / delta_x
        theta = math.atan(tan_theta)
        theta = 180*theta/math.pi
    if down: theta = -1*theta
    return theta


# 宽高比大于５
def judge_valid_box(loc):
    x0 = loc[0][0]
    y0 = loc[0][1]
    x1 = loc[1][0]
    y1 = loc[1][1]
    x2 = loc[2][0]
    y2 = loc[2][1]
    x3 = loc[3][0]
    y3 = loc[3][1]

    delta_y = abs((y0 - y1 + y3 - y2)/2)
    delta_x = abs((x1 - x0 + x2 - x3)/2)

    if delta_y != 0 and delta_x / delta_y > 5: return True
    else: return False


def cal_lu_rd(loc, wr, hr, M):

    x0 = loc[0][0]
    y0 = loc[0][1]
    x2 = loc[2][0]
    y2 = loc[2][1]

    lu = (int((x0*M[0][0] + y0*M[0][1]+M[0][2])/wr), int((x0*M[1][0] + y0*M[1][1]+M[1][2])/hr))
    rd = (int((x2*M[0][0] + y2*M[0][1]+M[0][2])/wr), int((x2*M[1][0] + y2*M[1][1]+M[1][2])/hr))
    return lu, rd


def cal_lu_rd_ori(loc, wr, hr, M):

    x0 = loc[0][0]
    y0 = loc[0][1]
    x2 = loc[2][0]
    y2 = loc[2][1]

    lu = (int((x0)/wr), int((y0)/hr))
    rd = (int((x2)/wr), int((y2)/hr))
    return lu, rd

files = []
def get_file_angle(jsn):
    angles = []
    for item in jsn["items"]:
        key = list(item.keys())[0]
        loc = item[key]["locations"]
        if not judge_valid_box(loc): continue
        angles.append(angle_cal(loc))  
    if len(angles) > 0: return np.mean(np.array(angles))
    else: return 0


def draw_box(jsn, wr, hr, M):
    for item in jsn["items"]:
        key = list(item.keys())[0]
        loc = item[key]["locations"]
        lu, rd = cal_lu_rd(loc, wr, hr, M)
        cv.rectangle(img, lu, rd, (111,231,1))
        lu, rd = cal_lu_rd_ori(loc, wr, hr, M)
        cv.rectangle(img, lu, rd, (1, 1, 1))

for i, file in enumerate(sorted(os.listdir(train_jsn_path))):
    img_path = os.path.join(train_img_path, file[:-4] + 'png')
    
    img, wr, hr = get_img(img_path)

    jsn = open_json(os.path.join(train_jsn_path, file))
    angle_mean = get_file_angle(jsn)

    if abs(angle_mean) > 0:
        print(i, angle_mean)
        img, M = rotate_bound(img, -angle_mean)
        draw_box(jsn, wr, hr, M)
        
        cv.imshow('1', img)
        cv.waitKey(0)
            # cv.rectangle(img, LU, RD, colors[sem_labels.index(key)])
            

    #     LU, RD = get_loc_with_ratio(loc, wr, hr)
    #     if not key == 'ignore':
    #         cv.rectangle(img, LU, RD, colors[sem_labels.index(key)])
    #         cv.putText(img, key, LU, cv.FONT_HERSHEY_COMPLEX, 0.4, color=colors[sem_labels.index(key)])
