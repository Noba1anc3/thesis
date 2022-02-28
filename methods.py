import json
from shutil import copyfile
from turtle import width
import cv2 as cv
import numpy as np

sem_labels = ['ignore', 'INVConsignee', 'INVShipper', 'INVTotalGW', 
'INVCommodity.COO', 'INVNo', 'INVCurrency', 'INVPage', 'INVCommodity.Desc', 
'INVDate', 'INVTermType', 'INVCommodity.Total', 'INVCommodity.Qty', 
'INVTotalQty', 'INVTotal', 'INVCommodity.Price', 'INVCommodity.ItemNo', 
'INVCommodity.PartNumber', 'INVCommodity.HSCode', 'INVCommodity.Unit', 
'INVWtUnit', 'INVCommodity.GW', 'INVCommodity.BoxNumber', 'INVTotalNW', 'INVQtyUom']


def tokenize(jsn):
    new_jsn = {'items':[]}
    for j in range(len(jsn["items"])):
        item = jsn["items"][j]
        key = list(item.keys())[0]

        loc = item[key]["locations"]
        val = item[key]["value"]

        vals = val.split(" ")
        single_ch_len = (loc[1][0] - loc[0][0]) / len(val)
        
        bef_start = loc[0][0]
        for i, val in enumerate(vals):
            x1 = min(bef_start + int((len(val) + 1/2)*single_ch_len), loc[1][0])
            cur_loc = [[bef_start, loc[0][1]], [x1, loc[1][1]]]
            if len(vals) == 1: pre = 'S'
            else:
                if i == 0: pre = 'B'
                elif i < len(vals) - 1: pre = 'I'
                else: pre = 'E'
            new_jsn['items'].append({key:{"value":val,"locations":cur_loc,"pre":pre}})
            bef_start = x1 + int(single_ch_len/2)
            if bef_start >= loc[1][0]: break
    return new_jsn

def shrink(img, jsn):
    for j in range(len(jsn["items"])):
        item = jsn["items"][j]
        key = list(item.keys())[0]
        loc = item[key]["locations"]
        for k in range(loc[1][0], loc[0][0], -1):
            # col_count = 0
            shrink = False
            for l in range(loc[0][1], loc[1][1]):
                if img[l][k].tolist() != [255, 255, 255]:
                    jsn["items"][j][key]["locations"][1][0] = k
                    # col_count += 1
                    shrink = True
                    break
            # if col_count / (loc[1][1] - loc[0][1]) > 0.2:
            if shrink: break

def sem_colors():
    import random
    colors = []
    for i in range(25):
        colors.append((random.random()*255, random.random()*255,random.random()*255))
    return colors

def get_img(file_path):
    img = cv.imread(file_path)
    ratio = img.shape[0] / 800
    img = cv.resize(img, (int(800*img.shape[1]/img.shape[0]), 800))
    return img, ratio

def get_img_ori(file_path):
    return cv.imread(file_path)

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


def get_token_json(jsn):
    new_jsn = {"items":[]}
    for j in range(len(jsn["items"])):
        item = jsn["items"][j]
        key = list(item.keys())[0]

        loc = item[key]["locations"]
        val = item[key]["value"]
        vals = val.split(" ")
        single_ch_len = int((loc[1][0] - loc[0][0]) / (len(val) - len(vals) + 1))
        
        bef_start = loc[0][0]
        for i, val in enumerate(vals):
            x1 = bef_start + int((len(val) + 1/3)*single_ch_len)
            cur_loc = [[bef_start, loc[0][1]], [x1, loc[1][1]]]
            bef_start = x1 + int(single_ch_len/3)
            new_jsn['items'].append({key:{"value":val,"locations":cur_loc}})


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


def cal_lu_rd_mean(loc, wr, hr):

    x0 = loc[0][0]
    y0 = loc[0][1] 
    x1 = loc[1][0]
    y1 = loc[1][1] 
    x2 = loc[2][0]
    y2 = loc[2][1]
    x3 = loc[3][0]
    y3 = loc[3][1] 

    lu = (int(min(x0,x3)/wr), int(min(y0, y1)/hr))
    rd = (int(max(x2, x1)/wr), int(max(y2, y3)/hr))
    return lu, rd

def cal_lu_rd_ori(loc, r):

    x0 = loc[0][0]
    y0 = loc[0][1]
    x2 = loc[1][0]
    y2 = loc[1][1]

    lu = (int((x0)/r), int((y0)/r))
    rd = (int((x2)/r), int((y2)/r))
    return lu, rd



def cal_lu_rd_origin(loc):

    x0 = loc[0][0]
    y0 = loc[0][1] 
    x1 = loc[1][0]
    y1 = loc[1][1] 
    x2 = loc[2][0]
    y2 = loc[2][1]
    x3 = loc[3][0]
    y3 = loc[3][1] 

    lu = [int(min(x0, x3)), int(min(y0, y1))]
    rd = [int(max(x2, x1)), int(max(y2, y3))]
    return lu, rd


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
