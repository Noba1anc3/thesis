import os
import cv2 as cv
import numpy as np
import json
import yaml
import sys
import torch
import wget
import zipfile
import tarfile
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
 

from Direction_Classify.tool.predict_system import TextSystem
from layoutlms.layoutlm.deprecated.examples.seq_labeling.inference import inference
from Direction_Classify.tool.utility import draw_ocr_box_txt
from data.preprocess import convert, seg

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
sem_labels = ['O', 'Consignee', 'Shipper', 'TotalGW', 
'C.COO', 'No', 'Currency', 'Page', 'C.Desc', 
'Date', 'TermType', 'C.Total', 'C.Qty', 
'TotalQty', 'Total', 'C.Price', 'C.ItemNo', 
'C.PartNumber', 'C.HSCode', 'C.Unit', 
'WtUnit', 'C.GW', 'C.BoxNumber', 'TotalNW', 'QtyUom']


def configParser():
    with open("configs.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

config = configParser()


def sem_colors():
    colors = []
    for _ in range(24):
        colors.append((random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)))
    colors = [(100, 100, 100)] + colors
    return colors
colors = sem_colors()


def prepare_models():
    PaddleOCR_url = 'https://github.com/PaddlePaddle/PaddleOCR/archive/refs/heads/release/2.4.zip'
    det_2_url = 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar'
    cls_2_url = 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
    rec_2_url = 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar'

    if not os.path.exists("PaddleOCR-release-2.4.zip"): 
        wget.download(PaddleOCR_url)
        with zipfile.ZipFile('PaddleOCR-release-2.4.zip') as z:
            z.extractall(".")
        os.rename("PaddleOCR-release-2.4", "PaddleOCR")

    if not os.path.exists("ch_ppocr_server_v2.0_det_infer.tar"): 
        # wget.download(det_2_url)
        shutil.copyfile('/content/drive/My Drive/ch_ppocr_server_v2.0_det_infer.tar', 
                        'ch_ppocr_server_v2.0_det_infer.tar')
    if not os.path.exists("ch_ppocr_server_v2.0_rec_infer.tar"): 
        # wget.download(rec_2_url)
        shutil.copyfile('/content/drive/My Drive/ch_ppocr_server_v2.0_rec_infer.tar', 
                        'ch_ppocr_server_v2.0_rec_infer.tar')
    if not os.path.exists("ch_ppocr_mobile_v2.0_cls_infer.tar"): 
        # wget.download(cls_2_url)
        shutil.copyfile('/content/drive/My Drive/ch_ppocr_mobile_v2.0_cls_infer.tar', 
                        'ch_ppocr_mobile_v2.0_cls_infer.tar')
    
    with tarfile.TarFile('ch_ppocr_server_v2.0_det_infer.tar') as t:
        t.extractall("./models")
        t.extractall("./Direction_Classify/inference")
    with tarfile.TarFile('ch_ppocr_server_v2.0_rec_infer.tar') as t:
        t.extractall("./models")
    with tarfile.TarFile('ch_ppocr_mobile_v2.0_cls_infer.tar') as t:
        t.extractall("./Direction_Classify/inference")

    shutil.copyfile('Direction_Classify/predict_system.py', 'PaddleOCR/tools/infer/predict_system.py')
    shutil.copyfile('Direction_Classify/rec_postprocess.py', 'PaddleOCR/ppocr/postprocess/rec_postprocess.py')

    if not os.path.exists('models/layoutlm'):
        shutil.copytree('/content/drive/My Drive/layoutlm', 'models/layoutlm')

    shutil.copyfile('/content/drive/My Drive/model.zip', 'layoutlms/layoutlmft/model.zip')
    with zipfile.ZipFile('layoutlms/layoutlmft/model.zip') as z:
        z.extractall("layoutlms/layoutlmft/")
    shutil.copyfile('/content/drive/My Drive/mode1.zip', 'layoutlms/layoutlmv2/mode1.zip')
    with zipfile.ZipFile('layoutlms/layoutlmv2/mode1.zip') as z:
        z.extractall("layoutlms/layoutlmv2/")
    
def change():
    folder = 'PaddleOCR'
    lis = ["benchmark", "configs", "deploy", "doc", "ppocr", "PPOCRLabel",
                                    "ppstructure", "StyleText", "test_tipc", "tools"]

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.find(".py") > 0 and file.find(".pyc") < 0:
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    content = f.readlines()
                f.close()
                ch = False
                for i in range(len(content)):
                    line = content[i]
                    if line.find("from") == 0 and line.find("import") > 0:
                        a = line.split("from")[1]
                        b = a.split("import")[0].strip()
                        c = a.split("import")[1].strip()
                        for l in lis:
                            if b.find(l+'.') == 0:
                                ch = True
                                new_line = "from " + "PaddleOCR."+b + " import " + c+'\n'
                                content[i] = new_line
                                break
                if ch:
                    with open(path, "w") as f:
                        f.writelines(content)
                    f.close()


def readJson(jsnPath):
    jsonfile = json.load(open(jsnPath))
    bboxes, words = [], []
    for item in jsonfile["items"]:
        key = list(item.keys())[0]
        if item[key]["locations"][2][0] < item[key]["locations"][0][0]: continue
        if item[key]["locations"][2][1] < item[key]["locations"][0][1]: continue
        if item[key]["value"] == "": continue
        bboxes.append(item[key]["locations"])
        words.append(item[key]["value"])
    return bboxes, words


def get_OCR_result(image, filePath):
    if not filePath.split(".png")[0][-1].isdigit(): 
      num_list = [1, 1, 57, 19, 13, 12]
      company_list = ["ALPHA", "AMPHENOLE", "INFINEON", "KOMAGUSAM", "SIIX", "TOSHIBA"]
      for i, company in enumerate(company_list):
        number = 0
        if filePath.find(company) > 0: 
          number = num_list[i]
          break
      filePath = filePath.split(".png")[0] + "_" + str(number) + ".png"
    if config["Model"]["Version"] == 2:
        jsnFilePath = "layoutlms/layoutlmv2/json" + filePath.split("images")[1][:-3] + 'json'
    else:
        jsnFilePath = "layoutlms/layoutlmft/json" + filePath.split("images")[1][:-3] + 'json'
    text_sys = OCRTextSystem()
    dt_boxes, rec_res = text_sys(image)
    bboxes, words = getOCR(dt_boxes, rec_res)
    if not config["Bbox"]["Cal"] and os.path.exists(jsnFilePath): return readJson(jsnFilePath)
    return bboxes, words


def getOCR(dt_boxes, rec_res):
    bboxes = []
    words = []
    
    for dt_box in dt_boxes:
        dt_box = dt_box.tolist()
        bbox = [[int(min(dt_box[0][0], dt_box[3][0])), int(min(dt_box[0][1], dt_box[1][1]))],
                [int(max(dt_box[1][0], dt_box[2][0])), int(max(dt_box[2][1], dt_box[3][1]))]]
        bboxes.append(bbox)
    
    for rec in rec_res:
        words.append(rec[0])

    assert len(bboxes) == len(words)
    return bboxes, words


def rectifyImage(img):
    text_sys = TextSystem(DET_MODEL_DIR='ch_ppocr_server_v2.0_det_infer',
                          # CLS_MODEL_DIR='ch_ppocr_mobile_v2.0_cls_infer',
                            GPU=torch.cuda.is_available(),
                            PhaseI = config["Rotate"]["PhaseI"],
                            PhaseII = config["Rotate"]["PhaseII"])
    new_img =  text_sys(img)
    
    return new_img


def drawImage_CV(image, bboxes, preds):
    for i in range(len(bboxes)):
        if not preds[i] == 'O':
            preds[i] = preds[i][2:]
            cv.putText(image, preds[i], tuple(bboxes[i][0]), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        color = colors[sem_labels_Upper.index(preds[i])]
        cv.rectangle(image, tuple(bboxes[i][0]), tuple(bboxes[i][1]), color)
    return image


def drawImage(image, bboxes, preds, sem_labels, colors):
    bboxes_four_pts = []
    for bbox in bboxes:
        if len(bbox) > 2: bbox_four_pts = np.array(bbox, dtype="float32")
        else: bbox_four_pts = np.array([[bbox[0][0], bbox[0][1]], \
                                    [bbox[1][0], bbox[0][1]], \
                                    [bbox[1][0], bbox[1][1]], \
                                    [bbox[0][0], bbox[1][1]]],
                                    dtype="float32")
        bboxes_four_pts.append(bbox_four_pts)
    pred_image = draw_ocr_box_txt(image, bboxes_four_pts, preds,
                './Direction_Classify/latin.ttf', sem_labels, colors)
    return pred_image


def organizeJson(bboxes, words, preds):
    data = []
    for i in range(min(len(bboxes), len(preds))):
        bbox = bboxes[i]
        word = words[i]
        if word == '': continue
        pred = preds[i]

        data.append({
                            "tokens": word,
                            "semantic": pred,
                            "location": bbox
                            })
    for i in range(len(preds), len(bboxes), 1):
        bbox = bboxes[i]
        word = words[i]
        if word == '': continue
        data.append({
                            "tokens": word,
                            "semantic": 'O',
                            "location": bbox
                            })
    
    data.sort(key= lambda x: x["semantic"])
                               
    return data


def deal_with_preds(preds):
    sem_labels_Upper = [sem_label.upper() for sem_label in sem_labels]
    
    for i, pred in enumerate(preds):
        pred = pred.replace("INV", "").replace("OMMODITY", "")

        if pred != 'O': 
            index = sem_labels_Upper.index(pred[2:])
            preds[i] = sem_labels[index]

    return preds

def simplify(words, preds):
    output_dict = {}
    for (word, pred) in zip(words, preds):
        if word == '': continue
        if not pred == 'O':
            if pred not in output_dict.keys():
                output_dict[pred] = word
            else:
                output_dict[pred] += (' ' + word)
  
    output = []
    for (key, value) in output_dict.items():
      output.append({"semantic": key, "value": value})

    return output

def get_LayoutLM_result(image, bboxes, words, file):
    print('-------------------- Making Testing Dataset --------------------')
    convert(image.shape, bboxes, words, file)
    seg()
    print('-------------------- Testing Dataset Made --------------------')
    preds = inference()
    
    preds = deal_with_preds(preds)

    pred_image = drawImage(image, bboxes, preds, 
                            sem_labels, colors)
    pred_json = organizeJson(bboxes, words, preds)
    simplified_output = simplify(words, preds)
    
    return pred_image, pred_json, simplified_output


def get_LayoutLMv2_base_result():
    pass

def get_LayoutLMv2_large_result():
    pass


if __name__ == "__main__":
    prepare_models()
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(__dir__, "PaddleOCR"))
    from PaddleOCR.tools.infer.predict_system import TextSystem as OCRTextSystem
    change()

    if not os.path.exists('output'): os.mkdir('output')
    if not os.path.exists('output/image'): os.mkdir('output/image')
    if not os.path.exists('output/rectify'): os.mkdir('output/rectify')
    if not os.path.exists('output/json'): os.mkdir('output/json')
    if not os.path.exists('output/simplify'): os.mkdir('output/simplify')
     
    if os.listdir(config["DocumentFolder"]["Path"]) == []: 
        print("No Data in Document Folder")
    
    for file in sorted(os.listdir(config["DocumentFolder"]["Path"])):
        if not file.find("png") >= 0 and not file.find("jpg") >= 0 \
            and not file.find("jpeg") >= 0: 
            continue
        
        print('\n\n\n\n--------------------', file, '--------------------', '\n')
        
        filePath = os.path.join(config["DocumentFolder"]["Path"], file)
        origin_img = cv.imread(filePath)
        rectified_img = rectifyImage(origin_img)
        cv.imwrite(os.path.join('output/rectify', file), rectified_img)
        
        bboxes, words = get_OCR_result(rectified_img, filePath)
        
        if config["ModelType"]["Name"] == "LayoutLM":
            img, jsn, output = get_LayoutLM_result(rectified_img, bboxes, words, file)
        elif config["ModelType"]["Name"] == "LayoutLMv2":
            if config["ModelType"]["Base"]:
                pass
            else:
                pass

        cv.imwrite(os.path.join('output/image', file), img)
        
        with open(os.path.join('output/json', file[:-3] + "json"), 'w') as f:
            json.dump(jsn, f)
          
        with open(os.path.join('output/simplify', file[:-3] + "json"), 'w') as f:
            json.dump(output, f)

