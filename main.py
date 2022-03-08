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

from Direction_Classify.tool.predict_system import TextSystem
from layoutlms.layoutlm.deprecated.examples.seq_labeling.inference import inference
from data.preprocess import convert, seg


sem_labels = ['O', 'INVConsignee', 'INVShipper', 'INVTotalGW', 
'INVCommodity.COO', 'INVNo', 'INVCurrency', 'INVPage', 'INVCommodity.Desc', 
'INVDate', 'INVTermType', 'INVCommodity.Total', 'INVCommodity.Qty', 
'INVTotalQty', 'INVTotal', 'INVCommodity.Price', 'INVCommodity.ItemNo', 
'INVCommodity.PartNumber', 'INVCommodity.HSCode', 'INVCommodity.Unit', 
'INVWtUnit', 'INVCommodity.GW', 'INVCommodity.BoxNumber', 'INVTotalNW', 'INVQtyUom']

sem_labels_Upper = [sem_label.upper() for sem_label in sem_labels]

def sem_colors():
    colors = []
    for _ in range(25):
        colors.append((random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)))
    return colors


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


def change_PaddleOCR():
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
        bboxes.append(item[key]["locations"])
        words.append(item[key]["value"])
    return bboxes, words


def get_OCR_result(image, filePath):
    jsnFilePath = filePath[:-3].replace("images", "json") + 'json'
    if os.path.exists(jsnFilePath): return readJson(jsnFilePath)
    text_sys = OCRTextSystem()
    dt_boxes, rec_res = text_sys(image)
    bboxes, words = getOCR(dt_boxes, rec_res)
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


def configParser():
    with open("configs.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def rectifyImage(img):
    text_sys = TextSystem(DET_MODEL_DIR='ch_ppocr_server_v2.0_det_infer', 
                            GPU=torch.cuda.is_available())
    return text_sys(img)


def get_LayoutLM_result(image, bboxes, words, file, colors):
    print('-------------------- Making Testing Dataset --------------------')
    convert(image.shape, bboxes, words, file)
    seg()
    print('-------------------- Testing Dataset Made --------------------')
    preds = inference()[0]
    print(len(bboxes), len(preds))
    for i in range(len(bboxes)):
        if not preds[i] == 'O':
            preds[i] = preds[i][2:]
            cv.putText(image, preds[i], tuple(bboxes[i][0]), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        color = colors[sem_labels_Upper.index(preds[i])]
        cv.rectangle(image, tuple(bboxes[i][0]), tuple(bboxes[i][1]), color)
    return image

def get_LayoutLMv2_base_result():
    pass

def get_LayoutLMv2_large_result():
    pass

if __name__ == "__main__":
    config = configParser()
    prepare_models()
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(__dir__, "PaddleOCR"))
    from PaddleOCR.tools.infer.predict_system import TextSystem as OCRTextSystem
    change_PaddleOCR()
    colors = sem_colors()

    for file in sorted(os.listdir(config["DocumentFolder"]["Path"])):
        if not file.find("png") >= 0 and file.find("jpg") >= 0: continue
        print('--------------------', file, '--------------------', '\n')
        filePath = os.path.join(config["DocumentFolder"]["Path"], file)
        origin_img = cv.imread(filePath)
        rectified_img = rectifyImage(origin_img)
        bboxes, words = get_OCR_result(rectified_img, filePath)
        if config["ModelType"]["Name"] == "LayoutLM":
            image = get_LayoutLM_result(rectified_img, bboxes, words, file, colors)
            cv.imwrite(os.path.join('output', file), image)
        elif config["ModelType"]["Name"] == "LayoutLMv2":
            pass

        # layoutlmv2_base
        # layoutlmv2_large