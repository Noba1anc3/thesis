import imp
import os
import cv2 as cv
import numpy as np
import yaml
import sys
from Direction_Classify.tool.predict_system import TextSystem
from data.preprocess import convert, seg

# https://mirror.baidu.com/pypi/simple

def change_PaddleOCR():
    folder = 'PaddleOCR'
    lis = ["benchmark", "configs", "deploy", "doc", "ppocr", "PPOCRLabel",
                                    "ppstructure", "StyleText", "test_tipc", "tools"]

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.find(".py")>0 and file.find(".pyc") < 0:
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

def get_OCR_result(image):
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
    text_sys = TextSystem(DET_MODEL_DIR='ch_ppocr_server_v2.0_det_infer', GPU=False)
    return text_sys(img)

def get_LayoutLM_result():
    pass

def get_LayoutLMv2_base_result():
    pass

def get_LayoutLMv2_large_result():
    pass

if __name__ == "__main__":
    config = configParser()
    
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(__dir__, "PaddleOCR"))
    from PaddleOCR.tools.infer.predict_system import TextSystem as OCRTextSystem
    change_PaddleOCR()

    for file in sorted(os.listdir(config["DocumentFolder"]["Path"])):
        print(file, '\n')
        # origin_img = cv.imread(os.path.join(config["DocumentFolder"]["Path"], file))
        # rectified_img = rectifyImage(origin_img)
        # bboxes, words = get_OCR_result(rectified_img)
        # if config["ModelType"]["Name"] == "LayoutLM":
        #     convert(rectified_img.shape, bboxes, words, file)
        seg()
        break

        # layoutlm_base
        # layoutlmv2_base
        # layoutlmv2_large