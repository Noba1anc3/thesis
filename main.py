import os
import cv2 as cv
import yaml
import sys
from Direction_Classify.tool.predict_system import TextSystem

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
                                print(new_line)
                                break
                if ch:
                    with open(path, "w") as f:
                        f.writelines(content)
                    f.close()

def del_():
    for i in range(len(sys.path)-1, -1, -1):
        if sys.path[i].find("Direction_Classify") >= 0:
            del sys.path[i]

def change_syspath_folder():
    ppocr = os.path.join(__dir__, "Direction_Classify")
    os.rename(ppocr, ppocr+"s")


def get_OCR_result(image):
    text_sys = OCRTextSystem()
    dt_boxes, rec_res = text_sys(image)
    print(dt_boxes, rec_res)

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
    
    for file in os.listdir(config["DocumentFolder"]["Path"]):
        origin_img = cv.imread(os.path.join(config["DocumentFolder"]["Path"], file))
        rectified_img = rectifyImage(origin_img)
        get_OCR_result(rectified_img)
        # layoutlm_base
        # layoutlmv2_base
        # layoutlmv2_large