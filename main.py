import os
import cv2 as cv
import yaml
from Direction_Classify.tools.infer.predict_system import TextSystem

def configParser():
    with open("configs.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def rectifyImage(img):
    text_sys = TextSystem(DET_MODEL_DIR='ch_ppocr_server_v2.0_det_infer', GPU=False)
    return text_sys(img)

def get_LayoutLM_result():

def get_LayoutLMv2_base_result():

def get_LayoutLMv2_large_result():

    
if __name__ == "__main__":
    config = configParser()
    
    for file in os.listdir(config["DocumentFolder"]["Path"]):
        origin_img = cv.imread(os.path.join(config["DocumentFolder"]["Path"], file))
        rectified_img = rectifyImage(origin_img)
        
        layoutlm_base
        layoutlmv2_base
        layoutlmv2_large