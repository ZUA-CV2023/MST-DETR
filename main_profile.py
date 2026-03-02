import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import RTDETR

if __name__ == '__main__':
    # root = 'D:/CODE/RTDETR-main/'
    # choose your yaml file
    model = RTDETR('ultralytics/cfg/visdrone_use/R18-AMSE_1280_V3.yaml')
    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=[1280, 1280])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()