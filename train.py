import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # Root  = 'D:/CODE/RTDETR-main/'
    model = RTDETR('ultralytics/cfg/Simd_use/R18-AMSE_1280_V3.yaml')

    # model = RTDETR(Root + 'ultralytics/cfg/models/rt-detr/rtdetr-r50-AggregatedAtt.yaml')
    model.load('runs/Visdrone_train/R18-AMSE_EUFRM_1280_V3/weights/best.pt') # loading pretrain weights
    model.train(data= 'dataset/VisDrone.yaml',
                cache=False,
                imgsz=1280,
                epochs=200,
                batch=4,
                workers=4,
                device='0',
                # seed = 21,
                resume='runs/Visdrone_train/R18-AMSE_1280_V3_bs45/weights/best.pt', # last.pt path
                project='runs/Visdrone_train',
                name='R18-AMSE_1280_V3_bs4',
                )