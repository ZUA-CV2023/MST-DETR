import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/Visdrone_train/R18-SAFM_down1280_V32/weights/best.pt')  # 自己训练结束后的模型权重
    model.val(data='dataset/VisDrone.yaml',
              split='test',
              project='runs/test',
              # split='val',
              # project='runs/val',
              imgsz=1280,
              batch=16,
              save_json=True,  # if you need to cal coco metrice
              name='exp',
              task = "test",
              augment = False
              )