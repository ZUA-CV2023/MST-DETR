import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/exp/weights/best.pt') # select your model.pt path
    model.predict(source='assets',
                  conf=0.5,
                  project='runs/detect',
                  name='exp',
                  save=True,
                #   visualize=True # visualize model features maps
                  )