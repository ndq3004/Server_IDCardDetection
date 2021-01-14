import matplotlib.pyplot as plt

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import yaml

class VietOcrUtils():
    def __init__(self):
        # self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config = CfgCustom.load_config_from_file('./config/vgg_transformer.yml')
        self.config['weights'] = 'model/ocr/transformerocr.pth'
        # config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
        self.config['cnn']['pretrained']=False
        self.config['device'] = 'cpu'
        self.config['predictor']['beamsearch']=False
        self.detector = Predictor(self.config)
        
    def predict(self, image):
        print('[INFO] ocr start predict...: ')
        return self.detector.predict(image)

class CfgCustom:
    @staticmethod
    def load_config_from_file(fname):
        with open('./config/base.yml', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)

        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)

        