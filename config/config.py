import yaml

class Config:
    def __init__(self):
        config_file = 'config/config.yaml'
        with open(config_file, 'r') as f:
            self.__dict__ = yaml.safe_load(f)

class Config_Predict:
    def __init__(self, model_name):
        config_file = f'trained_model/{model_name.lower()}_model/config.yaml'
        with open(config_file, 'r') as f:
            self.__dict__ = yaml.safe_load(f)