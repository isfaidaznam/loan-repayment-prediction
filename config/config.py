import yaml

class Config:
    def __init__(self, config_file='config/config.yaml'):
        with open(config_file, 'r') as f:
            self.__dict__ = yaml.safe_load(f)