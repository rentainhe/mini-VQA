from configs.base_cfgs import BaseCfgs
from configs.path_cfgs import Path
import os, torch, random
import numpy as np
from types import MethodType

class BuildCfgs(BaseCfgs, Path):
    def __init__(self):
        super(BuildCfgs, self).__init__()
        self.path_init()

    def path_init(self):
        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')

    def check_path(self):
        print('Checking dataset ........')


test = BuildCfgs()