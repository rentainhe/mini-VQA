import numpy as np
from types import MethodType
import os, random, torch

class BaseCfgs():
    def __init__(self):
        super(BaseCfgs, self).__init__()

        self.gpu = '0'
        self.seed = random.randint(0, 9999999)
        # version control
        self.version = str(self.seed)
        self.resume = False
        self.ckpt_version = self.version
        self.ckpt_epoch = 0
        self.ckpt_path = None  # if set the ckpt_path, 'ckpt_version' and 'ckpt_epoch' will not work any more

        # print loss every iteration
        self.verbose = True

        self.model = ''
        self.model_use = ''
        self.dataset = ''
        self.run_mode = ''
        self.eval_every_epoch = True
        self.test_save_pred = False
        self.train_split = 'train'
        self.use_glove = True
        self.word_embed_size = 300

        # features
        self.feat_size = {
            'frcn_feat_size': (100, 2048),
            'bbox_feat_size': (100, 5)
        }
        self.bbox_normalize = False

        # default training batch size: 64
        self.batch_size = 64

        # dataloader
        self.num_workers = 8
        self.pin_memory = True

        self.grad_accu_steps = 1
        self.loss_func = ''
        self.loss_reduction = ''
        self.lr_base = 0.0001
        self.lr_decay_rate = 0.2
        self.lr_decay_list = [10, 12]
        self.warmup_epoch = 3
        self.max_epoch = 13
        self.grad_norm_clip = -1
        self.opt = ''
        self.opt_params = {}