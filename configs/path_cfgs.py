import os

class Path:
    def __init__(self):
        self.feats_path = {
            'train': './data/feats/train2014',
            'val': './data/feats/val2014',
            'test': './data/feats/test2015'
        }
        self.annotations_path = {
            'train': './data/annotations/v2_OpenEnded_mscoco_train2014_questions.json',
            'train-anno': './data/annotations/v2_mscoco_train2014_annotations.json',
            'val': './data/annotations/v2_OpenEnded_mscoco_val2014_questions.json',
            'val-anno': './data/annotations/v2_mscoco_val2014_annotations.json',
            'vg': './data/annotations/VG_questions.json',
            'vg-anno': './data/annotations/VG_annotations.json',
            'test': './data/annotations/v2_OpenEnded_mscoco_test2015_questions.json'
        }
        self.result_path = './results/result_test'
        self.pred_path = './results/pred'
        self.log_path = './results/log'
        self.ckpts_path = './ckpts'
        self.tensorboard_path = './tensorboard_logs'

