import sys, os
from pprint import pprint
from .RL_config import *


# self.min_decay_lr = 3e-2
class MAML_config:
    def __init__(self,
                 num_iterations=50,
                 adapt_batch_size=128,
                 adapt_lr=5e-4,
                 meta_lr=5e-4,
                 adapt_steps=1,
                 meta_optimize_times=5,
                 use_lr_decay=False,
                 epslion=0.6,
                 gamma=0.99,
                 norm_adv = True,
                 pre_train=True):
        self.num_iterations = num_iterations
        self.adapt_batch_size = adapt_batch_size
        self.adapt_lr = adapt_lr
        self.meta_lr = meta_lr
        self.adapt_steps = adapt_steps
        self.meta_optimize_times = meta_optimize_times
        self.pre_train = pre_train
        self.use_lr_decay = use_lr_decay
        self.epslion=epslion
        self.gamma=gamma
        self.norm_adv = norm_adv
