import sys
import os
from loguru import logger as logging
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
from util import UTIL, color, Metric
from Config import *
from Meta_Algos import MAML


class Meta_Agent:
    def __init__(
        self,
        logger: SummaryWriter = None,
        use_wandb=False,
        policy_name="PPO",
        config: config = None,
        meta_algo="MAML",
        meta_config: MAML_config = None,
        config_file=None,
    ):
        self.meta_algo_name = meta_algo
        self.policy_name = policy_name
        self.meta_config = meta_config

        self.config_file = config_file
        self.tf_logger = logger
        self.use_wandb = use_wandb

        self.get_meta_agent(config=config,
                            hyperparameters=self.read_config_file())
        self.config = self.meta_agent.config
        self.name = f"MetaRL-{self.meta_agent.name}"

    def read_config_file(self):
        HyperParameters = {}
        if self.config_file:
            config_ = UTIL.read_yaml(self.config_file)
            config_keys = list(config_.keys())
            if "Meta_Algo" in config_keys:
                Meta_Algo = config_["Meta_Algo"]["name"]
                if Meta_Algo != self.meta_algo_name:
                    logging.critical(
                        f"Meta_Algo in Bot is {self.meta_algo_name}, while in config file is {Meta_Algo}"
                    )
                HyperParameters = config_["Meta_Algo"]["HyperParameters"]
        return HyperParameters

    def get_meta_agent(self, config, hyperparameters: dict = {}):
        if self.meta_algo_name in ["MAML", "maml"]:

            if not self.meta_config:
                self.meta_config = MAML_config(**hyperparameters)
            self.meta_agent = MAML(logger=self.tf_logger,
                                   use_wandb=self.use_wandb,
                                   policy_name=self.policy_name,
                                   config=config,
                                   meta_config=self.meta_config,
                                   config_file=self.config_file)

        else:
            exit(0)

    def get_meta_config(self, meta_config):

        meta_config = meta_config if meta_config else MAML_config()

    def train(self, task_list, valid_task, eval_task, eval_freq=5):
        Train_metric = EasyDict({
            "signal": Metric.Finished,
            "Train_Episode_Rewards": [],
            "Train_Episode_Steps": [],
            "Train_Success_Rate": [],
        })
        Train_metric = self.meta_agent.Meta_train(task_list=task_list,
                                                  valid_task=valid_task,
                                                  eval_task=eval_task)

        return Train_metric

    def Evaluate(
        self,
        target_list,
        policy=None,
        step_limit=10,
        manual=False,
        interactive=False,
        determinate=True,
        verbose=True,
    ):
        return self.meta_agent.Evaluate(target_list=target_list,
                                        policy=policy,
                                        interactive=interactive,
                                        determinate=determinate,
                                        verbose=verbose)

    def save(self, path):
        self.meta_agent.save(path=path)

    def load(self, path):
        self.meta_agent.load(path=path)
