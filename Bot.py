import os
import platform
import copy
import torch
import json
from pprint import pformat
import time
from loguru import logger as logging
import pandas as pd
import random
import csv
from prettytable import PrettyTable
from datetime import datetime
from util import Configure, UTIL, color, Metric, console, set_seed
from easydict import EasyDict
from agent import Agent
from agent_meta import Meta_Agent
from actions import *
from host import HOST, StateEncoder
import wandb
from pathlib import Path
import asciichartpy
from NLP_Module.Encoder import *
from rich.pretty import Pretty, pprint
from rich.panel import Panel
from rich.table import Table
from rich import box

# tensorboard --logdir runs --host localhost --port 8896
from torch.utils.tensorboard import SummaryWriter


class BOT:
    """Deep Q-Network Bot"""

    def __init__(
        self,
        mode=0,
        train_env_file=None,
        eval_env_file=None,
        cl_method="",
        meta_algo="",
        cl_train_num=0,
        policy: str = "PPO",
        config_file="",
        config=None,
        cl_config=None,
        meta_config=None,
        save_model=False,
        seed=0,
        note="",
        load_agent="",
        use_wandb=False,
        use_tensorboard=False,
        **kwargs,
    ):
        set_seed(seed)
        self.mode = int(mode)
        self.mode_name = UTIL.mode_name(mode)
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.wandb_run = None
        self.note = note
        self.seed = seed
        self.host_name = f"{platform.platform()}-{platform.node()}"
        self.train_env_file = Path(train_env_file) if train_env_file else None
        self.eval_env_file = (Path(eval_env_file)
                              if eval_env_file else self.train_env_file)
        self.policy_name = policy
        self.cl_method = cl_method
        self.meta_algo = meta_algo
        self.time_flag = datetime.now().strftime("%b%d_%H-%M-%S")
        self.cl_train_num = cl_train_num
        if self.use_tensorboard:
            self.tensorboard_logger = SummaryWriter()
        else:
            self.tensorboard_logger = None

        self.agent = self.get_agent(
            config=config,
            cl_config=cl_config,
            meta_config=meta_config,
            config_file=config_file,
        )
        self.train_env_name = (
            self.train_env_file.parent.name + "-" + self.train_env_file.stem
        ) if self.train_env_file else f"Unknown_env-{UTIL.current_time}"
        self.title = f"{self.agent.name}-{self.agent.policy_name}-{self.time_flag}-{self.train_env_name}-{seed}"
        # if testing_args:
        #     logging.info(f"Testing args : {testing_args}")

        self.load_agent = load_agent
        self.save_model = save_model

        self.running_config = self.get_running_config(kwargs)
        parameters_to_show = self.__dict__.copy()
        logging.info("=" * 10)
        logging.success(f"Bot Created: {self.title}")
        logging.debug(pformat(parameters_to_show))
        logging.debug(self.running_config)

        console.print(
            Panel(Pretty(parameters_to_show),
                  expand=False,
                  title="Bot parameters"))
        self.train_matrix = {}

    def get_agent(self,
                  config=None,
                  cl_config=None,
                  meta_config=None,
                  config_file=None):
        self.config_file = config_file if config_file else ""
        if self.meta_algo:
            return Meta_Agent(
                policy_name=self.policy_name,
                use_wandb=self.use_wandb,
                logger=self.tensorboard_logger,
                config=config,
                meta_algo=self.meta_algo,
                config_file=config_file,
                meta_config=meta_config,
            )
        else:
            return Agent(
                policy_name=self.policy_name,
                use_wandb=self.use_wandb,
                logger=self.tensorboard_logger,
                config_file=config_file,
                config=config,
            )

    def get_running_config(self, kwargs=None):
        running_config = {}
        # running_config["time_flag"] = self.time_flag
        running_config["RL_config"] = copy.deepcopy(self.agent.config.__dict__)
        if self.cl_method:
            running_config["CRL_config"](self.agent.cl_config.__dict__)
        if self.meta_algo:
            running_config[
                "MetaRL_config"] = self.agent.meta_config.__dict__.copy()
        running_config["Running_Title"] = self.title
        running_config["RL_Algo"] = self.policy_name
        running_config[
            "action_vulnerabilities_file"] = Action.action_vulnerabilities.name
        running_config["train_env"] = self.train_env_name
        running_config[
            "eval_env"] = self.eval_env_file.parent.name + "/" + self.eval_env_file.name if self.eval_env_file else ''
        running_config["load_agent"] = self.load_agent

        running_config["seed"] = self.seed
        running_config["state_dim"] = StateEncoder.state_space
        running_config["state_vector"] = StateEncoder.state_vector
        running_config["action_dim"] = Action.action_space
        running_config["config_file"] = self.config_file
        if self.load_agent:
            running_config["loaded"] = True
        else:
            running_config["loaded"] = False
        if kwargs:
            running_config.update(kwargs)
        running_config["host_name"] = self.host_name
        running_config["device"] = "cuda" if torch.cuda.is_available(
        ) else "cpu"
        running_config.update(
            Configure.read_configure_value(
                sections=["Embedding", "Support", "Exploit"]))
        # config_df = pd.DataFrame.from_dict(running_config, orient="index")
        return running_config

    def make_env(self, env_file=None):
        target_list: list[HOST] = []
        env_vuls = []
        with open(env_file, "r", encoding="utf-8") as f:  # *********
            self.environment_data = json.loads(f.read())
            train_ip_list = []
            for host in self.environment_data:
                ip = host["ip"]
                # assert ip not in train_ip_list, f"{ip} aready exist in {env_file}"
                train_ip_list.append(ip)
                vul = host["vulnerability"][0]
                if vul not in Action.Vul_cve_set:
                    logging.error(f"host vul {vul} is not exploitable")
                    exit(0)
                t = HOST(ip, env_data=host, mode=self.mode)
                env_vuls.append(vul)
                target_list.append(t)

        return target_list

    def train(self,
              train_env: list[HOST],
              eval_env: list[HOST] = [],
              eval_model=True,
              verbose=True):
        if not train_env:
            exit(0)
        if not eval_env:
            eval_env = train_env

        console.rule("[bold green]Starting training")

        # random.shuffle(train_env)
        start = time.time()

        UTIL.Running_title = self.title

        if self.meta_algo:
            meta_train_task = train_env[:-1]
            valid_task = train_env[-1]
            eval_task = eval_env
            self.train_matrix = self.agent.train(task_list=meta_train_task,
                                                 valid_task=valid_task,
                                                 eval_task=eval_task)
        else:
            self.train_matrix = self.agent.train(task_list=train_env)
            if self.train_matrix and verbose:
                logging.info("Learning Curve of Train_Episode_Rewards:")
                self.plot_reward(data=self.train_matrix.Train_Episode_Rewards)

        end = time.time()
        self.run_time = time.strftime("%H:%M:%S",
                                      time.gmtime(round(end - start)))
        if self.use_wandb:
            wandb.log(self.train_matrix.save_info)

        if eval_model and eval_env:
            attack_path, mean_eval_rewads, mean_success_rate = self.Eval_Simulate(
                env=eval_env)

        if self.wandb_run:
            self.wandb_run.tags += (self.train_matrix.signal, )

        # save training env data
        self.train_env_data = []
        for e in train_env:
            self.train_env_data.append(e.env_data.dict())

        # eval_sr = self.Eval_Simulate(verbose=verbose)
        cfg = self.log_paras(time=self.time_flag,
                             train_metric=self.train_matrix)

        if self.train_matrix.signal == Metric.Success:

            self.save_experiment_record(cfg=cfg)

            logging.success(f"{self.time_flag} training complete.")
        else:
            logging.warning(f"{self.time_flag} {self.train_matrix.signal}.")

        if self.tensorboard_logger:
            self.tensorboard_logger.add_text(
                "config",
                json.dumps(cfg, indent=2, sort_keys=True, ensure_ascii=False))

            self.tensorboard_logger.add_text(
                "result", pformat(self.train_matrix.save_info))
            self.tensorboard_logger.add_text("running time", self.run_time)
            if eval_model:
                self.tensorboard_logger.add_text("Eval/eval_env",
                                                 self.eval_env_file.name)
                self.tensorboard_logger.add_text("Eval/mean_eval_rewads",
                                                 str(mean_eval_rewads))
                self.tensorboard_logger.add_text("Eval/mean_success_rate",
                                                 str(mean_success_rate))
                for i in range(len(attack_path)):
                    path = attack_path[i]
                    self.tensorboard_logger.add_text(
                        f"Eval/attack_path/path_{i+1}", pformat(path["path"]))
        logging.info(f"Running Time: {self.run_time}")

        # logging.debug(f"Train matrix:\n{pformat(self.train_matrix)}")
        console.rule(f"Bot [bold red]{self.time_flag}[/] Training Over")

    def Eval_Simulate(self,
                      env,
                      eval_times=1,
                      interactive=False,
                      determinate=True,
                      verbose=True):
        Eval_metric = EasyDict({
            "Eval_after_train/attack_path": [],
            "Eval_after_train/mean_eval_rewads": 0,
            "Eval_after_train/mean_success_rate": 0
        })
        mean_eval_rewads = 0
        mean_success_rate = 0.0
        i = 0
        console.rule("[bold green]Starting evaluating")
        logging.success(f"Evaluation task : #{self.eval_env_file.name}")

        while i < eval_times:

            attack_path, eval_rewards, eval_sr = self.agent.Evaluate(
                target_list=env,
                interactive=interactive,
                verbose=verbose,
                determinate=determinate,
                step_limit=10)

            for host_attack_path in attack_path:
                path = host_attack_path["path"]
                table = Table(
                    title=f"Attack Path of {host_attack_path['ip']}",
                    highlight=True,
                    box=box.ROUNDED,
                )

                table.add_column("Step",
                                 justify="left",
                                 style="cyan",
                                 no_wrap=True)

                table.add_column("Action", style="magenta", no_wrap=True)
                table.add_column("Result", style="magenta", no_wrap=True)
                table.add_column("Reward", style="magenta", no_wrap=True)

                for step in path:
                    if len(step.result) > 100:
                        result = step.result[:100] + '...'
                    else:
                        result = step.result
                    table.add_row(str(step.step), step.action, result,
                                  str(step.reward))

                console.print(table, overflow="crop")
            mean_eval_rewads += eval_rewards
            mean_success_rate += eval_sr
            i += 1
            logging.success(f"Evaluation times : #{i}")
            logging.success(f"Evaluation rewards = {eval_rewards}")
            logging.success(f"Success_rate = {eval_sr}")

        mean_eval_rewads = mean_eval_rewads / eval_times
        mean_success_rate = mean_success_rate / eval_times
        if eval_times > 1:
            logging.success(
                f"Mean evaluation rewards = [green]{mean_eval_rewads}[/]")
            logging.success(
                f"Mean success_rate = [green]{mean_success_rate}[/]")
        Eval_metric.attack_path = pformat(attack_path)
        Eval_metric.mean_eval_rewads = mean_eval_rewads
        Eval_metric.mean_success_rate = mean_success_rate
        self.train_matrix.update(Eval_metric)
        if self.use_wandb:
            wandb.log(Eval_metric)
        return attack_path, mean_eval_rewads, mean_success_rate

    def log_paras(self, time, train_metric, log_file=None):

        cfg = {}
        cfg["time_flag"] = self.time_flag
        cfg["status"] = train_metric.signal
        # cfg["Running config"] = self.running_config
        cfg["Train result"] = train_metric.save_info
        cfg["Running_time"] = self.run_time
        for key, value in self.running_config.items():
            cfg[key] = pformat(value)

        header = cfg.keys()

        title = f"{self.agent.name}-{self.agent.policy_name}"
        file = f"log-{title}-{UTIL.today}.csv" if not log_file else log_file

        para_log_path = UTIL.log_path / file
        with open(para_log_path, "a+", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
            writer.writeheader()  # 写入列名
            writer.writerow(cfg)  # 写入数据

        return cfg

    def save_experiment_record(self, cfg):

        path = UTIL.running_record_path / self.title
        if not os.path.exists(path):
            os.makedirs(path)

        if self.save_model:
            # 2 parameters
            UTIL.save_json(path=path / f"config.json", data=cfg)
            # 3 agent model
            self.save_agent(cfg=cfg, path=path)
        logging.success(f"Running record saved in path : {path.name}")

    def save_agent(self, cfg, path):
        # path = UTIL.trained_agent_path  # trained_agent

        # path = path / self.agent.name / self.agent.policy_name / self.time_flag
        path = path / "saved_models"

        if not os.path.exists(path):
            os.makedirs(path)

        # cfg_path = path / f"{self.agent.policy_name}-config.json"
        # with open(cfg_path, "w", encoding="utf-8") as f:
        #     f.write(
        #         json.dumps(self.agent.config.__dict__, ensure_ascii=False, indent=4)
        #     )
        # readme_path = path / f"readme.json"
        # with open(readme_path, "w", encoding="utf-8") as f:
        #     f.write(json.dumps(cfg, ensure_ascii=False, indent=4))
        self.agent.save(path)
        logging.success(f"agent saved in path : {str(path)}")

    def load(self, agent_name):
        # path = UTIL.trained_agent_path  # trained_agent
        # path = path / self.agent.name / self.agent.policy_name / agent_name

        path = UTIL.running_record_path / agent_name / "saved_models"
        self.load_agent = agent_name
        # prefix = f"{self.agent.policy_name}"

        # cfg_path = path / f"{prefix}-config.json"
        # with open(cfg_path, "r", encoding="utf-8") as f:
        #     cfg = json.load(f)
        # if self.agent.config.__dict__ != cfg:
        #     logging.warning(
        #         "Parameters of the trained model do not match those of the loaded model."
        #     )
        #     # self.agent.config.__dict__ = cfg
        #     # self.agent = Agent(name=self.agent_name,
        #     #                    config=self.agent.config).agent
        self.agent.load(path)
        logging.success(f"model load: {path}")

    def plot_reward(self,
                    data: list,
                    smooth=True,
                    width=100,
                    smooth_weight=0.8):
        """
        width: number of sampled points

        """
        rewards = UTIL.smooth_data(data,
                                   weight=smooth_weight) if smooth else data
        length = len(rewards)
        iter = length // width if length > width else 1
        logging.info(
            asciichartpy.plot(
                rewards[0:length:iter],
                {
                    "height": 10,
                    "max": max(data) + 50,
                    "min": min(data) - 50
                },
            ))
