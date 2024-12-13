import sys
import os
from loguru import logger as logging
import time
from tqdm import tqdm, trange
import wandb
import random
from pprint import pformat
import torch
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
from util import UTIL, color, Metric
from RL_Policy.common import RewardScaling, Normalization
from Config.RL_config import *
from actions import Action
from host import StateEncoder, HOST
import RL_Policy
from defination import Attack_Path_Transition


class BaseAgent:

    def __init__(
        self,
        logger: SummaryWriter,
        policy_name,
        cfg: config = None,
        use_wandb=False,
        config_file=None,
    ):
        self.policy_name = policy_name
        self.config = cfg
        self.tf_logger = logger
        self.use_wandb = use_wandb

        hyperparameters = self.read_config_file(
            config_file) if config_file else {}
        self.Policy = self.get_policy(hyperparameters=hyperparameters)
        self.name = f"RL"

    def read_config_file(self, config_file):
        HyperParameters = {}
        config_ = UTIL.read_yaml(config_file)
        config_keys = list(config_.keys())
        if "RL_Policy" in config_keys:
            RL_Policy = config_["RL_Policy"]["name"]
            if RL_Policy != self.policy_name:
                logging.critical(
                    f"policy_name in Bot is {self.policy_name}, while in config file is {RL_Policy}"
                )
            HyperParameters = config_["RL_Policy"]["HyperParameters"]
        return HyperParameters

    def get_policy(self, hyperparameters: dict = {}):
        if self.policy_name == "PPO":

            if not self.config:
                self.config = PPO_Config(**hyperparameters)
            return RL_Policy.PPO(cfg=self.config)
        else:
            return None



class Agent(BaseAgent):

    def __init__(
        self,
        logger: SummaryWriter = None,
        use_wandb=False,
        policy_name="PPO",
        config: config = None,
        config_file=None,
    ):
        super().__init__(logger=logger,
                         use_wandb=use_wandb,
                         policy_name=policy_name,
                         config_file=config_file,
                         cfg=config)
        self.is_loaded_agent = False
        try:
            self.use_reward_scaling = self.config.use_reward_scaling
            self.use_state_norm = self.config.use_state_norm
            self.use_lr_decay = self.config.use_lr_decay
        except:
            self.use_reward_scaling = False
            self.use_state_norm = False
            self.use_lr_decay = False

        if self.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1,
                                                gamma=self.config.gamma)
        if self.use_state_norm:
            self.state_norm = Normalization(shape=StateEncoder.state_space)
        self.num_episodes = 0
        self.eval_times = 0

        self.task_num_episodes = 0
        self.total_training_step = 0
        self.best_return = -float("inf")
        self.best_action_set = []
        self.best_episode = 0
        self.best_reward_episode = []
        self.eval_rewards = 0
        self.eval_success_rate = 0
        self.is_loaded_agent = False
        self.max_reward = 1000
        self.min_reward = -1000
        self.mean_exp_coverage = 0
        self.last_episode_reward = -float("inf")
        

    def decison_making(self, observation):
        if self.use_state_norm:
            observation = self.state_norm(observation, update=False)
        a = self.Policy.evaluate(observation)
        return a



    def train(self, task_list, eval_freq=5):
        Train_metric = EasyDict({
            "signal": Metric.Finished,
            "save_info": {},
            "Train_Episode_Rewards": [],
            "Train_Episode_Steps": [],
            "Train_Success_Rate": [],
            "Train_Episode_Time": [],
            "Eval_Episode_Rewards": [],
            "Eval_Success_Rate": [],
            "last_task": -1,
        })
        self.first_hit_step = [-1] * len(task_list)
        self.first_hit_exp_coverage = [-1] * len(task_list)
        self.task_num_episodes = 0
        self.eval_rewards = 0
        """
        explore stage: prepare transitions
        """
        with tqdm(
                range(self.config.explore_eps),
                position=0,
                leave=True,
                desc=color.color_str("Exploring", c=color.RED),
        ) as tbar:
            for _ in tbar:
                ep_results = self.run_train_episode(
                    task_list,
                    explore=True,
                    update_norm=not self.is_loaded_agent)
                ep_return, ep_steps, success_rate = ep_results
                tbar.set_postfix(
                    ep_return=color.color_str(f"{ep_return}", c=color.PURPLE),
                    ep_steps=color.color_str(f"{ep_steps}", c=color.GREEN),
                )
        """
        exploit stage: train policy
        
        """
        with tqdm(
                range(self.config.train_eps),
                position=0,
                leave=True,
                desc=f"{color.color_str('Training',c=color.RED)}",
        ) as tbar:
            for _ in tbar:
                start = time.time()
                self.num_episodes += 1

                ep_results = self.run_train_episode(
                    task_list, update_norm=not self.is_loaded_agent)

                end = time.time()
                run_time = float(end - start)
                ep_return, ep_steps, success_rate = ep_results
                self.last_episode_reward = ep_return
                Train_metric.Train_Episode_Rewards.append(ep_return)
                Train_metric.Train_Episode_Steps.append(ep_steps)
                Train_metric.Train_Success_Rate.append(success_rate)
                Train_metric.Train_Episode_Time.append(run_time)
                if self.tf_logger:
                    self.tf_logger.add_scalar("Train/Episode_Rewards",
                                              ep_return, self.num_episodes)
                    self.tf_logger.add_scalar("Train/Episode_Steps", ep_steps,
                                              self.num_episodes)
                    self.tf_logger.add_scalar("Train/Success_Rate",
                                              success_rate, self.num_episodes)
                    self.tf_logger.add_scalar("Train/Episode_Time", run_time,
                                              self.num_episodes)
                    self.tf_logger.add_scalar(
                        "Train/Mean_Exp_coverage",
                        self.mean_exp_coverage,
                        self.num_episodes,
                    )
                    if self.use_state_norm:
                        self.tf_logger.add_scalar(
                            "Auxillary/state_norm_mean",
                            self.state_norm.running_ms.mean.mean(),
                            self.num_episodes,
                        )
                if self.use_wandb:
                    wandb.log({
                        "Train/Episode_Rewards": ep_return,
                        "Train/Success_Rate": success_rate,
                        "Train/Episode_Steps": ep_steps,
                        "Train/Episode_Time": run_time,
                        "Total_Train_Steps": self.total_training_step,
                        "num_episodes": self.num_episodes,
                    })

                if self.num_episodes % eval_freq == 0:
                    self.Evaluate(
                        target_list=task_list,
                        policy=self.Policy,
                        verbose=False,
                        step_limit=self.config.eval_step_limit,
                    )
                    Train_metric.Eval_Episode_Rewards.append(self.eval_rewards)
                    Train_metric.Eval_Success_Rate.append(
                        self.eval_success_rate)
                    if self.tf_logger:
                        self.tf_logger.add_scalar("Eval/Episode_Rewards",
                                                  self.eval_rewards,
                                                  self.num_episodes)
                        self.tf_logger.add_scalar(
                            "Eval/Success_Rate",
                            self.eval_success_rate,
                            self.num_episodes,
                        )

                    if self.use_wandb:
                        wandb.log({
                            "Eval/Episode_Rewards": self.eval_rewards,
                            "Eval/Success_Rate": self.eval_success_rate,
                        })
                """
                display info
                """
                tbar.set_postfix(
                    re_t=color.color_str(f"{ep_return}/{self.best_return}",
                                         c=color.PURPLE),
                    step=color.color_str(f"{ep_steps}", c=color.GREEN),
                    re_e=color.color_str(f"{self.eval_rewards}", c=color.BLUE),
                    rate_e=color.color_str(f"{self.eval_success_rate*100}%",
                                           c=color.CYAN),
                    rate_t=color.color_str(f"{success_rate*100}%",
                                           c=color.YELLOW),
                )

        Train_metric.save_info["eval_rewards"] = self.eval_rewards
        Train_metric.save_info["eval_success_rate"] = self.eval_success_rate
        Train_metric.save_info[
            "total_training_step"] = self.total_training_step

        first_hit_step = self.first_hit_step[0] if len(
            self.first_hit_step) == 1 else pformat(self.first_hit_step)
        self.first_hit_exp_coverage = self.first_hit_exp_coverage[0] if len(
            self.first_hit_exp_coverage) == 1 else pformat(self.first_hit_exp_coverage)
        
        Train_metric.save_info["first_hit"] = {
            "step": first_hit_step,
            "exp_action_coverage": self.first_hit_exp_coverage
        }
        
        
        if self.eval_success_rate > 0.99:
            Train_metric.signal = Metric.Success
        return Train_metric

    def run_train_episode(self, target_list, explore=False, update_norm=True):

        eps_steps = 0
        episode_return = 0
        self.action_set = []
        self.reward_set = []
        success_num = 0
        failed_num = 0
        target_id = 0
        self.task_num_episodes += 1

        if self.use_reward_scaling:
            self.reward_scaling.reset()
        # for target_id in range(len(self.target_list)):
        # random.shuffle(target_list)
        mean_exp_coverage = 0
        while target_id < len(target_list):
            done = 0
            target_step = 0
            target: HOST = target_list[target_id]
            """
            Init observation
            """
            o = target.reset()
            if self.use_state_norm:
                o = self.state_norm(o, update=update_norm)

            while not done:

                if target_step >= self.config.step_limit:
                    break
                """
                Output an action
                """
                action_info = self.Policy.select_action(
                    observation=o,
                    explore=explore,
                    is_loaded_agent=self.is_loaded_agent,
                    num_episode=self.task_num_episodes,
                )
                a = action_info[0]  # action_info 中第一位为动作id
                self.action_set.append(a)
                """
                Perform the action
                """
                next_o, r, done, result = target.perform_action(a)
                self.total_training_step += 1
                eps_steps += 1
                target_step += 1
                episode_return += r
                self.reward_set.append(r)
                """
                Store the transition
                """
                if done:
                    success_num += 1
                    dw = True
                    if self.first_hit_step[target_id] < 0:
                        self.first_hit_step[
                            target_id] = self.total_training_step
                        self.first_hit_exp_coverage[
                            target_id] = target.action.exp_coverage

                else:
                    dw = False
                if self.use_state_norm:
                    next_o = self.state_norm(next_o, update=update_norm)
                if self.use_reward_scaling:
                    r = self.reward_scaling(r)[0]
                self.Policy.store_transtion(
                    observation=o,
                    action=action_info,
                    reward=r,
                    next_observation=next_o,
                    done=dw,
                )
                """
                Update the policy
                """
                if not explore:

                    self.Policy.update_policy(
                        num_episode=self.task_num_episodes,
                        train_steps=self.total_training_step,
                    )
                    if self.use_lr_decay:
                        # NOTE Only support PPO
                        rate = ((
                            1 - self.task_num_episodes / self.config.train_eps)
                                if self.task_num_episodes
                                < self.config.train_eps else 1)
                        if rate <= self.config.min_decay_lr:
                            rate = self.config.min_decay_lr
                        self.Policy.lr_decay(rate=rate)
                o = next_o
            if not done:
                failed_num += 1
                # break
            target_id += 1
            mean_exp_coverage += target.action.exp_coverage
        self.mean_exp_coverage = mean_exp_coverage / len(target_list)
        sucess_rate = float(format(success_num / len(target_list), ".3f"))
        if episode_return >= self.best_return:
            self.best_return = episode_return
            self.best_action_set = self.action_set
            self.best_reward_episode = self.reward_set
            self.best_episode = self.num_episodes
        return episode_return, eps_steps, sucess_rate

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
        sucess_rate = 0.0
        target_id = 0
        total_rewards = 0
        sucess_list = []
        faild_list = []
        attack_path = []
        attack_path_key = ["target", "step", "action", "result", "reward"]
        Policy = self.Policy if not policy else policy
        # random.shuffle(target_list)
        while target_id < len(target_list):
            host: HOST = target_list[target_id]
            host_attack_path = {}
            host_attack_path["ip"] = host.ip
            host_attack_path["path"] = []
            if interactive:
                UTIL.line_break(symbol="=", length=50)
                logging.info("testing: " + host.ip)
                UTIL.line_break(symbol="=", length=50)
            o = host.reset()
            if self.use_state_norm:
                o = self.state_norm(o, update=False)
            done = 0
            steps = 0
            task_return = 0
            if interactive:
                input("Press enter to continue...")
            while not done and steps < step_limit:
                # process = dict.fromkeys(attack_path_key, None)
                if not manual:
                    if determinate:
                        with torch.no_grad():
                            a = Policy.evaluate(o)
                    else:
                        action_info = Policy.select_action(
                            observation=o,
                            explore=False,
                            is_loaded_agent=self.is_loaded_agent,
                            num_episode=0,
                        )
                        a = action_info[0]  # action_info 中第一位为动作id

                else:
                    a = input(
                        "Please select action number, input '-1' to exit: ")
                    a = int(a)
                next_o, r, done, result = host.perform_action(a)
                if self.use_state_norm:
                    next_o = self.state_norm(next_o, update=False)
                o = next_o
                total_rewards += r
                task_return += r
                steps += 1
                if interactive:
                    UTIL.line_break(symbol="-", length=50)
                    logging.info(f"Step {steps}")
                    logging.info(f"Action Performed = {Action.get_action(a)}")
                    logging.info(f"Result = {result}")
                    logging.info(f"Reward = {r}")
                # logging.info(f"Done = {done}")
                # input("Press enter to continue..")

                process = Attack_Path_Transition(
                    ip=host.ip,
                    step=steps,
                    action=Action.get_action(a),
                    result=result,
                    reward=r,
                )

                host_attack_path["path"].append(process)

                if done:
                    if interactive:
                        logging.info("SUCCESS: " + host.ip)
                        logging.info(f"Total steps = {steps}")
                        logging.info(f"Total reward = {total_rewards}")
                    sucess_list.append(host.ip)
                    break
            if not done:

                faild_list.append(host.ip)
            host_attack_path["reward"] = task_return
            host_attack_path["success"] = done
            host_attack_path_ = EasyDict(host_attack_path)
            attack_path.append(host_attack_path)
            target_id += 1
        sucess_rate = float(format(len(sucess_list) / len(target_list), ".3f"))

        self.eval_success_rate = sucess_rate
        mean_eval_rewards = float(
            format(total_rewards / len(target_list), ".3f"))

        self.eval_rewards = mean_eval_rewards
        # logging.info(f"FAILE HOSTS: {faild_list}")
        return attack_path, mean_eval_rewards, sucess_rate

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        assert os.path.exists(path), f"{path} does not exist"
        if self.use_state_norm:
            mean = self.state_norm.running_ms.mean
            std = self.state_norm.running_ms.std
            mean_checkpoint = path / f"state_norm_mean.pt"
            std_checkpoint = path / f"state_norm_std.pt"
            torch.save(mean, mean_checkpoint)
            torch.save(std, std_checkpoint)
        self.Policy.save(path)

    def load(self, path):
        if self.use_state_norm:
            self.state_norm = Normalization(shape=StateEncoder.state_space,
                                            finetune=True)
            mean_checkpoint = path / f"state_norm_mean.pt"
            std_checkpoint = path / f"state_norm_std.pt"
            mean = torch.load(mean_checkpoint)
            std = torch.load(std_checkpoint)
            self.state_norm.running_ms.mean = mean
            self.state_norm.running_ms.std = std
            self.state_norm.running_ms.S = std * std
        self.Policy.load(path)
        self.is_loaded_agent = True
