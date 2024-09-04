import sys
import os
from loguru import logger as logging
import time
from tqdm import tqdm, trange
import wandb
import copy
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from collections import namedtuple, defaultdict
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch import autograd
import random
import wandb

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
from util import UTIL, color, Metric
from RL_Policy.common import RewardScaling, Normalization, Memory
from Config import *
from actions import Action
from host import StateEncoder, HOST
from agent import Agent
from RL_Policy.PPO import Actor, Critic, PPO, ReplayBuffer_PPO
from .common import *

Transition = namedtuple(
    "pd_Transition",
    ("state", "action_prob", "action", "reward", "next_state", "done"))


def maml_update(model, lr, grads=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

    **Description**

    Performs a MAML update on model using grads and lr.
    The function re-routes the Python object, thus avoiding in-place
    operations.

    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.

    **Arguments**

    * **model** (Module) - The model to update.
    * **lr** (float) - The learning rate used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.

    **Example**
    ~~~python
    maml = l2l.algorithms.MAML(Model(), lr=0.1)
    model = maml.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    maml_update(model, lr=0.1, grads)
    ~~~
    """
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = "WARNING:maml_update(): Parameters and gradients have different length. ("
            msg += str(len(params)) + " vs " + str(len(grads)) + ")"
            print(msg)
        for p, g in zip(params, grads):
            if g is not None:
                p.update = -lr * g
    return update_module(model)


class MAMLPPO(PPO):
    def __init__(self, cfg: PPO_Config):
        super().__init__(cfg)

    def sample_action(self, observation, epsilon=0, determinate=False):
        # state = torch.tensor([observation], dtype=torch.float).to(self.device)
        # dist = Categorical(probs=self.actor(state))

        # action = dist.sample()
        # probs = torch.squeeze(dist.log_prob(action)).item()
        # action = torch.squeeze(action).item()

        observation = torch.tensor([observation],
                                   dtype=torch.float).to(self.device)
        with torch.no_grad():
            a_prob = self.actor(observation)
            dist = Categorical(probs=a_prob)

            if random.random() < epsilon:
                a = random.randint(0, Action.action_space - 1)
                probs = torch.squeeze(
                    dist.log_prob(torch.tensor([a]).to(self.device))).item()
            else:
                if not determinate:

                    action = dist.sample()
                    probs = torch.squeeze(dist.log_prob(action)).item()
                    a = torch.squeeze(action).item()
                else:
                    a = np.argmax(a_prob.cpu().detach().numpy().flatten())
                    probs = torch.squeeze(
                        dist.log_prob(torch.tensor([a]).to(
                            self.device))).item()

        ## show the networks after the first running
        # if self.first_run:
        #     self.first_run=False
        #     self.logger.add_graph(self.actor,input_to_model=state)
        #     self.logger.add_graph(self.critic,input_to_model=state)
        return (a, probs)


class MAML(Agent):
    def __init__(
        self,
        logger: SummaryWriter = None,
        use_wandb=False,
        policy_name="PPO",
        config: PPO_Config = None,
        meta_config: MAML_config = None,
        config_file=None,
    ):
        super().__init__(logger=logger,
                         use_wandb=use_wandb,
                         policy_name=policy_name,
                         config=config,
                         config_file=config_file)
        self.meta_config = meta_config if meta_config else MAML_config()

        if config:
            self.config = config

        self.meta_lr = self.meta_config.meta_lr
        self.adapt_lr = self.meta_config.adapt_lr
        self.Policy = MAMLPPO(cfg=self.config)
        # self.Policy.memory = ReplayBuffer_PPO(self.config.batch_size,
        #                                       StateEncoder.state_space)
        self.policy_name = self.Policy.name
        self.optimizer = torch.optim.Adam(self.Policy.actor.parameters(),
                                          lr=self.meta_lr)
        self.baseline = LinearValue(StateEncoder.state_space,
                                    Action.action_space)
        self.lr_decay_schedule = np.linspace(1.0, 0.1,
                                             self.meta_config.num_iterations)
        self.epslion_schedule = np.linspace(0, self.meta_config.epslion,
                                            self.meta_config.num_iterations)
        self.best_policy = None
        self.best_state_norm = None
        # self.use_state_norm = False
        self.norm_adv = self.meta_config.norm_adv
        self.name = "MAML"

    def pre_train(self, task_list, eval_freq=5):
        Train_metric = EasyDict({"signal": Metric.Finished})
        self.task_num_episodes = 0
        self.eval_rewards = 0

        with tqdm(
                range(self.config.train_eps),
                position=0,
                leave=True,
                desc=f"{color.color_str('PreTraining',c=color.RED)}",
        ) as tbar:
            for _ in tbar:
                start = time.time()
                self.num_episodes += 1

                ep_results = self.run_train_episode(task_list)

                end = time.time()
                run_time = float(end - start)
                ep_return, ep_steps, success_rate = ep_results
                self.last_episode_reward = ep_return

                if self.num_episodes % eval_freq == 0:
                    self.Evaluate(
                        target_list=task_list,
                        policy=self.Policy,
                        verbose=False,
                        step_limit=self.config.eval_step_limit,
                    )
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

        if self.eval_success_rate > 0.99:
            Train_metric.signal = Metric.Success
        return Train_metric

    def Meta_train(self, task_list, valid_task, eval_task=None, eval_freq=5):
        if self.meta_config.pre_train:
            self.pre_train(task_list=[copy.deepcopy(valid_task)])
            valid_task.reset()

        Train_metric = EasyDict({
            "signal": Metric.Finished,
            "save_info": {},
            "Adapting_iteration_reward": [],
            "Adapting_iteration_SR": [],
            "Valid_task_reward": [],
            "Eval_task_SR": [],
        })

        best_eval_task_sr = 0
        best_eval_task_reward = -1000
        best_valid_task_reward = -1000
        self.iterations = 0
        eval_task_sr = 0
        with tqdm(
                range(self.meta_config.num_iterations),
                position=0,
                leave=True,
                desc=f"{color.color_str('Meta Training',c=color.GREEN)}",
        ) as tbar:
            for _ in tbar:

                iteration_policies = []
                iteration_transitions = []
                iteration_reward = 0.0
                iteration_sr = 0.0
                task_num = 1
                random.shuffle(task_list)
                for task in task_list:
                    task_env = [task]
                    task_transitions = []
                    task_clone_policy_net = copy.deepcopy(self.Policy.actor)
                    task_clone_policy = copy.deepcopy(self.Policy)
                    for i in range(self.meta_config.adapt_steps):
                        task_train_transitions, mean_episodes_rewards = (
                            self.sample_transitions(
                                target=task,
                                batch_size=self.meta_config.adapt_batch_size,
                                Policy=task_clone_policy,
                                desc=f"task {task_num}",
                            ))
                        task_clone_policy_net_ = self.fast_adapt(
                            policy_net=task_clone_policy_net,
                            first_order=True,
                            transitions=task_train_transitions,
                        )
                        task_transitions.append(task_train_transitions)

                    task_validation_transitions, mean_episodes_rewards = (
                        self.sample_transitions(
                            target=task,
                            batch_size=self.meta_config.adapt_batch_size,
                            Policy=task_clone_policy,
                            desc="validation",
                        ))
                    task_num += 1
                    _, _, sr = self.Evaluate(target_list=[valid_task],
                                             policy=task_clone_policy)
                    task_transitions.append(task_validation_transitions)
                    iteration_transitions.append(task_transitions)
                    iteration_policies.append(task_clone_policy_net)
                    iteration_reward += mean_episodes_rewards
                    iteration_sr += sr
                """
                display info
                """
                mean_iteration_reward = int(iteration_reward / len(task_list))
                mean_iteration_sr = iteration_sr / len(task_list)
                self.meta_optimize(iteration_transitions, iteration_policies)
                if self.meta_config.use_lr_decay:
                    self.lr_decay(rate=self.lr_decay_schedule[self.iterations])
                _, Valid_eval_r, _ = self.Evaluate(target_list=[valid_task],
                                                   policy=self.Policy)

                if self.iterations % eval_freq == 0:
                    _, eval_task_r, eval_task_sr = self.Evaluate(
                        target_list=eval_task, policy=self.Policy)

                    if eval_task_r >= best_eval_task_reward and Valid_eval_r > 900:
                        best_eval_task_sr = eval_task_sr
                        best_eval_task_reward = eval_task_r
                        best_valid_task_reward = Valid_eval_r
                        self.best_policy = copy.deepcopy(self.Policy)
                        if self.use_state_norm:
                            self.best_state_norm = copy.deepcopy(
                                self.state_norm)

                tbar.set_postfix(
                    iteration_reward=color.color_str(
                        f"{mean_iteration_reward}", c=color.PURPLE),
                    Valid_eval_r=color.color_str(f"{Valid_eval_r}/{best_valid_task_reward}",
                                                 c=color.GREEN),
                    iteration_sr=color.color_str(f"{mean_iteration_sr}",
                                                 c=color.YELLOW),
                    eval_task_sr=color.color_str(f"{eval_task_sr}/{best_eval_task_sr}",
                                                 c=color.RED),
                    eval_task_r=color.color_str(f"{eval_task_r}/{best_eval_task_reward}", c=color.RED),
                )

                Train_metric.Adapting_iteration_reward.append(
                    mean_iteration_reward)
                Train_metric.Adapting_iteration_SR.append(mean_iteration_sr)
                Train_metric.Valid_task_reward.append(Valid_eval_r)
                Train_metric.Eval_task_SR.append(eval_task_sr)
                if self.tf_logger:
                    self.tf_logger.add_scalar(
                        "Adapting/iteration reward",
                        mean_iteration_reward,
                        self.iterations,
                    )
                    self.tf_logger.add_scalar("MRL-Adapting/iteration SR",
                                              mean_iteration_sr,
                                              self.iterations)
                    self.tf_logger.add_scalar("MRL-Adapting/iteration reward",
                                              mean_iteration_reward,
                                              self.iterations)
                    self.tf_logger.add_scalar("MRL-Valid_task/reward",
                                              Valid_eval_r, self.iterations)

                    self.tf_logger.add_scalar("MRL-Eval_tasks/SR",
                                              eval_task_sr, self.iterations)
                    self.tf_logger.add_scalar("MRL-Eval_tasks/mean_reward",
                                              eval_task_r, self.iterations)

                    self.tf_logger.add_scalar("Best_policy/best_eval_task_sr",
                                              best_eval_task_sr,
                                              self.iterations)
                    self.tf_logger.add_scalar(
                        "Best_policy/best_eval_task_reward",
                        best_eval_task_reward, self.iterations)
                    self.tf_logger.add_scalar(
                        "Best_policy/best_valid_task_reward",
                        best_valid_task_reward, self.iterations)

                if self.use_wandb:
                    wandb.log({
                        "MRL-Adapting/iteration SR": mean_iteration_sr,
                        "MRL-Adapting/iteration reward": mean_iteration_reward,
                        "MRL-Valid_task/reward": Valid_eval_r,
                        "MRL-Eval_tasks/SR": eval_task_sr,
                        "MRL-Eval_tasks/mean_reward": eval_task_r,
                        "Best_policy/best_eval_task_sr": best_eval_task_sr,
                        "Best_policy/best_eval_task_reward":
                        best_eval_task_reward,
                        "Best_policy/best_valid_task_reward":
                        best_valid_task_reward,
                        "MRL-iterations": self.iterations,
                    })

                self.iterations += 1

                # if self.iterations in [ 30] and best_eval_task_sr < 0.5:
                #     Train_metric.signal = Metric.EarlyTerminate
                #     break

        if best_valid_task_reward > 900 and best_eval_task_sr > 0.99:
            Train_metric.signal = Metric.Success
        # else:
        #     self.Policy = best_policy
        Train_metric.save_info["best_eval_task_sr"] = best_eval_task_sr
        Train_metric.save_info["best_eval_task_reward"] = best_eval_task_reward
        Train_metric.save_info["best_valid_task_reward"] = best_valid_task_reward
        Train_metric.save_info["Valid_task_reward"] = Valid_eval_r
        Train_metric.save_info["Eval_task_SR"] = eval_task_sr
        return Train_metric

    def meta_loss(self, iteration_transitions, iteration_policies,
                  policy_net: Actor):
        mean_loss = 0.0
        for task_replays, old_policy in zip(iteration_transitions,
                                            iteration_policies):
            train_replays = task_replays[:-1]
            valid_transitions = task_replays[-1]
            new_policy_net = clone_module(policy_net)

            # Fast Adapt
            for transitions in train_replays:
                new_policy_net = self.fast_adapt(
                    policy_net=new_policy_net,
                    first_order=False,
                    transitions=transitions,
                )

            states, actions, a_logprob, r, s_, dw, done = (
                valid_transitions.numpy_to_tensor(device=self.Policy.device)
            )  # Get training data

            # Useful values
            dist_old = Categorical(probs=old_policy(states))
            a_logprob_old = dist_old.log_prob(actions.squeeze()).view(
                -1, 1).detach()
            dist_new = Categorical(probs=new_policy_net(states))
            a_logprob_new = dist_new.log_prob(actions.squeeze()).view(-1, 1)
            dist_entropy = dist_new.entropy().view(
                -1, 1)  # shape(mini_batch_size X 1)

            advantages = self.compute_advantages(
                gamma=self.config.gamma,
                rewards=r,
                dones=done,
                states=states,
                next_states=s_,
            )
            if self.norm_adv:
                advantages = normalize(advantages)
            advantages = advantages.detach()
            ratios = torch.exp(a_logprob_new - a_logprob_old)

            # Only calculate the gradient of 'a_logprob_now' in ratios
            surr1 = ratios * advantages
            surr2 = (torch.clamp(ratios, 1 - self.Policy.policy_clip,
                                 1 + self.Policy.policy_clip) * advantages)
            actor_loss = -torch.min(surr1, surr2)
            actor_loss = (actor_loss - self.Policy.entropy_coef * dist_entropy
                          )  # shape(mini_batch_size X 1)
            actor_loss = actor_loss.mean()
            mean_loss += actor_loss

        mean_loss /= len(iteration_transitions)
        return mean_loss

    def meta_optimize(self, iteration_transitions, iteration_policies):
        with tqdm(
                range(self.meta_config.meta_optimize_times),
                leave=False,
                desc=f"{color.color_str('Meta optimizing',c=color.YELLOW)}",
        ) as tbar:
            for _ in tbar:

                loss = self.meta_loss(iteration_transitions,
                                      iteration_policies, self.Policy.actor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tbar.set_postfix(loss=color.color_str(f"{loss.item()}",
                                                      c=color.PURPLE), )

    def compute_advantages(self, gamma, rewards, dones, states, next_states):
        # Update baseline
        returns = discount(gamma, rewards, dones)

        self.baseline.fit(states, returns)
        values = self.baseline(states)
        next_values = self.baseline(next_states)
        bootstraps = values * (1.0 - dones) + next_values * dones
        next_value = torch.zeros(1, device=self.Policy.device)
        return generalized_advantage(
            tau=1,
            gamma=gamma,
            rewards=rewards,
            dones=dones,
            values=bootstraps,
            next_value=next_value,
        )

    def maml_a2c_loss(self, policy_net: Actor, memory: ReplayBuffer_PPO):
        # Update policy and baseline
        s, a, a_logprob, r, s_, dw, done = memory.numpy_to_tensor(
            device=self.Policy.device)  # Get training data
        dist = Categorical(probs=policy_net(s))

        action = dist.sample()
        logprobs = dist.log_prob(action).mean(dim=-1, keepdim=True)
        logprobs_ = dist.log_prob(action.squeeze()).view(-1, 1)
        # probs = torch.squeeze(dist.log_prob(action)).item()
        advantages = self.compute_advantages(gamma=self.meta_config.gamma,
                                             rewards=r,
                                             dones=done,
                                             states=s,
                                             next_states=s_)
        if self.norm_adv:
            advantages = normalize(advantages)
        loss = -torch.mean(logprobs_ * advantages.detach())
        return loss

    def fast_adapt(self,
                   policy_net: Actor,
                   transitions: ReplayBuffer_PPO,
                   first_order=False):
        second_order = not first_order
        loss = self.maml_a2c_loss(policy_net=policy_net, memory=transitions)
        gradients = autograd.grad(
            loss,
            policy_net.parameters(),
            retain_graph=second_order,
            create_graph=second_order,
        )
        return maml_update(model=policy_net, grads=gradients, lr=self.adapt_lr)

    def sample_transitions(
        self,
        target: HOST,
        batch_size,
        Policy: MAMLPPO,
        desc: str,
        epsilon=0.1,
        update_norm=True,
    ):
        iteration_memory = ReplayBuffer_PPO(self.meta_config.adapt_batch_size,
                                            StateEncoder.state_space)
        data_num = 0
        episode = 0
        total_episodes_rewards = 0
        with tqdm(
                range(int(batch_size)),
                leave=False,
                desc=
                f"{color.color_str(f'Sampling Transitions-{desc}',c=color.CYAN)}",
        ) as pbar:
            while data_num < batch_size:
                # for data_num in pbar:
                steps = 0
                done = 0
                episode += 1
                episode_return = 0
                o = target.reset()
                if self.use_state_norm:
                    o = self.state_norm(o, update=False)
                while not done and steps < self.config.step_limit:
                    """
                    Output an action
                    """
                    action_info = Policy.sample_action(
                        observation=o,
                        epsilon=self.epslion_schedule[self.iterations],
                        determinate=False)
                    a = action_info[0]  # action_info 中第一位为动作id

                    next_o, r, done, result = target.perform_action(a)
                    episode_return += r
                    steps += 1
                    """
                    Store the transition
                    """
                    if done:
                        dw = True
                    else:
                        dw = False
                    if self.use_state_norm:
                        next_o = self.state_norm(next_o, update=update_norm)
                    if self.use_reward_scaling:
                        r = self.reward_scaling(r)[0]
                    iteration_memory.store(
                        s=o,
                        a=action_info[0],
                        a_logprob=action_info[1],
                        r=r,
                        s_=next_o,
                        dw=done,
                        done=done,
                    )
                    o = next_o

                    data_num += 1

                    if data_num >= batch_size:
                        break
                pbar.update(steps)
                pbar.set_postfix(
                    r=color.color_str(f"{episode_return}", c=color.PURPLE),
                    step=color.color_str(f"{steps}", c=color.GREEN),
                )
                total_episodes_rewards += episode_return
        mean_episodes_rewards = total_episodes_rewards / episode
        return iteration_memory, mean_episodes_rewards

    def lr_decay(self, rate):

        meta_lr_now = self.meta_lr * rate
        self.adapt_lr = self.adapt_lr * rate
        for p in self.optimizer.param_groups:
            p['lr'] = meta_lr_now

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        assert os.path.exists(path), f"{path} does not exist"
        if self.use_state_norm:
            mean = self.best_state_norm.running_ms.mean
            std = self.best_state_norm.running_ms.std
            mean_checkpoint = path / f"state_norm_mean.pt"
            std_checkpoint = path / f"state_norm_std.pt"
            torch.save(mean, mean_checkpoint)
            torch.save(std, std_checkpoint)
        self.best_policy.save(path)
        baseline_checkpoint = path / f"{self.name}-baseline.pt"
        torch.save(self.baseline.state_dict(), baseline_checkpoint)

    def load(self, path):

        if self.use_state_norm:
            mean_checkpoint = path / f"state_norm_mean.pt"
            std_checkpoint = path / f"state_norm_std.pt"
            mean = torch.load(mean_checkpoint)
            std = torch.load(std_checkpoint)
            self.state_norm.running_ms.mean = mean
            self.state_norm.running_ms.std = std
        self.Policy.load(path)
        baseline_checkpoint = path / f"{self.name}-baseline.pt"
        if torch.cuda.is_available():
            self.baseline.load_state_dict(torch.load(baseline_checkpoint))
        else:
            self.baseline.load_state_dict(
                torch.load(baseline_checkpoint,
                           map_location=torch.device("cpu")))
        self.is_loaded_agent = True
