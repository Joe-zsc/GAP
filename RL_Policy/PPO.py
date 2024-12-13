import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as logging
import torch.optim as optim
import sys
import os
from torch.utils.tensorboard import SummaryWriter
import wandb

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
from util import UTIL
from Config import PPO_Config
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions.categorical import Categorical
from common import orthogonal_init, ReplayBuffer_PPO, build_net, BasePolicy
from actions import Action
from host import StateEncoder
'''
PPO version from https://github.com/Lizhi-sjtu/DRL-code-pytorch
'''


def _weight_init(module):
    if isinstance(module, nn.Linear):
        # nn.init.xavier_uniform_(module.weight)
        # module.bias.data.zero_()
        module.reset_parameters()


        # nn.init.normal_(module.weight.data, 0, 0.01)
        # module.bias.data.zero_()
class Actor(nn.Module):

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_shape,
                 use_orthogonal_init=False,
                 activate_func="relu"):
        super(Actor, self).__init__()
        self.net = build_net(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_shape=hidden_shape,
            #  use_layer_norm=True,
            # use_batchnorm=True,
            hid_activation=activate_func)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_shape = hidden_shape
        self.activate_func = activate_func

    def forward(self, state):
        dist = F.softmax(self.net(state), dim=-1)
        # dist = Categorical(dist)
        return dist

    def reset(self):

        self.apply(self._weight_init)

    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            # nn.init.xavier_uniform_(module.weight)
            # module.bias.data.zero_()
            module.reset_parameters()
            # orthogonal_init(module)


class Critic(nn.Module):

    def __init__(self,
                 state_dim,
                 hidden_shape,
                 use_orthogonal_init=False,
                 activate_func="relu"):
        super(Critic, self).__init__()

        self.net = build_net(
            input_dim=state_dim,
            output_dim=1,
            hidden_shape=hidden_shape,
            #  use_layer_norm=True,
            # use_batchnorm=True,
            hid_activation=activate_func)
        self.state_dim = state_dim
        self.hidden_shape = hidden_shape
        self.activate_func = activate_func

    def forward(self, state):
        value = self.net(state)
        return value

    def reset(self):
        self.apply(self._weight_init)

    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            # nn.init.xavier_uniform_(module.weight)
            # module.bias.data.zero_()
            module.reset_parameters()
            # orthogonal_init(module)


class PPO(BasePolicy):

    def __init__(self, cfg: PPO_Config):
        super().__init__()
        self.name = "PPO"
        self.config = cfg
        self.gamma = self.config.gamma
        self.policy_clip = self.config.policy_clip
        self.ppo_update_time = self.config.ppo_update_time
        self.gae_lambda = self.config.gae_lambda

        # self.device=torch.device("cpu")
        self.activate_func = self.config.activate_func
        self.actor = Actor(StateEncoder.state_space,
                           Action.action_space,
                           self.config.hidden_sizes,
                           use_orthogonal_init=self.config.use_orthogonal_init,
                           activate_func=self.activate_func).to(self.device)

        self.critic = Critic(
            StateEncoder.state_space,
            self.config.hidden_sizes,
            use_orthogonal_init=self.config.use_orthogonal_init,
            activate_func=self.activate_func).to(self.device)

        self.a_lr = self.config.actor_lr
        self.c_lr = self.config.critic_lr
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
            eps=self.config.Adam_Optimizer_Epsilon)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr,
            eps=self.config.Adam_Optimizer_Epsilon)
        self.memory = ReplayBuffer_PPO(self.config.batch_size,
                                       StateEncoder.state_space)
        self.loss = 0
        self.batch_size = self.config.batch_size
        self.mini_batch_size = self.config.mini_batch_size
        self.train_episodes = self.config.train_eps
        self.entropy_coef = self.config.entropy_coef
        self.first_run = True
        self.test = 0
        self.update_steps = 0
        self.gradiant_clip = self.config.gradiant_clip

    '''
    Required Functions for All Algos
    '''

    def select_action(self, observation, explore, is_loaded_agent,
                      num_episode):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        dist = Categorical(probs=self.actor(state))

        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()

        ## show the networks after the first running
        # if self.first_run:
        #     self.first_run=False
        #     self.logger.add_graph(self.actor,input_to_model=state)
        #     self.logger.add_graph(self.critic,input_to_model=state)
        return (action, probs)

    def store_transtion(self, observation, action, reward, next_observation,
                        done):
        self.memory.store(s=observation,
                          a=action[0],
                          a_logprob=action[1],
                          r=reward,
                          s_=next_observation,
                          dw=done,
                          done=done)

    def update_policy(self, num_episode, train_steps):
        if self.memory.count == self.batch_size:
            self.update(train_steps)
            self.memory.count = 0

    def evaluate(
        self,
        observation,
        determinate=True
    ):  # When evaluating the policy, we select the action with the highest probability

        observation = torch.tensor([observation],
                                   dtype=torch.float).to(self.device)
        with torch.no_grad():
            a_prob = self.actor(observation)

            if not determinate:
                dist = Categorical(probs=a_prob)
                action = dist.sample()
                a = torch.squeeze(action).item()
            else:
                a = np.argmax(a_prob.cpu().detach().numpy().flatten())

        return int(a)

    def lr_decay(self, rate):

        lr_a_now = self.a_lr * rate
        lr_c_now = self.c_lr * rate
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now

    def save(self, path):

        actor_checkpoint = os.path.join(path, f"{self.name}-actor.pt")
        critic_checkpoint = os.path.join(path, f"{self.name}-critic.pt")
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)

    def load(self, path):
        self.gradiant_clip = False
        self.policy_clip = 0.5
        actor_checkpoint = os.path.join(path, f"{self.name}-actor.pt")
        critic_checkpoint = os.path.join(path, f"{self.name}-critic.pt")
        if torch.cuda.is_available():
            self.actor.load_state_dict(torch.load(actor_checkpoint))
            # self.critic.load_state_dict(torch.load(critic_checkpoint))
        else:
            self.actor.load_state_dict(
                torch.load(actor_checkpoint, map_location=torch.device('cpu')))
            # self.critic.load_state_dict(
            #     torch.load(critic_checkpoint,
            #                map_location=torch.device('cpu')))

    '''
    ######################################################################################
    '''

    def update(self, train_steps):
        s, a, a_logprob, r, s_, dw, done = self.memory.numpy_to_tensor(
            device=self.device)  # Get training data

        self.update_steps += 1
        # done=done.to(self.device)
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = (r + self.gamma * (1.0 - dw) * vs_ - vs).cpu()
            for delta, d in zip(reversed(deltas.flatten().numpy()),
                                reversed(done.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1,
                                                            1).to(self.device)
            v_target = adv + vs
            if self.config.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        for _ in range(self.ppo_update_time):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.batch_size)),
                    self.mini_batch_size, False):
                dist_now = Categorical(probs=self.actor(s[index]))
                dist_entropy = dist_now.entropy().view(
                    -1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(
                    -1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                # shape(mini_batch_size X 1)
                ratios = torch.exp(a_logprob_now - a_logprob[index])

                # Only calculate the gradient of 'a_logprob_now' in ratios
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.policy_clip,
                                    1 + self.policy_clip) * adv[index]
                actor_loss = -torch.min(surr1, surr2)
                actor_loss = actor_loss - self.entropy_coef * \
                    dist_entropy  # shape(mini_batch_size X 1)
                actor_loss = actor_loss.mean()
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Trick 7: Gradient clip
                if self.gradiant_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                   0.5)
                self.actor_optimizer.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # Trick 7: Gradient clip
                if self.gradiant_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                                   0.5)
                self.critic_optimizer.step()
        # if self.tf_logger:
        #     self.tf_logger.add_scalar("loss/actor_loss", actor_loss.item(),
        #                               train_steps)
        #     self.tf_logger.add_scalar("loss/critic_loss", critic_loss.item(),
        #                               train_steps)
        # if self.use_wandb:
        #     wandb.log({
        #         "loss/actor_loss": actor_loss.item(),
        #         "loss/critic_loss": critic_loss.item(),
        #         "loss/delta": deltas.mean().item(),
        #         "step/ppo_update": self.update_steps
        #     })
