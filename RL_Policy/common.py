import numpy
import random
import numpy as np
import torch
from pprint import pprint, pformat
import time
from torch import nn, Tensor
from enum import Enum
import torch.nn.functional as F
from loguru import logger as logging
from collections import namedtuple
from itertools import chain, repeat
from torch.utils.tensorboard import SummaryWriter
from actions import Action
from host import StateEncoder


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


def clamp(n, min_, max_):
    return max(min_, min(n, max_))


class BasePolicy:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = StateEncoder.state_space
        self.action_dim = Action.action_space


def build_net(
        input_dim: int,
        output_dim: int,
        hidden_shape: list,
        use_layer_norm=False,
        use_batchnorm=False,
        hid_activation="relu",
        use_orthogonal_init=False,
        output_activation=nn.Identity(),
):
    """build net with for loop"""
    if hid_activation == "relu":
        hid_activation_ = nn.ReLU
    elif hid_activation == "leaky_relu":
        hid_activation_ = nn.LeakyReLU
    elif hid_activation == "tanh":
        hid_activation_ = nn.Tanh
    elif hid_activation == "softsign":
        hid_activation_ = nn.Softsign
    elif hid_activation == "tanhshrink":
        hid_activation_ = nn.Tanhshrink
    elif hid_activation == "elu":
        hid_activation_ = nn.ELU
    else:
        logging.error("activate_func error")
        hid_activation_ = nn.ReLU
    # if use_orthogonal_init:
    #     print("------use_orthogonal_init------")
    #     orthogonal_init(self.l1)
    #     orthogonal_init(self.l2)
    #     orthogonal_init(self.mean_layer, gain=0.01)
    #     orthogonal_init(self.log_std_layer, gain=0.01)
    layers = []
    input_layer = nn.Linear(input_dim, hidden_shape[0])
    output_layer = nn.Linear(hidden_shape[-1], output_dim)
    hidden_layers = []
    for l in range(len(hidden_shape) - 1):
        hidden_layers.append(nn.Linear(hidden_shape[l], hidden_shape[l + 1]))

    layers_ = [input_layer] + hidden_layers + [output_layer]

    for l in range(len(layers_)):
        layers.append(layers_[l])
        if l < len(layers_) - 1:
            layers.append(hid_activation_())
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_shape[l]))
    layers.append(output_activation)
    # layer_shape = [input_dim] + list(hidden_shape) + [output_dim]  #[input,hid,hid,out]
    # for j in range(len(layer_shape) - 1):
    #     activation_func = self.hid_activation_ if j < len(
    #         layer_shape) - 2 else output_activation
    #     if use_layer_norm:
    #         layers += [
    #             nn.Linear(layer_shape[j], layer_shape[j + 1]),
    #              activation_func, nn.BatchNorm1d(layer_shape[j + 1])
    #         ]
    #     if use_batchnorm:
    #         layers += [
    #             nn.Linear(layer_shape[j], layer_shape[j + 1]),
    #             nn.LayerNorm([layer_shape[j + 1]]), activation_func
    #         ]
    #     else:
    #         layers += [
    #             nn.Linear(layer_shape[j], layer_shape[j + 1]), activation_func
    #         ]

    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hid_activation="relu",
    ):
        super().__init__()
        if hid_activation == "relu":
            self.hid_activation_ = nn.ReLU
        elif hid_activation == "leaky_relu":
            self.hid_activation_ = nn.LeakyReLU
        elif hid_activation == "tanh":
            self.hid_activation_ = nn.Tanh
        elif hid_activation == "softsign":
            self.hid_activation_ = nn.Softsign
        elif hid_activation == "tanhshrink":
            self.hid_activation_ = nn.Tanhshrink
        elif hid_activation == "elu":
            self.hid_activation_ = nn.ELU
        else:
            logging.error("activate_func error")
            self.hid_activation_ = nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=out_features,
                      bias=True),  # V_i, c_i
            self.hid_activation_(),
            nn.Linear(in_features=out_features,
                      out_features=out_features,
                      bias=True),  # U_i, a_i
        )

    def forward(self, x):
        return self.net(x)


class SumTree:
    write = 0

    def __init__(self, capacity):
        capacity = int(capacity)
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
        # self._propagate(idx, change)

    # get priority and sample
    def get(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

default_Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done"))


class Memory(object):
    def __init__(self, Transition: namedtuple = default_Transition):
        self.memory = []
        self.Transition = Transition

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return self.Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    def save(self, file):
        import pickle

        with open(file, "wb") as f:
            data = pickle.dump(self.memory, f)

    def load(self, file):
        import pickle

        with open(file, "rb") as f:
            self.memory = pickle.load(f)


class ReplayMemoryPER:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, device="cuda"):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device

    def _get_priority(self, error):
        return (abs(error) + self.e)**self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(list(data))
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities,
                             -self.beta)
        is_weight /= is_weight.max()
        batchs = np.array(batch).transpose()
        s = torch.from_numpy(np.vstack(batchs[0])).to(self.device)
        a = torch.from_numpy(np.vstack(list(batchs[1])).astype(np.int64)).to(
            self.device)
        r = torch.from_numpy(np.array(list(batchs[2]))).to(self.device)
        s_ = torch.from_numpy(np.vstack(batchs[3])).to(self.device)
        d = torch.from_numpy(np.array(list(batchs[4])).astype(np.int32)).to(
            self.device)
        return s, a, s_, r, d, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class ReplayMemory(object):
    def __init__(self, state_dim, action_dim, memory_size, device):
        self.max_size = int(memory_size)
        self.count = 0
        self.size = 0
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw

        self.count = (
            self.count + 1
        ) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1,
                        self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size,
                                 size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index],
                               dtype=torch.float).to(self.device)
        if self.action_dim > 1:
            # continous action space
            batch_a = torch.tensor(self.a[index],
                                   dtype=torch.float).to(self.device)
        else:
            # descrete action space
            batch_a = torch.tensor(self.a[index],
                                   dtype=torch.int64).to(self.device)

        batch_r = torch.tensor(self.r[index],
                               dtype=torch.float).to(self.device)
        batch_s_ = torch.tensor(self.s_[index],
                                dtype=torch.float).to(self.device)
        batch_dw = torch.tensor(self.dw[index],
                                dtype=torch.float).to(self.device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    def reset(self):
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, self.state_dim))
        self.a = np.zeros((self.max_size, self.action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, self.state_dim))
        self.dw = np.zeros((self.max_size, 1))


class ReplayBuffer_PPO:
    def __init__(self, batch_size, state_dim):
        self.s = np.zeros((batch_size, state_dim))
        self.a = np.zeros((batch_size, 1))
        self.a_logprob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))
        self.s_ = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0
        self.batch_size = batch_size

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done

        self.count += 1
        # if self.count >= self.batch_size:
        #     self.count = 0

    def numpy_to_tensor(self, device):
        s = torch.tensor(self.s, dtype=torch.float)
        # In discrete action space, 'a' needs to be torch.long
        a = torch.tensor(self.a, dtype=torch.long)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        s = s.to(device)
        a = a.to(device)
        s_ = s_.to(device)
        a_logprob = a_logprob.to(device)
        r = r.to(device)
        dw = dw.to(device)
        done = done.to(device)
        return s, a, a_logprob, r, s_, dw, done


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self,
                 shape,
                 finetune=False):  # shape:the dimension of input data
        self.n = 0 if not finetune else 1
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
        self.is_finetune = finetune

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape, finetune=False):
        self.running_ms = RunningMeanStd(shape=shape, finetune=finetune)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x.astype(np.float32)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.0
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(
        self,
        theta,
        mu=0.0,
        sigma=1.0,
        dt=1e-2,
        x0=None,
        size=1,
        sigma_min=None,
        n_steps_annealing=1000,
    ):
        super(OrnsteinUhlenbeckProcess,
              self).__init__(mu=mu,
                             sigma=sigma,
                             sigma_min=sigma_min,
                             n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = (self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt +
             self.current_sigma * np.sqrt(self.dt) *
             np.random.normal(size=self.size))
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    @:param distance_metric: The distance metric function
    @:param margin: (float) The margin distance
    @:param size_average: (bool) Whether to get averaged loss

    Input example of forward function:
        rep_anchor: [[0.2, -0.1, ..., 0.6], [0.2, -0.1, ..., 0.6], ..., [0.2, -0.1, ..., 0.6]]
        rep_candidate: [[0.3, 0.1, ...m -0.3], [-0.8, 1.2, ..., 0.7], ..., [-0.9, 0.1, ..., 0.4]]
        label: [0, 1, ..., 1]

    Return example of forward function:
        0.015 (averged)
        2.672 (sum)
    """
    def __init__(
        self,
        distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
        margin: float = 0.5,
        size_average: bool = True,
    ):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average

    def forward(self, rep_anchor, rep_candidate, label: Tensor):
        # rep_anchor: [batch_size, hidden_dim] denotes the representations of anchors
        # rep_candidate: [batch_size, hidden_dim] denotes the representations of positive / negative
        # label: [batch_size, hidden_dim] denotes the label of each anchor - candidate pair
        label = torch.nn.functional.relu(label, inplace=True)
        distances = self.distance_metric(rep_anchor, rep_candidate)
        losses = 0.5 * (
            label.float() * distances.pow(2) +
            (1 - label).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()


class ContrastiveLoss_2(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0, metric=""):
        super(ContrastiveLoss_2, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = torch.nn.functional.relu(label, inplace=True)

        # distance = SiameseDistanceMetric.COSINE_DISTANCE(output1, output2)
        distance = SiameseDistanceMetric.EUCLIDEAN(output1, output2)
        loss_contrastive = torch.mean(
            (label) * torch.pow(distance, 2)  # calmp夹断用法
            + (1 - label) *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss_contrastive
