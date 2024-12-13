import sys, os

from pprint import pprint


class config:
    def __init__(self,
                 train_eps=500,
                 explore_eps=0,
                 step_limit=100,
                 eval_step_limit=10,
                 activate_func="relu",
                 use_state_norm=True,
                 use_lr_decay=False,
                 gradiant_clip=True,
                 use_reward_scaling=False):
        self.continual_learning = False
        self.train_eps = train_eps
        self.explore_eps = explore_eps
        self.step_limit = step_limit
        self.eval_step_limit = eval_step_limit
        self.activate_func = activate_func
        self.use_state_norm = use_state_norm
        self.use_lr_decay = use_lr_decay
        self.use_reward_scaling = use_reward_scaling
        self.gradiant_clip = gradiant_clip


class PPO_Config(config):
    def __init__(
            self,
            batch_size=512,
            mini_batch_size=64,
            gamma=0.99,
            ppo_update_time=8,
            actor_lr=5e-5,
            critic_lr=1e-4,  # 3e-4
            Adam_Optimizer_Epsilon=1e-7,
            gae_lambda=0.98,
            policy_clip=0.3,
            hidden_sizes=[512, 512],
            entropy_coef=0.02,
            activate_func="tanh",  #tanh
            use_orthogonal_init=False,
            min_decay_lr=5e-1,
            **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size  #64
        self.gamma = gamma
        self.ppo_update_time = ppo_update_time
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.Adam_Optimizer_Epsilon = Adam_Optimizer_Epsilon
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.hidden_sizes = hidden_sizes
        self.entropy_coef = entropy_coef
        self.activate_func = activate_func
        self.use_adv_norm = self.use_state_norm
        self.use_orthogonal_init = use_orthogonal_init
        self.min_decay_lr = min_decay_lr


