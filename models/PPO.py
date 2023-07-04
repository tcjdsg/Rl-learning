from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Categorical

from Params import configs
from models.actor_critic import ActorCritic
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler


class PPO:
    def __init__(self,
                 n_j,
                 n_m,
                 input_dim,
                 kernels,
                 hidden_dims,
                 hidden_dim,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 out_priority_dim,
                 device
                 ):
        self.lr = configs.lr
        self.gamma = configs.gamma
        self.eps_clip = configs.eps_clip
        self.k_epochs = configs.k_epochs
        self.set_adam_eps = configs.set_adam_eps
        self.batch_size = configs.batch_size
        self.mini_batch_size = configs.mini_batch_size
        self.entropy_coef = configs.entloss_coef
        self.use_grad_clip = configs.use_grad_clip
        self.use_lr_decay = configs.use_lr_decay
        self.max_updates=configs.max_updates

        self.policy = ActorCritic(n_j=n_j,
                                  n_m=n_m,
                                  input_dim=input_dim,
                                  kernel = kernels,
                                  hidden_dims=hidden_dims,
                                  hidden_dim =hidden_dim,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  out_priority_dim=out_priority_dim,
                                  device=device)

        self.policy_old = deepcopy(self.policy)

        '''self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_j) + '_' + str(n_m) + '_' + str(1) + '_' + str(99))))'''

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy_old.load_state_dict(self.policy.state_dict())
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)

        self.V_loss_2 = nn.MSELoss()

    def choose_action(self, s, evaluate=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
            logit = self.policy.actor(s)
            if evaluate:
                a = torch.argmax(logit)
                return a.item(), None
            else:
                dist = Categorical(logits=logit)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
                return a.item(), a_logprob.item()

    def get_value(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
            value = self.policy.critic(s)
            return value.item()

    def update(self, replay_buffer,total_steps):
        batch = replay_buffer.get_training_data()  # Get training data
        for _ in range(self.k_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):

                pi_now = self.policy.actor(
                    batch['s'][index])  # logits_now.shape=(mini_batch_size, max_episode_len, action_dim)
                values_now = self.policy.critic(batch['s'][index]).squeeze(
                    -1)  # values_now.shape=(mini_batch_size, max_episode_len)
                dist_now = Categorical(logits=pi_now)
                dist_entropy = dist_now.entropy()  # shape(mini_batch_size, max_episode_len)
                a_logprob_now = dist_now.log_prob(batch['a'][index])  # shape(mini_batch_size, max_episode_len)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - batch['a_logprob'][index])  # shape(mini_batch_size, max_episode_len)
                # actor loss
                surr1 = ratios * batch['adv'][index]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch['adv'][index]
                actor_loss = -torch.min(surr1,
                                        surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size, max_episode_len)
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                # critic_loss
                critic_loss = (values_now - batch['v_target'][index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                # Update
                self.optimizer.zero_grad()

                vloss_coef = configs.vloss_coef
                ploss_coef = configs.ploss_coef

                loss = actor_loss * ploss_coef + critic_loss * vloss_coef
                loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
            if self.use_lr_decay:  # Trick 6:learning rate Decay
                self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
            lr_now = 0.9 * self.lr * (1 - total_steps / self.max_updates) + 0.1 * self.lr
            for p in self.optimizer.param_groups:
                p['lr'] = lr_now

    def save_model(self, env_name, number, seed, total_steps):
            torch.save(self.policy.state_dict(),
                       "./model/PPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed,
                                                                                        int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
            self.policy.load_state_dict(torch.load(
                "./model/PPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))


