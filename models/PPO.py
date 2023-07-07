from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Categorical

from Params import configs
from models.actor_critic import ActorCritic
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import torch.nn.functional as F

from replaybuffer import Memory


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
        self.lamda = configs.lamda
        self.use_adv_norm = configs.use_adv_norm
        self.epsilon =configs.eps_clip
        self.K_epochs = configs.k_epochs
        self.device = device



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
        ss = torch.unsqueeze(s, 0)
        with torch.no_grad():
            p = self.policy.actor(ss)
            if evaluate:
                a = torch.argmax(p)
                return a.item(), None
            else:
                dist = Categorical(p)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
                return a.item(), a_logprob.item()

    def get_value(self, s):
        with torch.no_grad():
            ss = s.unsqueeze(0)
            value = self.policy.critic(ss)
            return value.item()

    def eval_actions(self,p, actions):
        softmax_dist = Categorical(p)
        ret = softmax_dist.log_prob(actions).reshape(-1)
        entropy = softmax_dist.entropy().mean()
        return ret, entropy

    def update(self, memory):
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr= memory.sample()
        values = vals_arr[:]

        # 计算GAE
        advantage = torch.zeros(len(reward_arr), dtype=torch.float32)
        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                discount *= self.gamma *  self.lamda
            advantage[t] = a_t
        advantage = advantage.to(self.device)

        values = values.to(self.device)

        states =  state_arr.to(self.device)
        old_probs = old_prob_arr.to(self.device)
        actions = action_arr.to(self.device)

        # 计算新的策略分布

        critic_value = torch.squeeze(self.policy.critic(states))

        # self.eval_actions( p, actions):
        p=self.policy.actor(states)
        dist_now = Categorical(p)
        new_probs = dist_now.log_prob(actions).reshape(-1)
        prob_ratio = new_probs.exp() / old_probs.exp()

        entropy = dist_now.entropy().view(-1, 1)
        entropy_loss = entropy.mean()

            # actor_loss
        weighted_probs = advantage * prob_ratio
        weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.epsilon,
                                                 1 + self.epsilon) * advantage

            # surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
            # surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
            # actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)

        actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

            # critic_loss
        returns = advantage + values
        critic_loss = (returns - critic_value) ** 2
        critic_loss = critic_loss.mean()

            # 更新

        total_loss = actor_loss + 0.5 * critic_loss - entropy_loss * self.entropy_coef
        self.loss = total_loss

        total_loss.backward()

        if self.use_grad_clip:  # Trick 7: Gradient clip
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return total_loss.item(),critic_loss.item()


        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.policy.critic(s)
            vs_ = self.policy.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = Categorical(probs=self.policy.actor(s[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)


                v_s = self.policy.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                loss = actor_loss + critic_loss * 0.5

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


