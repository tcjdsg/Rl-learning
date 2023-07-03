import torch
import numpy as np
import copy

# parser.add_argument('--input_dim', type=int, default=2, help='number of dimension of raw node features')
# parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dim of MLP in fea extract GNN')
# parser.add_argument('--num_mlp_layers_feature_extract', type=int, default=2, help='No. of layers of MLP in fea extract GNN')
# parser.add_argument('--num_mlp_layers_actor', type=int, default=2, help='No. of layers in actor MLP')
# parser.add_argument('--hidden_dim_actor', type=int, default=32, help='hidden dim of MLP in actor')
# parser.add_argument('--num_mlp_layers_critic', type=int, default=2, help='No. of layers in critic MLP')
# parser.add_argument('--hidden_dim_critic', type=int, default=32, help='hidden dim of MLP in critic')
# # args for PPO
# parser.add_argument('--num_envs', type=int, default=4, help='No. of envs for training')
# parser.add_argument('--max_updates', type=int, default=10000, help='No. of episodes of each env for training')
# parser.add_argument('--lr', type=float, default=2e-5, help='lr')
# parser.add_argument('--decayflag', type=bool, default=False, help='lr decayflag')
# parser.add_argument('--decay_step_size', type=int, default=2000, help='decay_step_size')
# parser.add_argument('--decay_ratio', type=float, default=0.9, help='decay_ratio, e.g. 0.9, 0.95')
# parser.add_argument('--gamma', type=float, default=1, help='discount factor')
# parser.add_argument('--k_epochs', type=int, default=1, help='update policy for K epochs')
# parser.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
# parser.add_argument('--vloss_coef', type=float, default=1, help='critic loss coefficient')
# parser.add_argument('--ploss_coef', type=float, default=2, help='policy loss coefficient')
# parser.add_argument('--entloss_coef', type=float, default=0.01, help='entropy loss coefficient')
# parser.add_argument('--set_adam_eps', type=bool, default=True, help='')
# parser.add_argument('--batch_size', type=int, default=64, help='batch')
# parser.add_argument('--mini_batch_size', type=int, default=32, help='mini_batch_size')
# parser.add_argument('--use_grad_clip', type=bool, default=True, help='use_grad_clip')

class ReplayBuffer:
    def __init__(self, args):
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.use_adv_norm = args.use_adv_norm
        self.input_dim = args.input_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.max_updates
        self.batch_size = args.batch_size
        self.jzjN = args.n_j
        self.taskN = args.n_m
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'s': np.zeros([self.batch_size, self.episode_limit, self.input_dim,self.jzjN,self.taskN]),
                       'v': np.zeros([self.batch_size, self.episode_limit + 1]),
                       'a': np.zeros([self.batch_size, self.episode_limit]),
                       'a_logprob': np.zeros([self.batch_size, self.episode_limit]),
                       'r': np.zeros([self.batch_size, self.episode_limit]),
                       'dw': np.ones([self.batch_size, self.episode_limit]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.batch_size, self.episode_limit])
                       }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, s, v, a, a_logprob, r, dw):
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v'][self.episode_num][episode_step] = v
        self.buffer['a'][self.episode_num][episode_step] = a
        self.buffer['a_logprob'][self.episode_num][episode_step] = a_logprob
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw

        self.buffer['active'][self.episode_num][episode_step] = 1.0

    def store_last_value(self, episode_step, v):
        self.buffer['v'][self.episode_num][episode_step] = v
        self.episode_num += 1
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def get_adv(self):
        # Calculate the advantage using GAE
        v = self.buffer['v'][:, :self.max_episode_len]
        v_next = self.buffer['v'][:, 1:self.max_episode_len + 1]
        r = self.buffer['r'][:, :self.max_episode_len]
        dw = self.buffer['dw'][:, :self.max_episode_len]
        active = self.buffer['active'][:, :self.max_episode_len]
        adv = np.zeros_like(r)  # adv.shape=(batch_size,max_episode_len)
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len)
            deltas = r + self.gamma * v_next * (1 - dw) - v
            for t in reversed(range(self.max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae  # gae.shape=(batch_size)
                adv[:, t] = gae
            v_target = adv + v  # v_target.shape(batch_size,max_episode_len)
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv_copy = copy.deepcopy(adv)
                adv_copy[active == 0] = np.nan  # 忽略掉active=0的那些adv
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
        return adv, v_target

    def get_training_data(self):
        adv, v_target = self.get_adv()
        batch = {'s': torch.tensor(self.buffer['s'][:, :self.max_episode_len], dtype=torch.float32),
                 'a': torch.tensor(self.buffer['a'][:, :self.max_episode_len], dtype=torch.long),  # 动作a的类型必须是long
                 'a_logprob': torch.tensor(self.buffer['a_logprob'][:, :self.max_episode_len], dtype=torch.float32),
                 'active': torch.tensor(self.buffer['active'][:, :self.max_episode_len], dtype=torch.float32),
                 'adv': torch.tensor(adv, dtype=torch.float32),
                 'v_target': torch.tensor(v_target, dtype=torch.float32)}

        return batch