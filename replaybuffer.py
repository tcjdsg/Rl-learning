import torch
import numpy as np
import copy

class Memory:
    def __init__(self):
        self.s = []
        self.v = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.dw = []





    def sample(self):
        return torch.cat(self.s,dim=0), torch.cat(self.a,dim=0),  torch.cat(self.a_logprob,dim=0),torch.cat(self.v,dim=0),torch.cat(self.r,dim=0), torch.cat(self.dw,dim=0)


    def push(self, state, vals, action, probs,  reward, done):
        self.s.append(torch.unsqueeze(state,0))
        self.a.append(torch.unsqueeze(torch.tensor(action),0))
        self.a_logprob.append(torch.unsqueeze(torch.tensor(probs),0))
        self.v.append(torch.unsqueeze(torch.tensor(vals),0))
        self.r.append(torch.unsqueeze(reward,0))
        self.dw.append(torch.unsqueeze(torch.tensor(done),0))


    def clear(self):
        self.s = []
        self.v = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.dw = []


class ReplayBuffer:
    def __init__(self, args):
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.use_adv_norm = args.use_adv_norm
        self.input_dim = args.input_dim


        self.batch_size = args.batch_size
        self.jzjN = args.n_j
        self.taskN = args.n_m
        self.episode_limit = self.jzjN * self.taskN
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'s': torch.zeros([self.batch_size, self.input_dim,self.jzjN,self.taskN]),
                       'v': torch.zeros([self.batch_size, 1]),
                       'a': torch.zeros([self.batch_size, 1]),
                       'a_logprob': torch.zeros([self.batch_size, 1]),
                       'r': torch.zeros([self.batch_size, 1]),
                       'dw': torch.ones([self.batch_size, 1]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': torch.zeros([self.batch_size, 1])
                       }

        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, s, v, a, a_logprob, r, dw):
        self.buffer['s'][self.episode_num] = s
        self.buffer['v'][self.episode_num] = v
        self.buffer['a'][self.episode_num] = a
        self.buffer['a_logprob'][self.episode_num] = a_logprob
        self.buffer['r'][self.episode_num] = r
        self.buffer['dw'][self.episode_num]= dw

        self.buffer['active'][self.episode_num] = 1.0

    def store_last_value(self, episode_step, v):
        self.buffer['v'][self.episode_num] = v
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
        adv = torch.zeros_like(r)  # adv.shape=(batch_size,max_episode_len)
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
                adv_copy[active == 0] = torch.nan  # 忽略掉active=0的那些adv
                adv = ((adv - torch.nanmean(adv_copy)) / (torch.std(adv_copy) + 1e-5))
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