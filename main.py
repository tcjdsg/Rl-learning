from JZJenv.FixedMess import FixedMes
from agent_utils import eval_actions
from agent_utils import select_action
from models.PPO import PPO
from copy import deepcopy
import torch
import time
from utils import *
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Params import configs
# from validation import validate

from replaybuffer import ReplayBuffer, Memory

device = torch.device(configs.device)

from JzjEnv import JZJ

class Runner:
    def __init__(self, configs,number,seed):
        self.args = configs

        self.seed = seed
        self.number = number
        self.episodeNum =0

        self.lamda = configs.lamda
        self.use_adv_norm = configs.use_adv_norm
        # Create env
        self.env = JZJ(configs.n_j, configs.n_m)

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # self.env.seed(seed)

  # Maximum number of steps per episode
        print("envJZJ={}".format(configs.n_j))
        print("episode_limit={}".format(configs.n_j*configs.n_m))
        self.replay_buffer = Memory()

        self.episode_limit = configs.n_j*configs.n_m

        self.agent = PPO(
              n_j=configs.n_j,
              n_m=configs.n_m,
              input_dim=configs.input_dim,
              hidden_dims=[32,64,32],
              kernels=[1,3,3],
              hidden_dim=configs.hidden_dim,

              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic,
              out_priority_dim = configs.action_dim,
              device = device,
              )

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/PPO_discrete/env_JZJ{}_tasksN{}_number_{}_seed_{}'.format(configs.n_j, configs.n_m, number, seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        if self.args.use_state_norm:
            print("------use state normalization------")
            self.state_norm = Normalization(shape=configs.state_dim)  # Trick 2:state normalization
        if self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num =-1  # Record the number of evaluations
        while self.episodeNum < self.args.max_updates:
            if self.episodeNum // self.args.evaluate_freq > evaluate_num:
                print("Evaluate the policy every 'evaluate_freq' steps")
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1
                
            print("------------one episode-------------")
            episode_reward = self.run_episode()  # Run an episode
            self.episodeNum+=1

            loss, v_loss =self.agent.update(self.replay_buffer)  # Training
            self.replay_buffer.clear()

            print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
                self.episodeNum + 1, episode_reward, v_loss))

        self.evaluate_policy()
        self.env.close()
    def run_episode(self, ):

        episode_reward = 0
        s = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        # self.agent.reset_rnn_hidden()
        for episode_step in range(configs.n_j * configs.n_m):
            if self.args.use_state_norm:
                s = self.state_norm(s)
            a, a_logprob = self.agent.choose_action(s, evaluate=False)
            v = self.agent.get_value(s)
            s_, r, done, _ = self.env.step(a)
            episode_reward += r
            if done and episode_step + 1 != self.episode_limit:
                dw = True
            else:
                dw = False
            if self.args.use_reward_scaling:
                r = self.reward_scaling(r)
            # Store the transition
            self.replay_buffer.push(s, v, a, a_logprob, torch.tensor(r), done)
            s = s_
            if done:
                break
        # An episode is over, store v in the last step
        if self.args.use_state_norm:
            s = self.state_norm(s)
        v = self.agent.get_value(s)

        return episode_reward

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, done = 0, False
            s = self.env.reset()
            # self.agent.reset_rnn_hidden()
            while not done:
                if self.args.use_state_norm:
                    s = self.state_norm(s, update=False)
                a, a_logprob = self.agent.choose_action(s, evaluate=True)
                s_, r, done, _ = self.env.step(a)
                episode_reward += r
                s = s_
            evaluate_reward += episode_reward
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(configs.n_j), evaluate_reward,
                               global_step=self.total_steps)
        # Save the rewards and models
        np.save('./data_train/PPO_env_{}_number_{}_seed_{}.npy'.format(configs.n_j, self.number, self.seed),
                np.array(self.evaluate_rewards))

if __name__ == '__main__':
    total1 = time.time()
    JZJ=Runner( configs, configs.n_j, configs.n_m)
    JZJ.run()
    total2 = time.time()
    # print(total2 - total1)