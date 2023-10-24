from Params import configs
from JzjEnv import JZJ, Normalization
import torch

from models.PPO import PPO

device = torch.device(configs.device)

def validate( ):
    env = JZJ(configs.n_j, configs.n_m)

    agent = PPO(
        n_j=configs.n_j,
        n_m=configs.n_m,
        input_dim=configs.input_dim,
        hidden_dims=[32, 64, 32],
        kernels=[1, 3, 3],
        hidden_dim=configs.hidden_dim,

        num_mlp_layers_actor=configs.num_mlp_layers_actor,
        hidden_dim_actor=configs.hidden_dim_actor,
        num_mlp_layers_critic=configs.num_mlp_layers_critic,
        hidden_dim_critic=configs.hidden_dim_critic,
        out_priority_dim=configs.action_dim,
        device=device,
    )

    path = './save/{}.pth'.format(str(configs.n_j) + '_' + str(configs.n_m) + '_')
    agent.policy.load_state_dict(torch.load(path))
    if configs.use_state_norm:
        print("------use state normalization------")
        state_norm = Normalization(shape=configs.state_dim)  # Trick 2:state normalization


    evaluate_reward = 0
    for _ in range(configs.evaluate_times):
        episode_reward, done = 0, False
        s = env.reset()
        # self.agent.reset_rnn_hidden()
        while not done:
            if configs.use_state_norm:
                s = state_norm(s, update=False)
            a, a_logprob = agent.choose_action(s, evaluate=True)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
    evaluate_reward = evaluate_reward / configs.evaluate_times
    print(0-evaluate_reward)




if __name__ == '__main__':


    validate()


    # print(min(result))

