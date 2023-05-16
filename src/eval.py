import numpy as np

import gym
import torch
from DQN import DQN

import train_DQN_donkey_kong as game

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('ALE/DonkeyKong-v5', render_mode='rgb_array')
    state, info = env.reset()
    env_size = np.prod(np.shape(game.transform(state))),env.action_space.n
    Q = DQN(*env_size).to(device)
    Q.load_state_dict(torch.load(game.PATH))
    
    total_run_time, total_reward = [], []
    for _ in range(20):
        run_time = reward = 0
        done = False
        while not done:
            action = Q(game.transform(state)).max(1)[1].view(1, 1)
            observation, reward, terminated, truncated, _ = env.step(action.item())            
            reward += reward
            done = terminated or truncated
            if done:
                break
            run_time += 1
        total_run_time.append(run_time)
        total_reward.append(reward)
    print(zip(total_run_time,total_reward))
            
    
