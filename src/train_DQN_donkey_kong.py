import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_grayscale, to_pil_image

from DQN import DQN, ReplayMemory

PATH = './models/DonkeyKongDQN'

def eps_treshold(t: int):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY)

def transform(data: torch.Tensor) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((120,120))
    return to_tensor(resize(to_grayscale(to_pil_image(data)))).to(device)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make('ALE/DonkeyKong-v5', render_mode='rgb_array')
    state, info = env.reset()
    env_size = np.prod(np.shape(transform(state))),env.action_space.n

    replay_memory = ReplayMemory(10000)
    policy_net = DQN(*env_size).to(device)
    target_net = DQN(*env_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    BATCH_SIZE = 128
    LEARNING_PERIOD = 30
    GAMMA = 0.99  
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    criterion = nn.SmoothL1Loss()

    num_episodes = 600 if torch.cuda.is_available() else 50
    for episode_idx in range(num_episodes):
        state, info = env.reset()
        for episode_duration in count():
            print(episode_idx,episode_duration)
            # choose an action
            steps_done = 0
            if random.random() > eps_treshold(steps_done):
                with torch.no_grad():
                    action = policy_net(transform(state)).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
            steps_done += 1

            # observe the state and push to memory
            observation, reward, terminated, truncated, _ = env.step(action.item())            
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            next_state = (None if terminated else transform(observation))
            replay_memory.push(state, action, reward, next_state)
            # next state transition
            state = next_state
            # perform optimization (on the policy network)
            if len(replay_memory) >= BATCH_SIZE and not episode_duration % LEARNING_PERIOD:
                transitions = replay_memory.sample(BATCH_SIZE)
                for sample in transitions:
                    policy_Q = policy_net(transform(sample.state))
                    target_Q = target_net(transform(sample.state))
                    td_target = sample.reward + GAMMA * target_Q
                    loss = criterion(policy_Q,td_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # update target and policy networks
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break
    
    torch.save(policy_net.state_dict(), PATH)
