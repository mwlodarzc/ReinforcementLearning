import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('Blackjack-v1', natural=False, sab=False)
n_observations = env.observation_space  # type: ignore
n_actions = env.action_space.n  # type: ignore
Q_table = np.zeros(
    (n_observations[0].n, n_observations[1].n, n_observations[2].n, n_actions))  # type: ignore
state, info = env.reset()

n_episodes = 10_000
max_iter_episode = 100
exploration_border = 1
exploration_decreasing_decay = 0.001
min_exploration_border = 0.01
gamma = 0.99
lr = 1e-3
total_rewards = list()

for episode in range(n_episodes):
    current_state, info = env.reset()
    done = False
    episode_reward = 0

    for _ in range(max_iter_episode):

        if np.random.uniform(0, 1) < exploration_border:
            action = env.action_space.sample()
        else:
            action = np.argmax(
                Q_table[current_state[0], current_state[1], int(current_state[2]), :])
        next_state, reward, done, *_ = env.step(action)
        Q_table[current_state[0], current_state[1], int(current_state[2]), action] = (
            1-lr) * Q_table[current_state[0], current_state[1], int(current_state[2]), action] + lr*(reward + gamma * max(Q_table[next_state[0], next_state[1], int(next_state[2]), :]))
        # print(Q_table[current_state[0], current_state[1],
        #   current_state[2], action])
        episode_reward += reward
        if done:
            break
        current_state = next_state
    exploration_border = max(min_exploration_border,
                             np.exp(-exploration_decreasing_decay*episode))
    total_rewards.append(episode_reward)

print("Mean reward per thousand episodes")
n_samples = 1000
for i in range(n_samples):
    print(
        f"{(i+1)*int(n_episodes/n_samples)} : mean espiode reward: {np.mean(total_rewards[int(n_episodes/n_samples)*i:int(n_episodes/n_samples)*(i+1)])}")


plt.figure('Rewards')
plt.plot([np.mean(total_rewards[int(n_episodes/n_samples) *
         i:int(n_episodes/n_samples)*(i+1)]) for i in range(n_samples)])
plt.xlabel('Samples')
plt.ylabel('Rewards')
plt.show()
