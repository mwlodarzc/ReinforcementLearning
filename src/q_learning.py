import numpy as np
import gym

atari_actions = {
    0: 'NOOP',
    1: 'FIRE',
    2: 'UP',
    3: 'RIGHT',
    4: 'LEFT',
    5: 'DOWN',
    6: 'UPRIGHT',
    7: 'UPLEFT',
    8: 'DOWNRIGHT',
    9: 'DOWNLEFT',
    10: 'UPFIRE',
    11: 'RIGHTFIRE',
    12: 'LEFTFIRE',
    13: 'DOWNFIRE',
    14: 'UPRIGHTFIRE',
    15: 'UPLEFTFIRE',
    16: 'DOWNRIGHTFIRE',
    17: 'DOWNLEFTFIRE'
}
env = gym.make('ALE/DonkeyKong-v5', render_mode='rgb_array')
n_observations = np.product(env.observation_space.shape)
n_actions = env.action_space.n

print(env.observation_space.shape)
print(n_observations)


Q_table = np.zeros((n_observations, n_actions))
# print(Q_table.shape)
state, info = env.reset()
# print(np.ndarray.flatten(state).shape)

n_episodes = 10_000
max_iter_episode = 100
exploration_border = 1
exploration_decreasing_decay = 0.001
min_exploration_lambda = 0.01
gamma = 0.99
lr = 0.1
total_rewards_episode = list()


for e in range(n_episodes):
    current_state, info = env.reset()
    done = False

    total_episode_reward = 0

    for i in range(max_iter_episode):
        if np.random.uniform(0, 1) < exploration_border:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[np.ndarray.flatten(current_state), :])

        next_state, reward, done, *_ = env.step(action)
        print(lr*(reward *
              gamma * max(Q_table[np.ndarray.flatten(next_state), :])))
        Q_table[current_state, action] = (1-lr) * Q_table[current_state, action] + lr*(
            reward * gamma * max(Q_table[np.ndarray.flatten(next_state), :]))

        total_episode_reward += reward

        if done:
            break

        current_state = next_state

    exploration_border = max(min_exploration_lambda,
                             np.exp(-exploration_decreasing_decay*e))
    total_rewards_episode.append(total_episode_reward)

# if __name__ == '__main__':
#     env.action_space.seed(42)
#     observation, info = env.reset(seed=42)
#     for _ in range(100000):
#         move = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(move)
#         if terminated or truncated:
#             observation, info = env.reset()
#     env.close()
print("Mean reward per thousand episodes")
for i in range(10):
    print(
        f"{(i+1)*1000} : mean espiode reward: {np.mean(total_episode_reward[1000*i:1000*(i+1)])}")
