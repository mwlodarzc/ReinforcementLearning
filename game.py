import gym
from PIL import Image
# import cv2
# cv2.namedWindow("The Arcade Learning Environment", cv2.WINDOW_AUTOSIZE)
# scale_percent = 500

# from gym.utils.play import play, PlayPlotl
env = gym.make('ALE/DonkeyKong-v5', render_mode='human')
env.action_space.seed(42)
observation, info = env.reset(seed=42)

# size = tuple(map(lambda x: int(x*scale_percent/100),
#  env.observation_space.shape[:-1]))


for _ in range(100000):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample())
    # resized = cv2.resize(observation, size, interpolation=cv2.INTER_AREA)
    # cv2.imshow('image', resized)
    # cv2.waitKey(10)
    if terminated or truncated:
        observation, info = env.reset()
env.close()

# def callback(obs_t, obs_tp1, action, rew, done, info,_):
#     return [rew,]
# plotter = PlayPlot(callback, 30 * 5, ["reward"])
# env = gym.make('ALE/DonkeyKong-v5', render_mode='rgb_array')
# play(env, callback=plotter.callback)
