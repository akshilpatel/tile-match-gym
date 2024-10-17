from sb3_contrib import QRDQN
import gymnasium as gym

import tile_match_gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env


from tile_match_gym.wrappers import OneHotWrapper, ProportionRewardWrapper
import numpy as np
env_kwargs = dict(num_rows=3, num_cols=3, num_colours=3, num_moves=10, colour_specials=[], colourless_specials=[], render_mode=None)
def run_qrdqn():

    def make_env():
        env = gym.make("TileMatch-v0", **env_kwargs)
        env = ProportionRewardWrapper(env)
        env = OneHotWrapper(env)
        return env

    eval_env = make_env()
    tsps = 500_000
    eval_freq = tsps // 50
    # Create the evaluation callback
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='./logs/',
                                 log_path='./logs/',
                                 eval_freq=eval_freq,
                                 deterministic=True,
                                 render=False, n_eval_episodes=30)

    env = make_env()
    print(env.action_space.n)
    policy_kwargs = dict(n_quantiles=75)
    model = QRDQN("MultiInputPolicy",  env, policy_kwargs=policy_kwargs, batch_size=512, train_freq=4, verbose=1)

    model_path = f"qrdqn_{env_kwargs['num_rows']}_{env_kwargs['num_cols']}_{env_kwargs['num_colours']}_{env_kwargs['num_moves']}_" + str(tsps)
    model.learn(total_timesteps=tsps, progress_bar=True, log_interval=eval_freq // env_kwargs["num_moves"], callback=[eval_callback])
    
    # model.save(model_path)
    total_reward = 0
    env_kwargs["render_mode"] = "human"
    env = gym.make("TileMatch-v0", **env_kwargs)
    env = ProportionRewardWrapper(env)
    env = OneHotWrapper(env)
    for i in range(10):
        obs, info = env.reset()
        env.render()
        print(f"obs = {env.board.board}")

        while True:

            action, _state = model.predict(obs, deterministic=True)
           
            next_obs, reward, done, truncated, next_info = env.step(action)
            if action in info["effective_actions"]:
                print(obs, action, reward, next_obs, done)
            total_reward += reward
            env.render()
            if done:
                print(total_reward)
                total_reward = 0
                break
            else:
                obs = next_obs
                info = next_info

        # print(f"total reward = {total_reward}")
if __name__ == '__main__':
    # run_qrdqn()
    from matplotlib import pyplot as plt
    x = np.load("logs/evaluations.npz")
    epi_rewards = x["results"]
    reward_means = np.mean(epi_rewards, axis=1)
    reward_mins = np.min(epi_rewards, axis=1)
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    reward_means_smoothed = np.convolve(reward_means, kernel, mode='same')
    reward_medians = np.median(epi_rewards, axis=1)
    reward_std = np.std(epi_rewards, axis=1)
    plt.plot(reward_medians, label="reward_medians")
    plt.plot(reward_means, label="reward_means")
    plt.plot(reward_means_smoothed, label="reward_means_smoothed")
    plt.plot(reward_mins, label="reward_mins")
    plt.legend()
    plt.show()
