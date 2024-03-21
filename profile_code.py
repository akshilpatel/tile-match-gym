import cProfile
from tile_match_gym.tile_match_env import TileMatchEnv
from tile_match_gym.wrappers import OneHotWrapper
import numpy as np

def main():
    env = TileMatchEnv(30, 30, 12, 10, [], [], seed=1)
    env = OneHotWrapper(env)
    # run for 1000 steps
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        next_obs, _, done, _, _ = env.step(action)
        obs = next_obs
        if done:
            obs, info = env.reset()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.print_stats()

