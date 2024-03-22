import cProfile
from tile_match_gym.tile_match_env import TileMatchEnv
from tile_match_gym.wrappers import OneHotWrapper
import numpy as np

def main():
    env = TileMatchEnv(30, 30, 12, 10, ["vertical_laser", "horizontal_laser", "bomb"], ["cookie"], seed=1)
    env = OneHotWrapper(env)
    # run for 1000 steps
    obs, info = env.reset()
    for _ in range(200):
        action = np.random.choice(info["effective_actions"])
        obs, _, done, _, info = env.step(action)
        if done:
            obs, info = env.reset()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.print_stats()

