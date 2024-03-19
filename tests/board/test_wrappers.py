from tile_match_gym.tile_match_env import TileMatchEnv
from tile_match_gym.wrappers import OneHotWrapper, MinOneHotWrapper
import numpy as np

def test_one_hot_wrapper():
    env = TileMatchEnv(4, 3, 5, 10, [], [], seed=1)
    env = OneHotWrapper(env)
    assert env.observation_space["board"].shape == (6, 4, 3)
    assert env.observation_space["num_moves_left"].n == 11

    obs, _ = env.reset()

    assert np.array_equal(env.unwrapped.board.board, 

    


    env = TileMatchEnv(4, 3, 5, 10, ["bomb", []],)