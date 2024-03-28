from tile_match_gym.tile_match_env import TileMatchEnv
from tile_match_gym.wrappers import OneHotWrapper
import numpy as np

def test_one_hot_wrapper():
    env = TileMatchEnv(4, 3, 5, 10, [], [], seed=1)
    env = OneHotWrapper(env)
    assert env.observation_space["board"].shape == (5, 4, 3)
    assert env.observation_space["num_moves_left"].n == 11

    obs, info = env.reset()
    # Check colours
    assert np.array_equal(obs["board"][:, 0, 0], np.array([0, 0, 1, 0, 0], dtype=np.float32)), obs["board"][:, 0, 0]
    assert np.array_equal(obs["board"][:, 0, 1], np.array([1, 0, 0, 0, 0], dtype=np.float32)), obs["board"][:, 0, 1]
    assert obs["num_moves_left"] == 10


    env = TileMatchEnv(5, 5, 3, 10, [], ["bomb"], seed=2)
    env = OneHotWrapper(env)
    obs, info = env.reset()

    # print(env.unwrapped.board.board)
    assert np.array_equal(obs["board"][:, 2, 2], np.array([1, 0, 0, 0], dtype=np.float32)), obs["board"][:, 2, 2]

    # Make a bomb
    obs, *_ = env.step(33)
    # print(env.unwrapped.board.board)
    assert obs["board"].shape == (4, 5, 5)
    assert obs["num_moves_left"] == 9
    assert np.array_equal(obs["board"][:, 3, 2], np.array([1, 0, 0, 1], dtype=np.float32)), (obs["board"][:, 3, 2], env.unwrapped.board.board[:, 3,2])

    env = TileMatchEnv(5, 5, 2, 12, ["cookie"], ["vertical_laser"], seed=2)
    env = OneHotWrapper(env)
    obs, info = env.reset()
    assert obs["board"].shape == (4, 5, 5)
    assert obs["num_moves_left"] == 12
    # Make a cookie.
    obs, *_ = env.step(2)
    assert np.array_equal(obs["board"][:, 1, 2], np.array([0, 0, 1, 0], dtype=np.float32)), (obs["board"][:, 3, 2], env.unwrapped.board.board[:, 3,2])
    assert obs["board"].shape == (4, 5, 5)
    assert obs["num_moves_left"] == 11

# def test_timing():
#     import time
#     env = TileMatchEnv(30, 30, 12, 10, [], [], seed=1)
#     env = OneHotWrapper(env)
#     start = time.time()
#     # run for 1000 steps
#     obs, info = env.reset()
#     for _ in range(100):
#         action = env.action_space.sample()
#         next_obs, _, done, _, _ = env.step(action)
#         obs = next_obs
#         if done:
#             obs, info = env.reset()
#     print(f"Time taken for 1000 steps: {time.time() - start} seconds")

#     assert False

