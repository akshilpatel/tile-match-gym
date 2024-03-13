from gymnasium.wrappers import ObservationWrapper
from collections import OrderedDict
from gymnasium.spaces import Box

import gymnasium as gym
import numpy as np

class TMGOneHotWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.num_colours = env.num_colours
        self.num_colour_specials = env.num_colour_specials
        self.num_colourless_specials = env.num_colourless_specials
        self.num_rows = env.num_rows
        self.num_cols = env.num_cols
        self.board_obs_space = Box(
            low=0, high=1, 
            dtype=np.int32, 
            shape = (1 + self.num_colours + self.num_colour_specials + self.num_colourless_specials, self.num_rows, self.num_cols))

        self.observation_space = gym.spaces.Dict({
            "board": self.board_obs_space,
            "num_moves_left": env._moves_left_observation_space
        })

    def _get_obs(self) -> dict:
        board = self.board.board
        num_moves_left = self.num_moves - self.timer
        ohe_board = self._one_hot_encode_board(board)

        return OrderedDict([("board", ohe_board), ("num_moves_left", num_moves_left)])

    def _one_hot_encode_board(self, board: np.ndarray) -> np.ndarray:
        ohe_board = np.zeros((self.num_colours + 1 + self.num_colour_specials + self.num_colourless_specials, self.num_rows, self.num_cols), dtype=np.int32)
        tile_colours = board[0] - 1
        tile_types = board[1] - 1 + self.num_colourless_specials # tile_types start at 1, so we need to subtract 1 to get the index
        rows, cols = np.indices(tile_colours.shape)
        ohe_board[tile_colours.flatten(), rows.flatten(), cols.flatten()] = 1
        ohe_board[tile_types.flatten(), rows.flatten(), cols.flatten()] = 1

        return ohe_board