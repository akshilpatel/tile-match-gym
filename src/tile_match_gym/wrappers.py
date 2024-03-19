from gymnasium import ObservationWrapper, RewardWrapper
from collections import OrderedDict
from gymnasium.spaces import Box

import gymnasium as gym
import numpy as np

# First num_colours slices are for colour. Absence in these slices means colourless.
# Then the next 1 is for ordinary type. 
# Then the next num_colourless_special slices are for colourless specials. 
# Finally the last num_colour_specials slices are for colour specials.
class OneHotWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env.unwrapped
        self.num_colours = self.env.num_colours
        self.num_colour_specials = self.env.num_colour_specials
        self.num_colourless_specials = self.env.num_colourless_specials
        self.num_rows = self.env.num_rows
        self.num_cols = self.env.num_cols
        self.board_obs_space = Box(
            low=0, high=1, 
            dtype=np.int32, 
            shape = (2 + self.num_colours + self.num_colour_specials + self.num_colourless_specials, self.num_rows, self.num_cols))

        self.observation_space = gym.spaces.Dict({
            "board": self.board_obs_space, 
            "num_moves_left": self.env._moves_left_observation_space
        })

        self.colour_specials = self.env.colour_specials
        self.colourless_specials = self.env.colourless_specials

        self.type_to_idx = {
            "ordinary": 1,
            "empty": 0,
            "cookie": -1,
            
        }

    def observation(self, obs) -> dict:
        board = obs["board"]
        ohe_board = self._one_hot_encode_board(board)
        return OrderedDict([("board", ohe_board), ("num_moves_left", obs["num_moves_left"])])
    
    
    def _one_hot_encode_board(self, board: np.ndarray) -> np.ndarray:
        
        
        
        ohe_board = np.zeros((
            self.num_colours + 2 + self.num_colour_specials + self.num_colourless_specials, # +1 for colourless, +1 for ordinary type
            self.num_rows,
            self.num_cols), dtype=np.int32)
        
        tile_colours = board[0]
        tile_types = board[1] + self.num_colourless_specials 
        rows, cols = np.indices(tile_colours.shape)
        colour_ohe = np.zeros((1 + self.num_colours, self.num_rows, self.num_cols))
        type_ohe = np.zeros((1 + 2 + self.num_colour_specials, self.num_rows, self.num_cols)) # +1 for ordinary, +1 for empty
        
        colour_ohe[tile_colours.flatten(), rows.flatten(), cols.flatten()] = 1
        type_ohe[tile_types.flatten(), rows.flatten(), cols.flatten()] = 1

        # Remove empty slice in type_ohe
        type_ohe = np.concatenate([type_ohe[:self.num_colourless_specials], type_ohe[self.num_colourless_specials + 1:]], axis=0)
        
        # Concatenate colour and type ohe
        ohe_board = np.concatenate([colour_ohe, type_ohe], axis=0) # 1 + num_colours + num_colourless_specials + num_colour_specials.

        # Colourless
        # Colour 1
        # Colour 2...
        # Colourless special 1 
        # colourless special 2 
        # ordinary
        # colour special 1
        # colour special 2
        
        # ohe_board[1:]
        # ohe_board[:num_colours] and ohe_board[num_colours + 2:]
        return ohe_board
    
class MinOneHotWrapper(OneHotWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.board_obs_space = Box(
            low=0, high=1, 
            dtype=np.int32, 
            shape = (self.num_colours + self.num_colour_specials + self.num_colourless_specials, self.num_rows, self.num_cols))

        self.observation_space = gym.spaces.Dict({
            "board": self.board_obs_space, 
            "num_moves_left": self.env._moves_left_observation_space
        })
    def _one_hot_encode_board(self, board: np.ndarray) -> np.ndarray:
        ohe_board = super()._one_hot_encode_board(board)
        ordinary_type_idx = 1 + self.num_colours + self.num_colourless_specials
        if self.num_colour_specials > 0:
            ohe_board = np.concatenate([ohe_board[1:ordinary_type_idx], ohe_board[ordinary_type_idx + 1:]], axis=0) 
        else:
            ohe_board = ohe_board[1:ordinary_type_idx]
        return ohe_board

class ProportionRewardWrapper(RewardWrapper):
    def __init__(self, env):
        self.env = env.unwrapped
        self.flat_size = self.env.num_rows * self.env.num_cols
    
    def reward(self, reward: float):
        return reward / self.flat_size