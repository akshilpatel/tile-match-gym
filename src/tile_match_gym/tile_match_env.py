from tile_match_gym.board import Board
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from typing import Optional, List
import numpy as np

class TileMatchEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(
            self, 
            num_rows:int, 
            num_cols:int, 
            num_colours:int, 
            num_moves: int,
            colourless_specials:List[str], 
            colour_specials: List[str],
            seed: Optional[int] = None
            ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_colours = num_colours
        self.colourless_specials = colourless_specials
        self.colour_specials = colour_specials
        self.num_moves = num_moves

        # Each coordingate can switch right or down but those one the right/bottom edge can't switch right/down
        self.num_actions = int((self.num_rows * self.num_cols * 2) - self.num_rows - self.num_cols )

        self.num_colour_specials = len(self.colour_specials)
        self.num_colourless_specials = len(self.colourless_specials)

        self.board = Board(num_rows, num_cols, num_colours, colourless_specials, colour_specials)

        obs_low = np.array([np.zeros((self.num_rows, self.num_cols), dtype=int), np.full((self.num_rows, self.num_cols), -self.num_colourless_specials, dtype=int)])
        obs_high = np.array([np.full((self.num_rows, self.num_cols), self.num_colours, dtype=int), np.full((self.num_rows, self.num_cols), self.num_colour_specials, dtype=int)])
        
        if seed is None: seed = 1
        self.seed = seed
        self.observation_space = Box(
            low=obs_low, 
            high=obs_high,
            shape=(2, self.num_rows, self.num_cols),
            dtype=int,
            seed = self.seed)
        
        self.timer = None
    
        self.action_space = Discrete(self.num_actions, seed=self.seed)
        self.np_random = np.random.default_rng(seed=self.seed)

    def set_seed(self, seed:int):
        self.action_space.seed = seed
        self.observation_space.seed = seed
        self.np_random = np.random.default_rng(seed=seed)

    def reset(self):
        self.board.generate_board()
        obs = self._get_obs()
        info = {"num_moves_left": self.num_moves}
        self.timer = 0
        return obs, info

    def step(self, action):
        if self.timer is None or self.timer >= self.num_moves:
            raise Exception("You must call reset before calling step")
        
        coord1, coord2 = self._action_to_coords(action)
        num_eliminations, is_combination_match, num_new_specials, num_specials_activated, shuffled = self.board.move(coord1, coord2)

        self.timer += 1

        if self.timer == self.num_moves:
            self.terminal = True
        
        info = {
            "is_combination_match": is_combination_match,
            "num_new_specials": num_new_specials,
            "num_specials_activated": num_specials_activated,
            "shuffled": shuffled,
            "num_moves_left": self.num_moves - self.timer
        }

        next_obs = self._get_obs()
        return next_obs, num_eliminations, self.terminal, False, info
    
    def _get_obs(self):
        return self.board.board

    def _action_to_coords(self, action:int):
        if not 0 <= action <= self.num_actions:
            raise ValueError(f"Action {action} is not valid for this board {self.num_rows, self.num_cols}")
        if action < self.num_cols * (self.num_rows - 1):
            row = action // self.num_cols
            col = action % self.num_cols
            return (row, col), (row + 1, col)
        else:
            action_ = action - self.num_cols * (self.num_rows - 1)
            row = action_ // (self.num_cols - 1)            
            col = action % (self.num_cols - 1)
            return (row, col), (row, col + 1)

    def render(self, mode="human"):
        print(self.board.board)

    def close(self):
        if self.renderer:
            self.renderer.close()