import gymnasium as gym
import numpy as np

from gymnasium.spaces import Discrete, Box
from typing import Optional, List, Tuple
from collections import OrderedDict
from tile_match_gym.board import Board
from tile_match_gym.board import is_move_effective

class TileMatchEnv(gym.Env):
    metadata = {'render_modes': ['string']}
    def __init__(
            self, 
            num_rows:int, 
            num_cols:int, 
            num_colours:int, 
            num_moves: int,
            colourless_specials:List[str], 
            colour_specials: List[str],
            seed: Optional[int] = 1,
            render_mode: str = "string"
            ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_colours = num_colours
        self.colourless_specials = colourless_specials
        self.colour_specials = colour_specials
        self.num_moves = num_moves

        if render_mode == "string":
            self.colour_map = self.np_random.choice(range(105, 230), size=self.num_colours + 1, replace=False)
        self.render_mode = render_mode

        # Each coordinate can switch right or down but those one the right/bottom edge can't switch right/down
        self.num_actions = int((self.num_rows * self.num_cols * 2) - self.num_rows - self.num_cols )

        self.num_colour_specials = len(self.colour_specials)
        self.num_colourless_specials = len(self.colourless_specials)

        self.seed = seed

        np_random = np.random.default_rng(seed=seed)
        self.board = Board(num_rows, num_cols, num_colours, colourless_specials, colour_specials, np_random)
        self.np_random = self.board.np_random
        obs_low = np.array([np.zeros((self.num_rows, self.num_cols), dtype=np.int32), np.full((self.num_rows, self.num_cols), - self.num_colourless_specials, dtype=np.int32)])
        obs_high = np.array([np.full((self.num_rows, self.num_cols), self.num_colours, dtype=np.int32), np.full((self.num_rows, self.num_cols), self.num_colour_specials + 2, dtype=np.int32)]) # + 1 for empty
        
        self._board_observation_space = Box(
            low=obs_low, 
            high=obs_high,
            shape=(2, self.num_rows, self.num_cols),
            dtype=np.int32,
            seed = self.seed)
        
        self._moves_left_observation_space = Discrete(self.num_moves + 1, seed=self.seed)

        self.observation_space = gym.spaces.Dict({
            "board": self._board_observation_space,
            "num_moves_left": self._moves_left_observation_space
        })
        
        self.timer = None
        self.renderer = None
        self.action_space = Discrete(self.num_actions, seed=self.seed)

    def set_seed(self, seed:int) -> None:
        self.action_space.seed = seed
        self.observation_space.seed = seed
        self.board.np_random = np.random.default_rng(seed=seed)

    def reset(self, seed=None, **kwargs)  -> Tuple[dict, dict]:
        if seed is not None:
            self.set_seed(seed)
        self.board.generate_board()
        self.timer = 0
        obs = self._get_obs()
        info = {'effective_actions': self._get_effective_actions()}
        return obs, info

    def step(self, action: int) -> Tuple[dict, int, bool, bool, dict]:
        if self.timer is None or self.timer >= self.num_moves:
            raise Exception("You must call reset before calling step")
        
        coord1, coord2 = self._action_to_coords(action)
        num_eliminations, is_combination_match, num_new_specials, num_specials_activated, shuffled = self.board.move(coord1, coord2)
        
        self.timer += 1
        done = self.timer == self.num_moves
        effective_actions = self._get_effective_actions()
        info = {
            "is_combination_match": is_combination_match,
            "num_new_specials": num_new_specials,
            "num_specials_activated": num_specials_activated,
            "shuffled": shuffled,
            "effective_actions": effective_actions
        }
        next_obs = self._get_obs()
        return next_obs, num_eliminations, done, False, info
    
    def _get_obs(self) -> dict:
        return OrderedDict([("board", self.board.board), ("num_moves_left", self.num_moves - self.timer)])

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
            col = action_ % (self.num_cols - 1)
            return (row, col), (row, col + 1)

    def _get_effective_actions(self) -> List[int]:
        if self.timer == self.num_moves:
            return []
        effective_actions = []

        action_check = lambda a: is_move_effective(self.board.board, *self._action_to_coords(a))
        effective_actions = list(filter(action_check, range(self.num_actions)))
        # for a in range(self.num_actions):
        #     coord1, coord2 = self._action_to_coords(a)
        #     if is_move_effective(self.board.board, coord1, coord2):
        #         effective_actions.append(a)
        return effective_actions

    def render(self) -> None:
        if self.render_mode == "string":
            color = lambda id, c: "\033[48;5;16m" + f"\033[38;5;{self.colour_map[id]}m{c}\033[0m"
            height = self.board.board.shape[1]
            width = self.board.board.shape[2]

            print(" " + "-" * (width * 2 + 1))
            for row_num in range(height):
                print("| ", end="\033[48;5;16m")
                for col in range(width):
                    tile_colour = self.board.board[0, row_num, col]
                    tile_type = self.board.board[1, row_num, col]
                
                    print(color(tile_colour, tile_type), end="\033[48;5;16m ")
                    print("\033[0m", end="")

                print("|", end="\n")
            print(" " + "-" * (width * 2 + 1))

        elif self.render_mode == "image":
            pass

        

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
