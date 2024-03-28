
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import distinctipy
import importlib.resources as pkg_resources

from gymnasium.spaces import Discrete, Box
from typing import Optional, List, Tuple
from collections import OrderedDict
from tile_match_gym.board import Board
from IPython.display import clear_output
from matplotlib import colors




class TileMatchEnv(gym.Env):
    metadata = {'render_modes': ['string', "image"]}
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


        self.render_mode = render_mode

        # Each coordinate can switch right or down but those one the right/bottom edge can't switch right/down
        self.num_actions = int((self.num_rows * self.num_cols * 2) - self.num_rows - self.num_cols )

        self.num_colour_specials = len(self.colour_specials)
        self.num_colourless_specials = len(self.colourless_specials)

        self.seed = seed
        self.board = Board(num_rows, num_cols, num_colours, colourless_specials, colour_specials, seed=seed)
        self.np_random = self.board.np_random

        if render_mode == "string":
            self.colour_map = self.np_random.choice(range(105, 230), size=self.num_colours + 1, replace=False)
        else:
            render_types = {1: "ordinary", 2:"vertical", 3:"horizontal", 4:"bomb", -1:"cookie"}
            self.colours = distinctipy.get_colors(self.num_colours, colorblind_type="Deuteranopia", pastel_factor=0.7, rng=self.seed)
            self.images = {x: plt.imread(pkg_resources.path(f"{x}.png")) for x in render_types}        
            render_colours = np.concatenate([np.array(self.colours), np.ones((num_colours, 1))], axis=1)
            self.colour_images = {
                x: [colour_image(self.images[x], render_colours[j]) for j in render_colours] for x in self.images
            }

            self.colour_images = []
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
        for a in range(self.num_actions):
            coord1, coord2 = self._action_to_coords(a)
            if self.board.is_move_effective(coord1, coord2):
                effective_actions.append(a)
        return effective_actions

    def render(self) -> None:
        if self.render_mode == "string":
            
            colour = lambda id, c: "\033[48;5;16m" + f"\033[38;5;{self.colour_map[id]}m{c}\033[0m"
            height = self.board.board.shape[1]
            width = self.board.board.shape[2]

            print(" " + "-" * (width * 2 + 1))
            for row_num in range(height):
                print("| ", end="\033[48;5;16m")
                for col in range(width):
                    tile_colour = self.board.board[0, row_num, col]
                    tile_type = self.board.board[1, row_num, col]
                
                    print(colour(tile_colour, tile_type), end="\033[48;5;16m ")
                    print("\033[0m", end="")

                print("|", end="\n")
            print(" " + "-" * (width * 2 + 1))

        elif self.render_mode == "image":
            line_colour = np.array((60, 60, 60)) / 255
            fig = plt.figure(num="env_render", figsize=(12, 9))
            ax = plt.gca()
            ax.clear()
            clear_output(True)
            fig.patch.set_facecolor(np.array(0,0,0,1))
            # Render the board
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    tile_type = self.board.board[1, r, c]
                    tile_colour = self.board.board[0, r, c]
                    if tile_type in self.colour_images:
                        image = self.colour_images[tile_type][tile_colour]
                        ax.imshow(image, extent=(c-0.5, c+0.5, r-0.5, r+0.5), aspect='auto', zorder=1, alpha=1)
            
            ax.grid(which='major', axis='both', linestyle='-', color=line_colour, linewidth=2, zorder=1)
            ax.set_xticks(np.arange(-0.5, self.num_cols, 1))
            ax.set_xticklabels([])
            ax.set_yticks(np.arange(-0.5, self.num_rows, 1))
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)
            plt.show()
                
        

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


def colour_image(image, colour):
    mask = np.all(image == [0, 0, 0, 1], axis=-1)
    image[mask] = colour
    return image