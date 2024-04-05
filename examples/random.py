import numpy as np
from tile_match_gym.tile_match_env import TileMatchEnv

class RandomAgent:
    def __init__(self, num_actions, rng, use_effective_actions = False):
        self.num_actions = num_actions
        self.use_effective_actions = use_effective_actions
        self.rng = rng

    def choose_action(self, obs, *args, effective_actions=None, **kwargs):
        if self.use_effective_actions and effective_actions != None:
            return self.rng.choice(effective_actions)
        return self.rng.choice(range(self.num_actions))