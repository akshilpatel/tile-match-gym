# Tile Matching Reinforcement Learning Environments

Welcome to the Reinforcement Learning Environments for Tile Matching Games repository! Here you can find a collection of tile matching game environments (like Bejeweled or Candy Crush), poised to push reinforcement learning research forwards.

This genre of games is characterised by the following features, which we find useful for reinforcement learning research:

- Large action spaces
- Intuitive action hierarchies
- Procedurally generated levels
- Structured complex stochasticity in transition dynamics

## Installation

You can install the package via pip:

```pip install tile-match-gym```

## Example Usage

We follow the the Farama Foundation Gymnasium API:

```
from tile_match_gym.tile_match_env import TileMatchEnv

env = TileMatchEnv(
  num_rows=10, 
  num_cols=10, 
  num_colours=4, 
  num_moves=30, 
  colourless_specials=[], 
  colour_specials=[], 
  seed=2
  )

obs, _ = env.reset()

while True:
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    if done:
        break
    else:
      next_obs = obs
```

## Citation

We'd love it if you use our package for your research! If you do use code from this repository please cite us as below:

```
@software{tile_match_gym,
  author = {Patel, Akshil and Elson, James},
  title = {{Tile Matching Game Reinforcement Learning Environments}},
  url = {https://github.com/akshilpatel/tile-match-gym},
  version = {0.0.5},
  year = {2023}
  }
```
