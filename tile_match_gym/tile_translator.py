import numpy as np

from typing import Tuple


class TileTranslator:
    def __init__(self, num_colours: int, board_shape: Tuple[int, int]):
        self.num_colours = num_colours
        self.board_shape = board_shape

    def _tile_type_str(self, tile_idx: int):
        tile_type = self.get_tile_type(tile_idx)
        if tile_type == 0:
            return "ordinary"
        elif tile_type == 1:
            return "vertical_stripe"
        elif tile_type == 2:
            return "horizontal_stripe"
        elif tile_type == 3:
            return "bomb"
        elif tile_type == 4:
            return "cookie"
        else:
            raise ValueError("Invalid tile type.")

    def is_tile_ordinary(self, tile_idx: int) -> bool:
        return (tile_idx - 1) < self.num_colours

    def get_tile_type(self, tile_idx: int) -> int:
        """
        Convert the tile index to whether the tile is ordinary, or which type of special it is.

        Args:
            tile_idx (int): Raw tile encoding.

        Returns:
            int: Index of tile type
        """
        return tile_idx // self.num_colours

    def get_tile_colour(self, tile_idx: int):
        return (tile_idx - 1) % self.num_colours