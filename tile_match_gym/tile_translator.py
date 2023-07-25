import numpy as np

from typing import Tuple


class TileTranslator:
    def __init__(self, num_colours: int, board_shape: Tuple[int, int], num_specials: int = 4):
        self.num_colours = num_colours
        self.num_specials = num_specials
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

    def tile_to_id(self, tile_type: str):
        if tile_type == "norm":
            return 1
        elif tile_type == "vertical_laser":
            return 2
        elif tile_type == "horizontal_laser":
            return 3
        elif tile_type == "bomb":
            return 4
        elif tile_type == "cookie":
            return 0
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

    def get_tile_number(self, tile: str, colour: int):
        """
        Convert the tile type and colour to the tile index.

        0 = empty
        1 = cookie
        2 = color1 norm
        3 = color1 vertical laser
        4 = color1 horizontal laser
        5 = color1 bomb
        6 = color2 norm
        7 = color2 vertical laser
        8 = color2 horizontal laser
        9 = color2 bomb

        equation -> color * num_specials + tile_type + 2
        """

        print("tile: ", tile)
        type_encoding = self.tile_to_id(tile)

        if tile == "cookie":
            return 1
        else:
            return colour * self.num_specials + type_encoding + 1

