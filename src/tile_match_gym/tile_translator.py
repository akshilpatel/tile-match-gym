import numpy as np

from typing import Tuple


class TileTranslator:
    def __init__(self, num_colours: int, board_shape: Tuple[int, int], num_specials: int = 4):
        self.num_colours = num_colours
        self.num_specials = num_specials
        self.board_shape = board_shape
        self.type_names = ["none", "cookie", "norm", "vertical_laser", "horizontal_laser", "bomb"]
    
    def get_tile_encoding(self, tile: str, colour: int):
        """
        Convert the tile type and colour to the tile index.

        0 = none
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
        type_encoding = self.type_names.index(tile)
        if tile == "none":
            return 0
        elif tile == "cookie":
            return 1
        else:
            return colour * self.num_specials + type_encoding - 4


    def get_type_color(self, encoding: int) -> Tuple[int, int]:
        """
        Convert the encoding to a tile type and colour.

        types:
            0 = none
            1 = cookie
            2 = norm
            3 = vertical laser
            4 = horizontal laser
            5 = bomb
        """

        if encoding == 0:
            return 0, 0
        elif encoding == 1:
            return 1, 20
        else:
            return (encoding - 2) % self.num_specials + 2, (encoding - 2) // self.num_specials + 1

    def is_same_color(self, encoding1: int, encoding2: int) -> bool:
        """
        Check if the two encodings are the same colour.
        """
        _, c1 = self.get_type_color(encoding1)
        _, c2 = self.get_type_color(encoding2)
        return c1 == c2

    def get_str(self, encoding: int) -> str:
        """
        Convert the encoding to a string.
        """
        t, c = self.get_type_color(encoding)

        if t == 0:
            return "none"
        elif t == 1:
            return "cookie"
        elif t == 2:
            return "color{} norm".format(c)
        elif t == 3:
            return "color{} vertical laser".format(c)
        elif t == 4:
            return "color{} horizontal laser".format(c)
        elif t == 5:
            return "color{} bomb".format(c)
        
        return "?"

    def get_char(self, encoding: int) -> str:
        """
        Convert the encoding to a character.
        """
        t, _ = self.get_type_color(encoding)

        if t == 0:
            return "0"
        elif t == 1:
            return "o"
        elif t == 2:
            return str(encoding)
        elif t == 3:
            return "|"
        elif t == 4:
            return "-"
        elif t == 5:
            return "*"
        
        return "?"

    def is_special(self, encoding: int) -> bool:
        """
        Check if the encoding is a special tile.
        """
        t, _ = self.get_type_color(encoding)
        return t != 2 and t != 0

