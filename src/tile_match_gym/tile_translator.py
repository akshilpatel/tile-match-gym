import numpy as np

from typing import Tuple, List

# 0 = empty
# 1 = non_colour_special 0
# ...
# k = non_colour_special k
# k+1 normal colour 0
# ...
# k+n+1 normal colour n
# k+n+2 colour_special_1 colour 0
# ...
# k+2n+1 colour_special_1 colour n
# k+2n+2 colour_special_2 colour 0
# ...


class TileTranslator:
    def __init__(
        self,
        num_colours: int,
        board_shape: Tuple[int, int],
        colourless_specials: List[str] = ["cookie"],
        colour_specials: List[str] = ["vertical_laser", "horizontal_laser", "bomb"],
    ):
        self.num_colours = num_colours
        self.colour_specials = colour_specials
        self.colourless_specials = colourless_specials
        self.num_colourless_specials = len(colourless_specials)
        self.num_colour_specials = len(colour_specials)
        self.num_specials = self.num_colour_specials + self.num_colourless_specials
        self.board_shape = board_shape
        self.all_tile_types = ["none"] + self.colourless_specials + ["normal"] + self.colour_specials
        self.max_tile_encoding = self.num_colourless_specials + (1 + self.num_specials) * self.num_colours

    def get_tile_encoding(self, tile_type: str, tile_colour: int):
        """
        Convert the tile type and colour to the tile index.
        0 = none
        1 = colourless_special_0
        2 = colourless_special_1
        ...
        k = colourless_special_k-1
        k+1 = colour0 normal
        ...
        k+n+1 = colour0 vertical laser
        ...
        k+2n+1 = colour0 horizontal laser
        ...
        k+3n+1 = colour0 bomb
        ...
        k+4n = colourn-1 bomb

        """
        if not 0 <= tile_colour < self.num_colours:
            raise ValueError(f"The tile colour {tile_colour} not in possible tile colours (0, {self.num_colours})")

        elif tile_type not in self.all_tile_types:
            raise ValueError(f"The tile type {tile_type} not in possible tile types {self.all_tile_types}")

        if tile_type == "none":
            return 0
        elif tile_type in self.colourless_specials:
            return self.colourless_specials.index(tile_type) + 1  # 1 -> num_non_colour_specials-1
        elif tile_type == "normal":
            return 1 + self.num_colourless_specials + tile_colour  # num_non_colour_specials -> num_non_colour_specials + num_colours + 1
        else:
            type_idx = 1 + self.colour_specials.index(tile_type)  # To get past normal tiles, need to add 1.
            return (type_idx * self.num_colours) + tile_colour + 1 + self.num_colourless_specials

    def get_type_colour(self, encoding: int) -> Tuple[int, int]:
        """
        Convert the encoding to a tile type and colour.

        types:
            none: 0
            colourless_special_type_0: 1 (zero-based_indexing)
            ...
            colourless_special_type_k-1: k
            norm: k+1
            vertical laser: k+2
            horizontal laser: k+3
            bomb: k+4
        """
        if not 0 <= encoding <= self.max_tile_encoding:
            raise ValueError(f"The encoding {encoding} is not valid.")

        if encoding == 0:
            return 0, 0
        elif 0 < encoding <= self.num_colourless_specials:
            return encoding, 0
        else:
            recentered_encoding = encoding - 1 - self.num_colourless_specials
            tile_colour = recentered_encoding % self.num_colours
            tile_type = (recentered_encoding // self.num_colours) + self.num_colourless_specials + 1
            return tile_type, tile_colour

    def is_same_colour(self, encoding1: int, encoding2: int) -> bool:
        """
        Check if the two encodings are the same colour.
        """
        _, c1 = self.get_type_colour(encoding1)
        _, c2 = self.get_type_colour(encoding2)
        return c1 == c2

    def get_str(self, encoding: int) -> str:
        """
        Convert the encoding to a string.
        """
        t, c = self.get_type_colour(encoding)

        if t == 0:
            return "none"
        elif 0 < t <= self.num_colourless_specials:
            return f"Colourless special: {self.num_colourless_specials[t - 1]}"
        elif t == self.num_colour_specials + 1:
            return f"Colour{c} norm"
        elif t == self.num_colour_specials + 2:
            return f"Colour{c} vertical laser"
        elif t == self.num_colour_specials + 3:
            return f"Colour{c} horizontal laser"
        elif t == self.num_colour_specials + 4:
            return f"Colour{c} bomb"
        else:
            raise NotImplementedError(f"The encoding {encoding} is not valid.")

    def get_char(self, encoding: int) -> str:
        """
        Convert the encoding to a character.
        """
        # t, _ = self.get_type_colour(encoding)

        # if t == 0:
        #     return "0"
        # elif t == 1:
        #     return "o"
        # elif t == 2:
        #     return str(encoding)
        # elif t == 3:
        #     return "|"
        # elif t == 4:
        #     return "-"
        # elif t == 5:
        #     return "*"

        raise NotImplementedError("This method should not exist.")

    def is_special(self, encoding: int) -> bool:
        """
        Check if the encoding is a special tile.
        """
        t, _ = self.get_type_colour(encoding)
        return not (t == 0 or t == self.num_colourless_specials + 1)
