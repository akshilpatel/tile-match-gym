import pytest
from tile_match_gym.board import Board, TileTranslator
import numpy as np
from typing import Optional, List, Tuple
from tests.utils import create_alternating_array, contains_threes

################################################################################
################## This is just templating - needs to be changed ###############
################################################################################

# @pytest.fixture
# def board():
#     example_board = Board(0,0,0,0,np.array([
#         [2,3,2,3,2,3,2],
#         [3,2,3,2,3,2,3],
#         [2,3,2,3,2,3,2],
#         [3,2,3,2,3,2,3],
#         [2,3,2,3,2,3,2],
#     ]))
#     yield example_board


def test_get_tile_encoding():
    tile_translator = TileTranslator(
        num_colours=4, board_shape=(5, 10), colourless_specials=["cookie"], colour_specials=["vertical_laser", "horizontal_laser", "bomb"]
    )

    # Empty tile
    assert tile_translator.get_tile_encoding("none", 0) == 0
    # Cookie
    assert tile_translator.get_tile_encoding("cookie", 0) == 1

    # Normal tiles
    assert tile_translator.get_tile_encoding("normal", 0) == 2
    assert tile_translator.get_tile_encoding("normal", 3) == 5

    # Lasers
    assert tile_translator.get_tile_encoding("vertical_laser", 3) == 9
    assert tile_translator.get_tile_encoding("horizontal_laser", 1) == 11

    tile_translator = TileTranslator(num_colours=7, board_shape=(5, 10), colourless_specials=[], colour_specials=["vertical_laser", "horizontal_laser", "bomb"])
    assert tile_translator.get_tile_encoding("normal", 0) == 1
    assert tile_translator.get_tile_encoding("normal", 4) == 5
    assert tile_translator.get_tile_encoding("bomb", 6) == 28

    tile_translator = TileTranslator(num_colours=34, board_shape=(12, 5), colourless_specials=[], colour_specials=[])
    assert tile_translator.get_tile_encoding("normal", 0) == 1
    assert tile_translator.get_tile_encoding("normal", 33) == 34
    print(tile_translator.all_tile_types)
    with pytest.raises(ValueError):
        tile_translator.get_tile_encoding("bomb", 10)


def test_get_type_color():
    tile_translator = TileTranslator(
        num_colours=4, board_shape=(5, 10), colourless_specials=["cookie"], colour_specials=["vertical_laser", "horizontal_laser", "bomb"]
    )
    assert tile_translator.get_type_colour(0) == (0, 0)
    assert tile_translator.get_type_colour(1) == (1, 0)
    assert tile_translator.get_type_colour(4) == (2, 2)
    assert tile_translator.get_type_colour(12) == (4, 2)

    tile_translator = TileTranslator(num_colours=5, board_shape=(5, 10), colourless_specials=["cookie"], colour_specials=["bomb"])
    assert tile_translator.get_type_colour(0) == (0, 0)
    assert tile_translator.get_type_colour(1) == (1, 0)
    assert tile_translator.get_type_colour(4) == (2, 2)
    assert tile_translator.get_type_colour(8) == (3, 1)

    tile_translator = TileTranslator(num_colours=5, board_shape=(5, 10), colourless_specials=[], colour_specials=[])
    assert tile_translator.get_type_colour(0) == (0, 0)
    assert tile_translator.get_type_colour(1) == (1, 0)
    assert tile_translator.get_type_colour(4) == (1, 3)
    assert tile_translator.get_type_colour(5) == (1, 4)


def test_is_special():
    tile_translator = TileTranslator(
        num_colours=4, board_shape=(5, 10), colourless_specials=["cookie"], colour_specials=["vertical_laser", "horizontal_laser", "bomb"]
    )
    assert not tile_translator.is_special(0)
    assert tile_translator.is_special(1)
    assert not tile_translator.is_special(4), tile_translator.get_type_colour(4)
    assert tile_translator.is_special(12)

    tile_translator = TileTranslator(num_colours=7, board_shape=(5, 10), colourless_specials=[], colour_specials=[])
    assert not tile_translator.is_special(0)
    assert not tile_translator.is_special(1)
    assert not tile_translator.is_special(4)
    assert not tile_translator.is_special(7)
