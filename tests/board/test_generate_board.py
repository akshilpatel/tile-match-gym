import pytest
from tile_match_gym.board import Board
import numpy as np
from typing import Optional, List, Tuple
from tests.utils import create_alternating_array, contains_threes

################################################################################
################## This is just templating - needs to be changed ###############
################################################################################


# @pytest.mark.parametrize("height, width, num_colours", [
#     (4, 4, 4),
#     (90, 2, 12),
#     # (50, 50, 4),
#     # (8, 8, 1),
# ])


def test_does_not_contain_threes(height: int, width: int, num_colours: int, seed: Optional[int] = None):
    board = Board(height, width, num_colours, ["cookie"], ["vertical_laser", "horizontal_laser", "bomb"], seed)
    assert not contains_threes(board.board)
    assert board.board.shape == (height, width)
    assert board.num_colours == num_colours


def test_correct_numbers():
    board = Board(4, 4, 4, ["cookie"], ["vertical_laser", "horizontal_laser", "bomb"])
    assert set(board.board.flatten()).issubset(set(range(1, 5)))


def test_no_matches():
    assert False


# TODO: Rewrite test to work for generic tile encodings. 
    
# Assumes that _get_colour_lines works correctly.
def test_generate_board():
    # All inclusive 
    for i in range(100):
        b = Board(num_rows=3, num_cols=4, num_colours=3, colour_specials= ["vertical_laser", "horizontal_laser", "bomb"], colourless_specials=["cookie"], seed=i)
        # No matches
        line_matches = b._get_colour_lines()
        assert line_matches == []
        # All numbers within num_colourless_specials + 1 ,..., (1 + num_colour_specials) * num_colours + num_colourless_specials + 1
        assert np.all(b.board > 0)
        for i in range(b.num_rows):
            for j in range(b.num_cols):
                assert not b.tile_translator.is_special(b.board[i,j])

        # No colourless specials
        b = Board(num_rows=4, num_cols=14, num_colours=21, colour_specials= ["vertical_laser", "horizontal_laser", "bomb"], colourless_specials=[], seed=i)
        line_matches = b._get_colour_lines()
        assert line_matches == []
        assert np.all(b.board > 0)
        for i in range(b.num_rows):
            for j in range(b.num_cols):
                assert not b.tile_translator.is_special(b.board[i,j])



        # No colour specials
        b = Board(num_rows=10, num_cols=10, num_colours=7, colour_specials= [], colourless_specials=["cookie"], seed=i)
        line_matches = b._get_colour_lines()
        
        assert line_matches == []        
        assert np.all(b.board > 0)
        for i in range(b.num_rows):
            for j in range(b.num_cols):
                assert not b.tile_translator.is_special(b.board[i,j])
