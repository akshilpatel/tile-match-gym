import pytest
from tile_match_gym.board import Board
import numpy as np
from typing import Optional, List, Tuple
from tests.utils import create_alternating_array, contains_threes


# TODO: Rewrite test to work for generic tile encodings.     
# Assumes that _get_colour_lines works correctly.
def test_generate_board():
    
    for i in range(500):
        b = Board(num_rows=3, num_cols=3, num_colours=3, colour_specials= ["vertical_laser", "horizontal_laser", "bomb"], colourless_specials=["cookie"], seed=i)
        b.generate_board()
        # No matches
        line_matches = b.get_colour_lines()
        assert line_matches == []
        # All numbers within num_colourless_specials + 1 ,..., (1 + num_colour_specials) * num_colours + num_colourless_specials + 1
        assert np.all(b.board[0] > 0)
        assert np.all(b.board[1] == 1)

        # No colourless specials
        b = Board(num_rows=6, num_cols=5, num_colours=5, colour_specials= ["vertical_laser", "horizontal_laser", "bomb"], colourless_specials=["cookie"], seed=i)
        b.generate_board()
        line_matches = b.get_colour_lines()
        assert line_matches == []
        assert np.all(b.board[0] > 0)
        assert np.all(b.board[1] == 1)
        # print(b.board)

        # # No colour specials
        b = Board(num_rows=10, num_cols=10, num_colours=7, colour_specials= [], colourless_specials=["cookie"], seed=i)
        b.generate_board()
        line_matches = b.get_colour_lines()
        
        assert line_matches == []        
        assert np.all(b.board[0] > 0)
        assert np.all(b.board[1] == 1)

        print("DONE WITH GENERATE BOARD")
