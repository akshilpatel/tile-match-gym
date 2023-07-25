import pytest
from tile_match_gym.board import Board
import numpy as np
from typing import Optional, List, Tuple
from tests.utils import create_alternating_array, contains_threes

################################################################################
################## This is just templating - needs to be changed ###############
################################################################################


@pytest.mark.parametrize("height, width, num_colours", [
    (4, 4, 4),
    (90, 2, 12),
    # (50, 50, 4),
    # (8, 8, 1),
])
def test_does_not_contain_threes(height: int, width:int, num_colours:int, seed:Optional[int] = None):
    board = Board(height, width, num_colours, seed)
    assert not contains_threes(board.board)
    assert board.board.shape == (height, width)
    assert board.num_colours == num_colours
