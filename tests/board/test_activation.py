import pytest
from src.board import Board
import numpy as np
from typing import Optional, List, Tuple
from utils import print_board_diffs

################################################################################
################## This is just templating - needs to be changed ###############
################################################################################

t1 = {
    "original": np.array([
        [2,3,2,3,2,3,2],
        [3,2,3,2,3,2,3],
        [2,3,2,3,2,3,2],
        [3,2,3,2,3,2,3],
        [2,3,2,3,2,3,2],
        ]),
    "expected": np.array([
        [2,3,2,3,2,3,2],
        [3,2,3,2,3,2,3],
        [2,3,2,3,2,3,2],
        [3,2,3,2,3,2,3],
        [2,3,2,3,2,3,2],
        ]),
}



def test_activation(height: int, width:int, num_colours:int, seed:Optional[int] = None):
    board = Board(height, width, num_colours, seed)
    assert board.board.shape == (height, width)
    assert board.num_colours == num_colours


