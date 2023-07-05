import pytest
from src.tile_match_gym.board import Board
import numpy as np
from typing import Optional, List, Tuple
from utils import print_board_diffs

################################################################################
################## This is just templating - needs to be changed ###############
################################################################################

# Need to call 'pytest -k test_activation -s' to run the file and get output if error
@pytest.fixture
def t1():

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

    yield t1


def test_activation():
    board = Board(5, 7, 4, 0)
    # board = Board(height, width, num_colours, seed)
    board.board = t1["original"]
    board.width = 7
    board.height = 5
    board.num_colours = 4
    board.apply_activation((1,1), 2)
    
    if not np.array_equal(board.board, t1["expected"]):
        print()
        print_board_diffs.highlight_board_diff(board.board, t1["expected"])