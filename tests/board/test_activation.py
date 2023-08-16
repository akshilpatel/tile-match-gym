import pytest
from tile_match_gym.board import Board
import numpy as np
from typing import Optional, List, Tuple
from tile_match_gym.utils import print_board_diffs

################################################################################
################## This is just templating - needs to be changed ###############
################################################################################

# Need to call 'pytest -k test_activation -s' to run the file and get output if error
# @pytest.fixture
# def t1():
#     t1 = {
#         "original": np.array([
#             [2,3,2,3,2,3,2],
#             [3,2,3,2,3,2,3],
#             [2,3,2,3,2,3,2],
#             [3,2,3,2,3,2,3],
#             [2,3,2,3,2,3,2],
#             ]),
#         "expected": np.array([
#             [2,3,2,3,2,3,2],
#             [3,2,3,2,3,2,3],
#             [2,3,2,3,2,3,2],
#             [3,2,3,2,3,2,3],
#             [2,3,2,3,2,3,2],
#             ]),
#     }
#
#     yield t1

t1 = {
    "original": np.array(
        [
            [2, 3, 2, 3, 2, 3, 2],
            [3, 2, 3, 2, 3, 2, 3],
            [2, 3, 2, 3, 2, 3, 2],
            [3, 2, 3, 2, 3, 2, 3],
            [2, 3, 2, 3, 2, 3, 2],
        ]
    ),
    "expected": np.array(
        [
            [2, 3, 2, 3, 2, 3, 2],
            [3, 2, 3, 2, 3, 2, 3],
            [2, 3, 2, 3, 2, 3, 2],
            [3, 2, 3, 2, 3, 2, 3],
            [2, 3, 2, 3, 2, 3, 2],
        ]
    ),
}


def test_activation():
    board = Board(5, 7, 4, ["cookie"], ["vertical_laser", "horizontal_laser", "bomb"], seed=0)
    board.board = t1["original"]
    board.width = 7
    board.height = 5
    board.num_colours = 4
    board.apply_activation((2, 2), activation_type=1)

    if not np.array_equal(board.board, t1["expected"]):
        print()
        print_board_diffs.highlight_board_diff(board.board, t1["expected"])
