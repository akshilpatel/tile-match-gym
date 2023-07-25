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


@pytest.mark.parametrize("height, width, num_colours", [
    # (4, 4, 4),
    (90, 2, 12),
    # (50, 50, 4),
    # (8, 8, 1),
])
def test_(height: int, width:int, num_colours:int, seed:Optional[int] = None):
    board = Board(height, width, num_colours, seed)
    
