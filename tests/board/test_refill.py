import pytest
from tile_match_gym.board import Board
import numpy as np
from typing import Optional, List, Tuple
from tests.utils import create_alternating_array, wipe_coords
import copy

example_board = Board(5, 7, 3) 
example_board.board = np.array([
                               [2,3,2,3,2,3,2],
                               [3,2,3,2,3,2,3],
                               [2,3,2,3,2,3,2],
                               [3,2,3,2,3,2,3],
                               [2,3,2,3,2,3,2],
                               ])



def test_refill(board: np.ndarray, seed:Optional[int] = None):
    board = Board(0, 0, 0, seed)
    board.refill()
    assert 0 not in board.board
