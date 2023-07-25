import pytest
from tile_match_gym.board import Board
import numpy as np
from typing import Optional, List, Tuple
from tests.utils import create_alternating_array
import copy

example_board = Board(0,0,0,0,np.array([
    [2,3,2,3,2,3,2],
    [3,2,3,2,3,2,3],
    [2,3,2,3,2,3,2],
    [3,2,3,2,3,2,3],
    [2,3,2,3,2,3,2],
]))


def wipe_coords(board: Board, coords: List[Tuple[int, int]]) -> None:
    b = copy.deepcopy(board)
    for coord in coords:
        b.board[coord] = 0
    return b

@pytest.mark.parametrize("board", [
    wipe_coords(example_board, [(0, 0)]), # single value
    wipe_coords(example_board, [(0, 0), (0, 1), (0, 2)]), # single row
    wipe_coords(example_board, [(0, 0), (0, 1), (2, 2), (4, 3)]), # random
])
def test_refill(board: np.ndarray, seed:Optional[int] = None):
    board = Board(0, 0, 0, seed)
    board.refill()
    assert 0 not in board.board


