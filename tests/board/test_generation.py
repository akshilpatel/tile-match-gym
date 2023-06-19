import pytest
from src.board import Board
import numpy as np
from typing import Optional, List, Tuple

################################################################################
################## This is just templating - needs to be changed ###############
################################################################################

@pytest.mark.parametrize("height, width, num_colours", [
    (4, 4, 4),
    (90, 2, 12),
    (50, 50, 4),
    (8, 8, 1),
])
def test_does_not_contain_threes(height: int, width:int, num_colours:int, seed:Optional[int] = None):
    board = Board(height, width, num_colours, seed)
    assert not contains_threes(board.board)

def contains_threes(arr: np.ndarray) -> bool:
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if j < cols - 2 and arr[i, j] == arr[i, j + 1] == arr[i, j + 2]:
                return True
            if i < rows - 2 and arr[i, j] == arr[i + 1, j] == arr[i + 2, j]:
                return True
    return False

