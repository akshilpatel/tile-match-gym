import numpy as np
from tile_match_gym.board import Board
from typing import List, Tuple
from copy import deepcopy


# create an array of alternating 2's and 3's
def create_alternating_array(height: int, width: int) -> np.ndarray:
    arr = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            arr[i, j] = 2 - int((i % 2) == (j % 2))
    return arr


def create_board_from_array(arr: np.ndarray) -> Board:
    num_colours = len(np.unique(arr))
    height, width = arr.shape
    seed = 1
    board = Board(height, width, num_colours, seed)
    board.board = deepcopy(arr)
    return board


def create_alternating_board(height: int, width: int) -> Board:
    arr = create_alternating_array(height, width)
    return create_board_from_array(arr)


def contains_threes(arr: np.ndarray) -> bool:
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if j < cols - 2 and arr[i, j] == arr[i, j + 1] == arr[i, j + 2]:
                return True
    if i < rows - 2 and arr[i, j] == arr[i + 1, j] == arr[i + 2, j]:
        return True
    return False


def wipe_coords(board: Board, coords: List[Tuple[int, int]]) -> np.ndarray:
    b = deepcopy(board.board)
    for coord in coords:
        b[coord] = 0
    return b
