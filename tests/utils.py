import numpy as np
from tile_match_gym.board import Board
from typing import List, Tuple
from copy import deepcopy



# create an array of alternating 2's and 3's
def create_alternating_array(height: int, width: int) -> np.ndarray:
    arr = np.ones((2, height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            arr[0, i, j] = 2 - int((i % 2) == (j % 2))
    return arr 


def create_board_from_array(arr: np.ndarray, num_colours=4) -> Board:
    
    height, width, _ = arr.shape
    seed = 1
    board = Board(height, width, num_colours, ["cookie"], ["vertical_laser", "horizontal_laser", "bomb"], seed)
    # board.board[0] = deepcopy(arr)
    # board.board[1] = np.ones_like(arr)
    return board


def create_alternating_board(height: int, width: int) -> Board:
    arr = create_alternating_array(height, width)
    return create_board_from_array(arr)


def contains_threes(arr: np.ndarray) -> bool:
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if j < cols - 2 and arr[0, i, j] == arr[0, i, j + 1] == arr[0, i, j + 2] != 0:
                return True
            if i < rows - 2 and arr[0, i, j] == arr[0, i + 1, j] == arr[0, i + 2, j] !=0:
                return True
    return False


def wipe_coords(board: Board, coords: List[Tuple[int, int]]) -> np.ndarray:    
    for coord in coords:
        board.board[:, coord[0], coord[1]] *= 0


def get_special_locations(board: Board) -> List[Tuple[int, int]]:
    locations = []
    for i in range(board.num_rows):
        for j in range(board.num_cols):
            if board.board[1, i, j] not in [0, 1]:
                locations.append((i, j))
    return locations
