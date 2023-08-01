import numpy as np
import pytest

from tile_match_gym.board import Board

def make_board():
    board = Board(8, 7, 4)
    return board

@pytest.fixture
def make_board_fixture():
    return make_board()

def test_get_lowest_v_match_coords():
    board = make_board_fixture
    board.board = np.array([
        [4, 4, 2, 1, 3, 2, 3], # 0
        [3, 1, 1, 1, 2, 3, 4], # 1
        [4, 4, 2, 1, 3, 2, 3], # 2
        [2, 3, 2, 2, 2, 3, 2], # 3
        [1, 1, 2, 3, 3, 3, 4], # 4
        [1, 3, 2, 4, 1, 2, 4], # 5
        [4, 3, 1, 3, 1, 4, 4], # 6
        [3, 3, 3, 1, 1, 1, 4], # 7
    ])
    print(board)

    board.board = np.array([
        [4, 4, 2, 2, 2, 2, 3], # 0
        [3, 1, 1, 2, 2, 3, 4], # 1
        [4, 4, 2, 1, 3, 2, 3], # 2
        [2, 3, 2, 2, 1, 3, 2], # 3
        [1, 1, 2, 3, 2, 3, 3], # 4
        [1, 3, 2, 4, 1, 2, 4], # 5
        [4, 3, 1, 3, 1, 4, 4], # 6
        [3, 2, 3, 2, 1, 1, 4], # 7
    ])

    print(board.get_lowest_h_match_coords())

    board.board = np.array([
        [4, 4, 4, 3, 2, 2, 2], # 0
        [3, 1, 1, 2, 2, 3, 4], # 1
        [4, 4, 2, 1, 3, 2, 3], # 2
        [2, 3, 2, 2, 1, 3, 2], # 3
        [1, 1, 2, 3, 2, 3, 3], # 4
        [1, 3, 2, 4, 1, 2, 4], # 5
        [4, 3, 1, 3, 1, 4, 4], # 6
        [3, 2, 3, 2, 1, 1, 4], # 7
    ])

    print(board.get_lowest_h_match_coords())

    print("original")
    board.print_board()