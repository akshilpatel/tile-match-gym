import pytest
from tests.utils import create_board_from_array, create_alternating_array
import numpy as np

# Test that gravity pushes down tiles

from copy import deepcopy
def test_gravity_falls():

    # Case 1
    arr = create_alternating_array(4, 3)
    board = create_board_from_array(arr)
    board.board[0, 0] = 0
    board.board[2, 0] = 0
    board.board[2, 2] = 0

    board.board[2, 1] = 0
    board.board[3, 2] = 0

    print(board.board)
    print("--")
    board.gravity()
    assert np.array_equal(board.board, np.array([[0, 0, 0],
                                                 [0, 2, 0],
                                                 [2, 1, 2],
                                                 [1, 2, 1]
                                                 ]))

    arr2 = create_alternating_array(height=8, width=7)
    board2 = create_board_from_array(arr2)
    old_board = deepcopy(board2.board)
    # Board shouldn't change
    board2.gravity()
    assert np.array_equal(board2.board, old_board)

    board2.board[0:2, 0] = 0
    board2.board[3:6, 4] = 0
    board2.board[3, 2] = 0
    board2.board[7, :] = 0
    board2.gravity()


def test_gravity_updates_activation_queue():
    arr = create_alternating_array(4, 3)
    board = create_board_from_array(arr)

    activation_queue = []
    
if __name__ == '__main__':
    test_gravity_falls()
