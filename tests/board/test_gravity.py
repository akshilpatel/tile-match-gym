import pytest
from tests.utils import create_board_from_array, create_alternating_array
import numpy as np

# Test that gravity pushes down tiles
def test_gravity():
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
    assert np.all(board.board == np.array([
        [0, 0, 0],
        [0, 2, 0],
        [2, 1, 2],
        [1, 2, 1]
    ]))


    board.print_board()
    # print(board.board[new_idcs])
if __name__ == '__main__':
    test_gravity()
