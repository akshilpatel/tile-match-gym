import numpy as np

from tests.utils import create_alternating_board, wipe_coords
from copy import deepcopy
from tile_match_gym.board import Board

# Test that gravity pushes down tiles
def test_gravity():
    # Case 1
    board = Board(4, 3, 5)
    board.generate_board()
    board.board[0] =  np.array([[4, 1, 2], 
                                [3, 2, 3], 
                                [2, 4, 4], 
                                [2, 2, 1]])

    wipe_coords(board, [(0, 0), (2, 0), (2, 2), (2, 1), (3, 2)])

    board.gravity()
    print(board.board)
    assert np.array_equal(board.board[0], np.array([[0, 0, 0], 
                                                    [0, 1, 0], 
                                                    [3, 2, 2],
                                                    [2, 2, 3]])), board.board[0]
    assert np.array_equal(board.board[1], np.array([[0, 0, 0], 
                                                    [0, 1, 0], 
                                                    [1, 1, 1],
                                                    [1, 1, 1]]))

    # Case 2
    board = Board(8, 7, 5)
    board.generate_board()
    board.board[0] = np.array([[2, 1, 1, 2, 3, 3, 2],
                                [1, 3, 2, 3, 4, 3, 2],
                                [4, 3, 3, 2, 4, 2, 3],
                                [2, 4, 3, 3, 3, 3, 2],
                                [3, 4, 2, 2, 2, 2, 3],
                                [2, 1, 2, 3, 3, 3, 2],
                                [3, 2, 3, 2, 2, 2, 3],
                                [2, 4, 2, 3, 2, 3, 2]])

    board.board[1] = np.array([[1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1],
                                [2, 1, 1, 1, 1, 1, 1],
                                [3, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 3, 1, 4],
                                [-1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 3, 1, 1],
                                [1, 3, 1, 1, 2, 1, 2]])

    old_board = deepcopy(board.board)
    board.gravity()

    # No change
    assert np.array_equal(board.board, old_board)

    wipe_coords(board, [(0, 0), (1, 0), (3, 4), (4, 4), (5, 4), (3, 2), *[(7, i) for i in range(7)]])
    board.gravity()
    assert np.array_equal(
        board.board[0],
        np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 2, 0, 3, 2],
                    [0, 3, 1, 3, 0, 3, 2],
                    [4, 3, 2, 2, 0, 2, 3],
                    [2, 4, 3, 3, 3, 3, 2],
                    [3, 4, 2, 2, 4, 2, 3],
                    [2, 1, 2, 3, 4, 3, 2],
                    [3, 2, 3, 2, 2, 2, 3]]))

    assert np.array_equal( board.board[1], np.array([[0, 0, 0, 0, 0, 0, 0],
                                                    [0, 1, 0, 1, 0, 1, 1],
                                                    [0, 1, 1, 1, 0, 1, 1],
                                                    [2, 1, 1, 1, 0, 1, 1],
                                                    [3, 1, 1, 1, 1, 1, 1],
                                                    [1, 1, 1, 1, 1, 1, 4],
                                                    [-1, 1, 1, 1, 1, 1, 1],
                                                    [1, 1, 1, 1, 3, 1, 1]])), board.board[1]
    
