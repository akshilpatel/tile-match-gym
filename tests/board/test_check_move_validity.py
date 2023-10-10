import numpy as np
from tile_match_gym.board import Board
from copy import deepcopy


def test_check_move_validity():
    # 3 in a row and 3 in a col
    board = Board(10, 10, 5) 
    board.board[0] = np.array([[4, 4, 5, 4, 2, 2, 6, 6, 3, 3],
                                [3, 6, 3, 6, 3, 4, 5, 4, 2, 2],
                                [6, 5, 6, 4, 6, 3, 4, 5, 2, 3],
                                [2, 4, 6, 2, 3, 4, 6, 3, 4, 3],
                                [2, 5, 2, 3, 4, 4, 2, 6, 5, 6],
                                [4, 5, 3, 4, 6, 3, 5, 2, 5, 6],
                                [4, 4, 3, 2, 4, 5, 4, 5, 3, 5],
                                [5, 6, 4, 2, 5, 4, 6, 4, 3, 2],
                                [4, 5, 5, 6, 3, 4, 6, 3, 5, 6],
                                [4, 4, 5, 4, 6, 5, 2, 2, 4, 6]])

    old_board = deepcopy(board.board)
    assert board.check_move_validity((1,1), (1, 2))
    assert board.check_move_validity((1,1), (2, 1))
    assert np.array_equal(board.board, old_board)

    # 3 in a col
    board = Board(4, 3, 4)
    board.board[0] = np.array([[2, 4, 3],
                            [2, 1, 3],
                            [1, 3, 1],
                            [1, 4, 1]])

    old_board = deepcopy(board.board)
    assert board.check_move_validity((2, 1), (2, 2))
    assert np.array_equal(board.board, old_board)
    assert board.check_move_validity((1, 1), (1, 0))
    assert np.array_equal(board.board, old_board)
    assert board.check_move_validity((1, 1), (1, 2))
    assert np.array_equal(board.board, old_board)

    # 3 in a col with special.
    board.board[1] = np.array([[1, 1, 1],
                            [2, 1, 3],
                            [1, 1, 1],
                            [1, 1, 2]])

    old_board = deepcopy(board.board)
    assert board.check_move_validity((2, 1), (2, 2))
    assert np.array_equal(board.board, old_board)


    # 4 in a row with special
    board = Board(4, 5, 5)
    board.board[0] = np.array([[1, 2, 2, 3, 4],
                            [1, 2, 1, 2, 4],
                            [4, 4, 1, 4, 2],
                            [1, 1, 4, 3, 2]]) 

    board.board[1] = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 3, 1, 1, 1],
                            [1, 1, 2, 1, 1]]) 
    old_board = deepcopy(board.board)
    assert board.check_move_validity((2, 2), (3, 2))
    assert np.array_equal(board.board, old_board)

    # 4 in a column
    board.board[0] = np.array([[1, 2, 2, 3, 4],
                            [1, 2, 1, 2, 4],
                            [4, 4, 2, 4, 2],
                            [1, 2, 3, 3, 2]]) 

    board.board[1] = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 3, 1, 1, 1],
                            [1, 3, 2, 1, 1]]) 

    old_board = deepcopy(board.board)
    assert board.check_move_validity((2, 1), (2, 2))
    assert np.array_equal(board.board, old_board)

    # 5 in a col
    board = Board(5, 5, 7, seed=2)  # cookie , and vertical lase, horizontal laser, bomb

    board.board[0] = np.array([[1, 2, 3, 3, 4],
                            [1, 2, 3, 2, 4],
                            [4, 4, 2, 3, 2],
                            [1, 2, 3, 2, 2],
                            [1, 2, 3, 1, 2]]) 

    board.board[1] = np.array([[1, 1, 1, 2, 1],
                            [1, 1, 1, 1, 1],
                            [1, 3, 1, 1, 1],
                            [1, 3, 2, 1, 1],
                            [1, 2, 3, 3, 2]])
    old_board = deepcopy(board.board)
    assert board.check_move_validity((2, 3), (2, 2))
    assert np.array_equal(board.board, old_board)

    # Cookie + special
    board = Board(7, 5, 12, seed=3)
    print(board.board)
    board.board[0] = np.array([[8, 2, 9, 4, 5], 
                            [1, 9, 10, 6, 8], 
                            [4, 6, 11, 4, 9], 
                            [12, 11, 3, 4, 6], 
                            [8, 10, 8, 7, 8], 
                            [12, 1, 6, 6, 2], 
                            [4, 1, 12, 9, 12]])

    board.board[1] = np.array([[1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 3, 1, 1], 
                            [1, 1, -1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1]])
    board.board[0, 4, 2] = 0
    old_board = deepcopy(board.board)
    assert board.check_move_validity((3, 2), (4, 2))
    assert np.array_equal(board.board, old_board)


    # Cookie + normal
    board.board[0, 4, 2] = 8
    board.board[0, 0, 0] = 0
    board.board[1] = np.array([[-1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1]])

    old_board = deepcopy(board.board)
    assert board.check_move_validity((0, 0), (0, 1))
    assert board.check_move_validity((0, 0), (1, 0))
    assert np.array_equal(board.board, old_board)


    # Cookie + cookie
    board.board[0, 0, 0] = 8
    board.board[0, 6, 4] = 0
    board.board[0, 6, 3] = 0

    board.board[1] = np.array([[-1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, -1, -1]])

    old_board = deepcopy(board.board)
    assert board.check_move_validity((6, 4), (6, 3))
    assert np.array_equal(board.board, old_board)


    # Not adjacent
    assert not board.check_move_validity((0, 0), (0, 2))
    assert not board.check_move_validity((2, 3), (3, 4))
    assert not board.check_move_validity((0, 0), (6, 5))

    # Same coordinate
    assert not board.check_move_validity((0, 0), (0, 0))
    assert not board.check_move_validity((3, 3), (3, 3))
