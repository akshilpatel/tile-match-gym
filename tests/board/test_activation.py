import pytest
import numpy as np

from tile_match_gym.board import Board
from tile_match_gym.utils import print_board_diffs
from typing import Optional, List, Tuple

# For each cases, check that normals are deleted in correct range and specials are chained in correct range.
def test_activate_special():
    #### Check coord is deleted.

    # Bomb #
    b = Board(4, 5, 5)
    b.generate_board()
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.generate_board()
    b.board[0] = np.array([[3, 1, 4, 2, 2],
                        [1, 4, 2, 3, 4],
                        [3, 3, 2, 1, 1],
                        [4, 4, 5, 4, 3]]
                        )
    b.board[1] = np.array([[1, 1, 1, 1, 1],
                        [1, 1, 4, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]])

    b.activate_special((1, 2), 4, 2)
    assert np.array_equal(b.board[0], np.array([[3, 0, 0, 0, 2],
                                                [1, 0, 0, 0, 4],
                                                [3, 0, 0, 0, 1],
                                                [4, 4, 5, 4, 3]])), b.board[0]

    assert np.array_equal(b.board[1], np.array([[1, 0, 0, 0, 1],
                                                [1, 0, 0, 0, 1],
                                                [1, 0, 0, 0, 1],
                                                [1, 1, 1, 1, 1]]))


    # Bomb activates horizontal laser.
    b.board[0] = np.array([[3, 1, 4, 2, 2],
                           [1, 4, 2, 3, 4],
                           [3, 3, 2, 1, 1],
                           [4, 4, 5, 4, 3]]   
                           )
    b.board[1] = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 4, 1, 1],
                           [1, 3, 1, 1, 1],
                           [1, 1, 1, 1, 1]])

    b.activate_special((1, 2), 4, 2)
    assert np.array_equal(b.board[0], np.array([[3, 0, 0, 0, 2],
                                                [1, 0, 0, 0, 4],
                                                [0, 0, 0, 0, 0],
                                                [4, 4, 5, 4, 3]])), b.board[0]

    assert np.array_equal(b.board[1], np.array([[1, 0, 0, 0, 1],
                                                [1, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 1]]))

    # Activating near each edge.
    b.board[0] = np.array([[3, 1, 4, 2, 2],
                        [1, 4, 2, 3, 4],
                        [3, 3, 2, 1, 1],
                        [4, 4, 5, 4, 3]]
                        )
    b.board[1] = np.array([[1, 1, 1, 1, 1],
                        [4, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]])

    b.activate_special((1, 0), 4, 1)
    assert np.array_equal(b.board[0], np.array([[0, 0, 4, 2, 2],
                                                [0, 0, 2, 3, 4],
                                                [0, 0, 2, 1, 1],
                                                [4, 4, 5, 4, 3]])), b.board[0]

    assert np.array_equal(b.board[1], np.array([[0, 0, 1, 1, 1],
                                                [0, 0, 1, 1, 1],
                                                [0, 0, 1, 1, 1],
                                                [1, 1, 1, 1, 1]]))


    # Activating near each edge.
    b.board[0] = np.array([[3, 1, 4, 2, 2],
                        [1, 4, 2, 3, 4],
                        [3, 3, 2, 1, 1],
                        [4, 4, 5, 4, 3]]
                        )
    b.board[1] = np.array([[1, 1, 1, 1, 1],
                        [4, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]])

    b.activate_special((1, 0), 4, 1)
    assert np.array_equal(b.board[0], np.array([[0, 0, 4, 2, 2],
                                                [0, 0, 2, 3, 4],
                                                [0, 0, 2, 1, 1],
                                                [4, 4, 5, 4, 3]]))

    assert np.array_equal(b.board[1], np.array([[0, 0, 1, 1, 1],
                                                [0, 0, 1, 1, 1],
                                                [0, 0, 1, 1, 1],
                                                [1, 1, 1, 1, 1]]))


    # Activating near each edge.
    b.board[0] = np.array([[3, 1, 4, 2, 2],
                        [1, 4, 2, 3, 4],
                        [3, 3, 2, 1, 1],
                        [4, 4, 5, 4, 3]]
                        )
    b.board[1] = np.array([[1, 1, 1, 1, 1],
                        [1, 1, 2, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 4]])

    b.activate_special((3, 4), 4, 3)
    assert np.array_equal(b.board[0], np.array([[3, 1, 4, 2, 2],
                                                [1, 4, 2, 3, 4],
                                                [3, 3, 2, 0, 0],
                                                [4, 4, 5, 0, 0]]))

    assert np.array_equal(b.board[1], np.array([[1, 1, 1, 1, 1],
                                                [1, 1, 2, 1, 1],
                                                [1, 1, 1, 0, 0],
                                                [1, 1, 1, 0, 0]]))

    b.activate_special((3, 3), 4, 3)
    assert np.array_equal(b.board[0], np.array([[3, 1, 4, 2, 2],
                                                [1, 4, 2, 3, 4],
                                                [3, 3, 0, 0, 0],
                                                [4, 4, 0, 0, 0]]))

    assert np.array_equal(b.board[1], np.array([[1, 1, 1, 1, 1],
                                                [1, 1, 2, 1, 1],
                                                [1, 1, 0, 0, 0],
                                                [1, 1, 0, 0, 0]]))

    # Smaller board.
    b = Board(3, 3, 3)
    b.generate_board()
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.activate_special((1,1), 4, 3)
    assert np.array_equal(b.board[0], np.zeros((3,3)))
    assert np.array_equal(b.board[1], np.zeros((3,3)))


    ########### V Laser #############
    b = Board(7, 4, 5)
    b.generate_board()
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[4, 4, 3, 3],
                        [2, 4, 5, 4],
                        [5, 1, 2, 2],
                        [4, 3, 1, 4],
                        [2, 5, 3, 1],
                        [3, 2, 2, 4],
                        [1, 3, 5, 1]])

    b.board[1] = np.array([[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 2, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]])

    b.activate_special((3, 1), 2, 1)


    assert np.array_equal(b.board[0], np.array([[4, 0, 3, 3],
                                                [2, 0, 5, 4],
                                                [5, 0, 2, 2],
                                                [4, 0, 1, 4],
                                                [2, 0, 3, 1],
                                                [3, 0, 2, 4],
                                                [1, 0, 5, 1]]))

    assert np.array_equal(b.board[1], np.array([[1, 0, 1, 1],
                                                [1, 0, 1, 1],
                                                [1, 0, 1, 1],
                                                [1, 0, 1, 1],
                                                [1, 0, 1, 1],
                                                [1, 0, 1, 1],
                                                [1, 0, 1, 1]]))

    # Vertical laser activates cookie
    b.board[0] = np.array([[4, 4, 3, 3],
                           [2, 4, 5, 4],
                           [5, 1, 2, 2],
                           [4, 3, 1, 4],
                           [2, 5, 3, 1],
                           [3, 2, 4, 4],
                           [1, 0, 5, 1]])

    b.board[1] = np.array([[1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 2, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, -1, 1, 1]])


    b.activate_special((3, 1), 2, 1)
    assert np.array_equal(b.board[0], np.array([[0, 0, 3, 3],
                                                [2, 0, 5, 0],
                                                [5, 0, 2, 2],
                                                [0, 0, 1, 0],
                                                [2, 0, 3, 1],
                                                [3, 0, 0, 0],
                                                [1, 0, 5, 1]])), b.board[0]

    assert np.array_equal(b.board[1], np.array([[0, 0, 1, 1],
                                                [1, 0, 1, 0],
                                                [1, 0, 1, 1],
                                                [0, 0, 1, 0],
                                                [1, 0, 1, 1],
                                                [1, 0, 0, 0],
                                                [1, 0, 1, 1]]))


    b.board[0] = np.array([[4, 4, 3, 3],
                        [2, 4, 5, 4],
                        [5, 1, 2, 2],
                        [4, 3, 1, 4],
                        [2, 5, 3, 1],
                        [3, 2, 2, 4],
                        [1, 3, 5, 1]])

    b.board[1] = np.array([[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [2, 1, 1, 1]])

    b.activate_special((5, 0), 2, 1)


    assert np.array_equal(b.board[0], np.array([[0, 4, 3, 3],
                                                [0, 4, 5, 4],
                                                [0, 1, 2, 2],
                                                [0, 3, 1, 4],
                                                [0, 5, 3, 1],
                                                [0, 2, 2, 4],
                                                [0, 3, 5, 1]]))

    assert np.array_equal(b.board[1], np.array([[0, 1, 1, 1],
                                                [0, 1, 1, 1],
                                                [0, 1, 1, 1],
                                                [0, 1, 1, 1],
                                                [0, 1, 1, 1],
                                                [0, 1, 1, 1],
                                                [0, 1, 1, 1]]))


    b.board[0] = np.array([[4, 4, 3, 3],
                        [2, 4, 5, 4],
                        [5, 1, 2, 2],
                        [4, 3, 1, 4],
                        [2, 5, 3, 1],
                        [3, 2, 2, 4],
                        [1, 3, 5, 1]])

    b.board[1] = np.array([[1, 1, 1, 1],
                        [1, 1, 2, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]])

    b.activate_special((1, 2), 2, 5)
    assert np.array_equal(b.board[0], np.array([[4, 4, 0, 3],
                                                [2, 4, 0, 4],
                                                [5, 1, 0, 2],
                                                [4, 3, 0, 4],
                                                [2, 5, 0, 1],
                                                [3, 2, 0, 4],
                                                [1, 3, 0, 1]]))

    assert np.array_equal(b.board[1], np.array([[1, 1, 0, 1],
                                                [1, 1, 0, 1],
                                                [1, 1, 0, 1],
                                                [1, 1, 0, 1],
                                                [1, 1, 0, 1],
                                                [1, 1, 0, 1],
                                                [1, 1, 0, 1]]))

    # H Laser
    b = Board(4, 9, 7)
    b.generate_board()
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[6, 6, 7, 2, 7, 5, 5, 2, 3],
                        [7, 5, 6, 5, 5, 4, 7, 4, 2],
                        [7, 4, 4, 5, 1, 5, 2, 3, 3],
                        [5, 3, 4, 7, 5, 1, 5, 3, 6]])

    b.board[1] = np.array([[1, 3, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1]])


    b.activate_special((0, 1), 3, 6)

    assert np.array_equal(b.board[0], np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [7, 5, 6, 5, 5, 4, 7, 4, 2],
                                                [7, 4, 4, 5, 1, 5, 2, 3, 3],
                                                [5, 3, 4, 7, 5, 1, 5, 3, 6]]))

    assert np.array_equal(b.board[1], np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [1, 1, 1, 1, 1, 1, 1, 1, 1]]))


    b.board[0] = np.array([[6, 6, 7, 2, 7, 5, 5, 2, 3],
                        [7, 5, 6, 5, 5, 4, 7, 4, 2],
                        [7, 4, 4, 5, 1, 5, 2, 3, 3],
                        [5, 3, 4, 7, 5, 1, 5, 3, 6]])

    b.board[1] = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 3],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1]])


    b.activate_special((2, 8), 3, 3)

    assert np.array_equal(b.board[0], np.array([[6, 6, 7, 2, 7, 5, 5, 2, 3],
                                                [7, 5, 6, 5, 5, 4, 7, 4, 2],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [5, 3, 4, 7, 5, 1, 5, 3, 6]]))

    assert np.array_equal(b.board[1], np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 1, 1, 1, 1, 1]]))


    # Horizontal laser activates bomb activates another horizontal laser.

    b.board[0] = np.array([[6, 6, 7, 2, 7, 5, 5, 2, 3],
                           [7, 5, 6, 5, 5, 4, 7, 4, 2],
                           [7, 4, 4, 5, 1, 5, 2, 3, 3],
                           [5, 3, 4, 7, 5, 1, 5, 3, 6]])

    b.board[1] = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 3, 1, 1, 1, 1, 1, 1],
                           [1, 1, 4, 1, 1, 1, 1, 1, 3],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1]])

    b.activate_special((2, 8), 3, 3)

    assert np.array_equal(b.board[0], np.array([[6, 6, 7, 2, 7, 5, 5, 2, 3],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [5, 0, 0, 0, 5, 1, 5, 3, 6]]))

    assert np.array_equal(b.board[1], np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 1, 1, 1, 1, 1]]))

    # Cookie
    b = Board(4, 3, 4)
    b.generate_board()
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[1, 4, 4],
                           [4, 0, 2],
                           [4, 1, 3],
                           [3, 3, 1]])

    b.board[1] = np.array([[1, 1, 1],
                           [1, -1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])


    b.activate_special((1, 1), -1, 0)
    assert np.array_equal(b.board[0], np.array([[1, 0, 0],
                                                [0, 0, 2],
                                                [0, 1, 3],
                                                [3, 3, 1]]))

    assert np.array_equal(b.board[1], np.array([[1, 0, 0],
                                                [0, 0, 1],
                                                [0, 1, 1],   
                                                [1, 1, 1]]))

    b = Board(4, 3, 4)
    b.generate_board()
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[1, 3, 4],
                           [3, 0, 2],
                           [4, 1, 3],
                           [2, 3, 1]])

    b.board[1] = np.array([[1, 1, 1],
                           [1, -1, 1],
                           [1, 1, 1],
                           [1, 3, 1]])

    # Cookie activates horizontal special.
    b.activate_special((1, 1), -1, 0)
    assert np.array_equal(b.board[0], np.array([[1, 0, 4],
                                                [0, 0, 2],
                                                [4, 1, 0],   
                                                [0, 0, 0]])), b.board[0]

    assert np.array_equal(b.board[1], np.array([[1, 0, 1],
                                                [0, 0, 1],
                                                [1, 1, 0], 
                                                [0, 0, 0]]))

    # Try to activate a cookie when the board is empty
    b = Board(4, 3, 4)
    b.generate_board()
    b.num_specials_activated = 0
    b.num_new_specials = 0

    b.board = np.zeros((2, 4, 3))
    b.activate_special((1, 1), -1, 0, False)
    assert np.array_equal(b.board, np.zeros((2, 4, 3)))


def tests_get_special_creation_pos():
    """
    Test the special creation is in the correct position.
    get_special_creation_pos function
    """
    special_position = get_special_pos([[0, 1, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 1, 0, 0]])
    assert all(special_position == np.array([1, 1])), f"Special position is not correct. Expected: {np.array([1, 1])}, got: {special_position}"
    # get the coords where the board is 1


    # Match of even length.
    special_position = get_special_pos([[0, 1, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 1, 0, 0]])
    assert all(special_position == np.array([1, 1])), f"Special position is not correct. Expected: {np.array([1, 1])}, got: {special_position}"

    # Match of length 5
    special_position = get_special_pos([[0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0]])
    assert all(special_position == np.array([2, 1])), f"Special position is not correct. Expected: {np.array([2, 1])}, got: {special_position}"

    # Match corner
    special_position = get_special_pos([[0, 0, 0, 0, 0],
                                        [0, 1, 1, 1, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0]], straight=False)
    assert all(special_position == np.array([1, 1])), f"Special position is not correct. Expected: {np.array([1, 1])}, got: {special_position}"

    # Match flipped corner
    special_position = get_special_pos([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 1, 1, 1, 0],
                                        [0, 0, 0, 0, 0]], straight=False)
    assert all(special_position == np.array([3, 3])), f"Special position is not correct. Expected: {np.array([3, 3])}, got: {special_position}"

    # Match where middle is special.
    special_position = get_special_pos([[0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0]],
                                       [
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 2, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]])
    assert all(special_position == np.array([2, 1])), f"Special position is not correct. Expected: {np.array([2, 1])}, got: {special_position}"

    # Match where non-middle is special.
    special_position = get_special_pos([[0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0]],
                                       [
                                        [0, 0, 0, 0, 0],
                                        [0, 2, 0, 0, 0],
                                        [0, 2, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]])
    assert all(special_position == np.array([2, 1])), f"Special position is not correct. Expected: {np.array([2, 1])}, got: {special_position}"


    # Match where a lot of the middle is special.
    special_position = get_special_pos([[0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 1, 0, 0, 0]],
                                       [
                                        [0, 0, 0, 0, 0],
                                        [0, 2, 0, 0, 0],
                                        [0, 2, 0, 0, 0],
                                        [0, 2, 0, 0, 0],
                                        [0, 0, 0, 0, 0]])
    assert all(special_position == np.array([2, 1])), f"Special position is not correct. Expected: {np.array([2, 1])}, got: {special_position}"

    # already special in corner
    special_position = get_special_pos([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 1, 1, 1, 0],
                                        [0, 0, 0, 0, 0]],
                                       [
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 2, 0],
                                        [0, 0, 0, 0, 0]], straight=False)
    assert all(special_position == np.array([3, 3])), f"Special position is not correct. Expected: {np.array([3, 3])}, got: {special_position}"

    # Match where the colours are different
    special_position = get_special_pos([[3, 2, 5, 6, 7],
                                        [4, 2, 4, 5, 6],
                                        [5, 2, 3, 4, 5],
                                        [6, 2, 4, 3, 4],
                                        [7, 6, 5, 4, 3]], expected_color=2)
    assert all(special_position == np.array([1, 1])), f"Special position is not correct. Expected: {np.array([1, 1])}, got: {special_position}"



def get_special_pos(grid, type_grid=None, num_colours=3, straight=True,
                    expected_color=1):
    """
    Helper function to setup a board with a given grid.
    """
    b = Board(num_rows=len(grid), num_cols=len(grid[0]), num_colours=num_colours)
    b.generate_board()
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.zeros((b.num_rows, b.num_cols))
    b.board[1] = np.ones((b.num_rows, b.num_cols))
    b.board[0] = grid
    if type_grid is not None:
        b.board[1] = type_grid
    coords = np.argwhere(b.board[0] == expected_color)
    coords = [tuple(c.tolist()) for c in coords]
    taken_pos = set()
    special_position = b.get_special_creation_pos(coords, taken_pos=taken_pos, straight_match=straight)
    return special_position
