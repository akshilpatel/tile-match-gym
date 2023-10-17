import pytest
import numpy as np

from tile_match_gym.board import Board
from tile_match_gym.utils import print_board_diffs
from typing import Optional, List, Tuple

# def test_activate_special():
#     # Bomb
# 
#     # V Laser
# 
#     # H Laser
# 
#     # Cookie
# 
#     # Adding to queue.
# 
#     # Test that a special can't be readded to queue after activation due to another tile activation
# 
# 
#     # Test case that cookie is activated but surrounded already by colourless specials - current code does not work.
#     assert False

def test_get_special_creation_pos():
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
    assert all(special_position == np.array([1, 1])), f"Special position is not correct. Expected: {np.array([1, 1])}, got: {special_position}"

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
    assert all(special_position == np.array([3, 1])), f"Special position is not correct. Expected: {np.array([3, 1])}, got: {special_position}"


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
    assert all(special_position == np.array([0, 1])), f"Special position is not correct. Expected: {np.array([0, 1])}, got: {special_position}"

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
    assert all(special_position == np.array([2, 3])), f"Special position is not correct. Expected: {np.array([2, 3])}, got: {special_position}"

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
    b.board[0] = np.zeros((b.num_rows, b.num_cols))
    b.board[1] = np.zeros((b.num_rows, b.num_cols))
    b.board[0] = grid
    if type_grid is not None:
        b.board[1] = type_grid
    coords = np.argwhere(b.board[0] == expected_color)
    special_position = b.get_special_creation_pos(coords, straight)
    return special_position
