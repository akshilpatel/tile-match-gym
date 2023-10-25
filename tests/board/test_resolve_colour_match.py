import numpy as np
import random
import pytest

from tile_match_gym.board import Board

def print_board(b):
    for l in b[0]:
        print(l)
    print()
    for l in b[1]:
        print(l)

def print_boards(b1,b2):
    print("expected:")
    print_board(b1)
    print("got:")
    print_board(b2)
    print("###")

def test_resolve_colour_match():

    # No lines
    # Match where the colours are different
    new_board = run_resolve_colour_match(
        [[3, 1, 2, 2],
         [1, 3, 2, 3],
         [3, 1, 1, 2]]
        )
    assert np.array_equal(new_board, np.array(
        [
        [[3, 1, 2, 2],
         [1, 3, 2, 3],
         [3, 1, 1, 2]],
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
        ]
        )), print_board(new_board)


    # Single vertical line
    new_board = run_resolve_colour_match(
        np.array([[2, 3, 4, 3],
                  [3, 1, 3, 2],
                  [3, 1, 3, 2],
                  [4, 1, 2, 1]]))
    expected = np.array(
        [
        [[2, 3, 4, 3],
         [3, 0, 3, 2],
         [3, 0, 3, 2],
         [4, 0, 2, 1]],
        [[1, 1, 1, 1],
         [1, 0, 1, 1],
         [1, 0, 1, 1],
         [1, 0, 1, 1]],
        ]
        )
    assert np.array_equal(new_board, expected), print_boards(expected, new_board)

    # Single horizontal line
    new_board = run_resolve_colour_match(
        np.array([[2, 3, 3, 4, 3],
                  [3, 2, 4, 3, 2],
                  [4, 1, 1, 1, 3],
                  [3, 4, 2, 3, 2]]))
    assert np.array_equal(new_board, np.array(
        [
        [[2, 3, 3, 4, 3],
         [3, 2, 4, 3, 2],
         [4, 0, 0, 0, 3],
         [3, 4, 2, 3, 2]],
        [[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 0, 0, 0, 1],
         [1, 1, 1, 1, 1]]
        ]
        )), new_board


    # T 
    new_board = run_resolve_colour_match(
        np.array([[2, 3, 3, 4, 3],
                  [4, 1, 1, 1, 3],
                  [3, 2, 1, 3, 2],
                  [3, 4, 1, 3, 2]]))
    assert np.array_equal(new_board, np.array(
        [
        [[2, 3, 3, 4, 3],
         [4, 0, 1, 0, 3],
         [3, 2, 0, 3, 2],
         [3, 4, 0, 3, 2]],
        [[1, 1, 1, 1, 1],
         [1, 0, 4, 0, 1],
         [1, 1, 0, 1, 1],
         [1, 1, 0, 1, 1]]
        ]
        )), new_board

    # L

    # Disjoint lines should not be merged.
    new_board = run_resolve_colour_match(
        np.array([[2, 3, 3, 2, 3],
                  [4, 1, 4, 1, 3],
                  [3, 1, 3, 1, 2],
                  [3, 1, 4, 1, 2]]))
    assert np.array_equal(new_board, np.array(
        [
           [[2, 3, 3, 2, 3],
            [4, 0, 4, 0, 3],
            [3, 0, 3, 0, 2],
            [3, 0, 4, 0, 2]],
           [[1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1]]
           ]
        )), new_board
    # Lines > 3.
    new_board = run_resolve_colour_match(
        np.array([[2, 1, 3, 2, 3],
                  [4, 1, 4, 1, 3],
                  [3, 1, 3, 2, 2],
                  [3, 1, 4, 1, 2]]))
    assert np.array_equal(new_board, np.array(
        [
            [[2, 0, 3, 2, 3],
             [4, 1, 4, 1, 3],
             [3, 0, 3, 2, 2],
             [3, 0, 4, 1, 2]],
            [[1, 0, 1, 1, 1],
             [1, 2, 1, 1, 1],
             [1, 0, 1, 1, 1],
             [1, 0, 1, 1, 1]]
        ]
    )), new_board



    # Lines > 3 where the board config doesn't include the corresponding special.
    new_board = run_resolve_colour_match(
        np.array([[2, 1, 3, 2, 3],
                  [4, 1, 4, 1, 3],
                  [2, 1, 3, 2, 2],
                  [4, 1, 4, 1, 3],
                  [3, 1, 3, 2, 2],
                  [3, 1, 4, 1, 2]]))
    assert np.array_equal(new_board, np.array([
        [[2, 0, 3, 2, 3],
         [4, 0, 4, 1, 3],
         [2, 0, 3, 2, 2],
         [4, 0, 4, 1, 3],
         [3, 0, 3, 2, 2],
         [3, 1, 4, 1, 2]],
        [[1, 0, 1, 1, 1],
         [1, 0, 1, 1, 1],
         [1, -1, 1, 1, 1],
         [1, 0, 1, 1, 1],
         [1, 0, 1, 1, 1],
         [1, 1, 1, 1, 1]
        ]
        ]
    )), new_board

    new_board = run_resolve_colour_match(
        np.array(
            [[3, 1, 2, 2],
             [3, 1, 2, 3],
             [3, 1, 1, 2]]
        ))
    assert np.array_equal(new_board, np.array([
            [[0, 0, 2, 2],
             [0, 0, 2, 3],
             [0, 0, 1, 2]],
            [[0, 0, 1, 1],
             [0, 0, 1, 1],
             [0, 0, 1, 1]]
        ]))




def run_resolve_colour_match(grid, type_grid=None, num_colours=3):
    """
    Helper function to setup a board with a given grid.
    """
    # b = Board(num_rows=len(grid), num_cols=len(grid[0]), num_colours=num_colours)
    # b.board[0] = np.zeros((b.num_rows, b.num_cols))
    # b.board[1] = np.ones_like(b.board[0])
    # b.board[0] = np.array(grid)
    b = Board(num_rows=len(grid), num_cols=len(grid[0]),
              num_colours=num_colours, board=grid)
    # if type_grid is not None:
    #     b.board[1] = np.array(type_grid)

    lines = b.get_colour_lines()
    tile_coords, tile_names, tile_colours = b.process_colour_lines(lines)

    b.resolve_colour_matches(tile_coords, tile_names, tile_colours)

    # return tile_coords, tile_names, tile_colours
    return b.board

if __name__ == "__main__":
    test_resolve_colour_match()
