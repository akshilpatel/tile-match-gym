import numpy as np
import random
import pytest

from tile_match_gym.board import Board

def test_resolve_colour_match():
    # No lines
    # Match where the colours are different
    new_board = run_resolve_colour_match(
        [[3, 1, 2, 2],
         [1, 3, 2, 3],
         [3, 1, 1, 2]])
    [print(line) for line in new_board]


    # Single vertical line
    new_board = run_resolve_colour_match(
        np.array([[2, 3, 4, 3],
                  [3, 1, 3, 2],
                  [3, 1, 3, 2],
                  [4, 1, 2, 1]]))

    [print(line) for line in new_board]

    # Single horizontal line
    new_board = run_resolve_colour_match(
        np.array([[2, 3, 3, 4, 3],
                  [3, 2, 4, 3, 2],
                  [4, 1, 1, 1, 3],
                  [3, 4, 2, 3, 2]]))
    [print(line) for line in new_board]




def run_resolve_colour_match(grid, type_grid=None, num_colours=3):
    """
    Helper function to setup a board with a given grid.
    """
    b = Board(num_rows=len(grid), num_cols=len(grid[0]), num_colours=num_colours)
    b.board[0] = np.zeros((b.num_rows, b.num_cols))
    b.board[1] = np.ones_like(b.board[0])
    b.board[0] = np.array(grid)
    if type_grid is not None:
        b.board[1] = np.array(type_grid)

    lines = b.get_colour_lines()
    tile_coords, tile_names, tile_colours = b.process_colour_lines(lines)

    b.resolve_colour_matches(tile_coords, tile_names, tile_colours)

    # return tile_coords, tile_names, tile_colours
    return b.board

if __name__ == "__main__":
    test_resolve_colour_match()
