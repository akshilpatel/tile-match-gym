import numpy as np
import random
import pytest

from tile_match_gym.board import Board

def test_move():

    random.seed(0)
    np.random.seed(0)

    print_board = lambda b: [[print(l) for l in p] for p in b]

    # Single vertical line
    old_board = np.array(
        [
        [[3, 1, 2, 2],
         [1, 3, 2, 3],
         [3, 1, 1, 2]],
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
        ]
    )
    expected_new = np.array(
        [
        [[2, 2, 3, 2],
         [1, 2, 2, 3],
         [2, 4, 1, 2]],
        [[1, 3, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
        ]
    )
    new_board = run_move(old_board, (1,0), (1,1))
    print("new_board = ", new_board)
    print("expected_new = ", expected_new)
    assert np.array_equal(expected_new, new_board), print_board(new_board)



def run_move(grid, coord1, coord2, num_colours=4):
    """
    Helper function to setup a board with a given grid.
    """
    print("\n&&&&&&&&&&&&\nBoard in run_move = ", grid)
    print(f"num_rows = {len(grid[0])}, num_cols = {len(grid[0][0])}")
    b = Board(num_rows=len(grid), num_cols=len(grid[0]),
              num_colours=num_colours, board=grid)
    
    print(f"Board after init = {b.board} with shape {b.board.shape} and b.num_rows = {b.num_rows}, b.num_cols = {b.num_cols}\n&&&&&&&&&&&&\n")
    
    b.move(coord1, coord2)

    # return tile_coords, tile_names, tile_colours
    return b.board

if __name__ == "__main__":
    test_move()
