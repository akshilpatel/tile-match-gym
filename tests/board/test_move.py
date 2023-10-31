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
    new_board, num_eliminations, is_combination_match, num_new_specials, num_activations = run_move(old_board, (1,0), (1,1))
    print("new_board = ", new_board)
    print("expected_new = ", expected_new)
    assert np.array_equal(expected_new, new_board), print_board(new_board)

    b = Board(4, 5, 4, seed=10)
    b.generate_board()
    b.board[0] = np.array([[4, 1, 2, 1, 4], 
                           [4, 3, 1, 4, 3], 
                           [1, 1, 2, 3, 2], 
                           [4, 1, 2, 3, 4]])
    

    num_eliminations, is_combination_match, num_new_specials, num_activations, shuffled = b.move((2, 2), (1, 2))

    assert num_eliminations == 12
    assert not is_combination_match 
    assert num_new_specials == 0
    assert num_activations == 0
    
    
    b = Board(4, 5, 4, seed=10)
    b.generate_board()
    b.board[0] = np.array([[4, 1, 2, 1, 4], 
                           [4, 3, 1, 4, 3], 
                           [1, 1, 2, 3, 2], 
                           [4, 1, 2, 3, 4]])
    
    b.board[1] = np.array([[1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1]])
    
    num_eliminations, is_combination_match, num_new_specials, num_activations, shuffled = b.move((1, 1), (1, 2))
    assert num_eliminations == 4
    assert num_new_specials == 1
    assert num_activations == 0
    assert not is_combination_match

    b = Board(4, 5, 4, seed=10)
    b.generate_board()
    b.board[0] = np.array([[4, 4, 2, 1, 4], 
                           [4, 1, 1, 4, 3], 
                           [1, 3, 2, 3, 2], 
                           [4, 1, 2, 3, 4]])
    
    b.board[1] = np.array([[1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 2, 1, 1, 1]])

    num_eliminations, is_combination_match, num_new_specials, num_activations, shuffled = b.move((2, 0), (2, 1))
    assert num_eliminations == 4
    assert num_new_specials == 0
    assert num_activations == 1
    assert not is_combination_match

    b = Board(4, 5, 4, seed=11)
    b.generate_board()
    b.board[0] = np.array([[4, 4, 2, 1, 4], 
                           [4, 1, 1, 4, 3], 
                           [1, 3, 2, 3, 2], 
                           [4, 1, 2, 3, 4]])
    
    b.board[1] = np.array([[1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 3, 1, 1, 1], 
                           [1, 2, 1, 1, 1]])

    num_eliminations, is_combination_match, num_new_specials, num_activations, shuffled = b.move((2, 1), (3, 1))
    assert num_eliminations == 11
    assert num_new_specials == 0
    assert num_activations == 0
    assert is_combination_match



def run_move(grid, coord1, coord2, num_colours=4):
    """
    Helper function to setup a board with a given grid.
    """
    print("\n&&&&&&&&&&&&\nBoard in run_move = ", grid)
    print(f"num_rows = {len(grid[0])}, num_cols = {len(grid[0][0])}")
    b = Board(num_rows=len(grid), num_cols=len(grid[0]),
              num_colours=num_colours, board=grid)
    
    print(f"Board after init = {b.board} with shape {b.board.shape} and b.num_rows = {b.num_rows}, b.num_cols = {b.num_cols}\n&&&&&&&&&&&&\n")
    
    num_eliminations, is_combination_match, num_new_specials, num_activations, shuffled = b.move(coord1, coord2)

    # return tile_coords, tile_names, tile_colours
    return b.board, num_eliminations, is_combination_match, num_new_specials, num_activations

if __name__ == "__main__":
    test_move()
