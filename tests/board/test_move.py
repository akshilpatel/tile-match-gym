import numpy as np
import random
import pytest

from tile_match_gym.board import Board


def print_board(b):
    for l in b:
        for row in l:
            for c in row:
                print(c, end=" ")
            print()
        print("\n")


def print_boards(expected, actual):
    print("Expected:")
    # [[print(l) for l in p] for p in expected]
    print_board(expected)
    
    # get the indices where the values are different
    idx = np.where(expected != actual)

    # set x to be a copy of actual but with all values converted to strings
    x = actual.astype(str)

    # print actual but with the different values highlighted
    for i, j, k in zip(*idx):
        x[i, j, k] = '\x1b[41m' + str(actual[i, j, k]) + '\x1b[0m'
    print("Actual:")
    print_board(x)
    #[[print(l) for l in p] for p in x]

def test_move():

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
        [[4, 4, 2, 2],
         [2, 1, 2, 3],
         [4, 4, 1, 2]],
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
        ]
    )
    b = Board(3, 4, 4, np_random=np.random.default_rng(10), board=old_board)
    # b.board = old_board

    b.move((1,0), (1,1))
    
    assert np.array_equal(expected_new, b.board), b.board

    b = Board(4, 5, 4, np_random=np.random.default_rng(10))
    b.generate_board()
    b.board[0] = np.array([[4, 1, 2, 1, 4], 
                           [4, 3, 1, 4, 3], 
                           [1, 1, 2, 3, 2], 
                           [4, 1, 2, 3, 4]])
    

    num_eliminations, is_combination_match, num_new_specials, num_activations, shuffled = b.move((2, 2), (1, 2))
    expected_new = np.array(
        [
        [[4, 1, 4, 1, 4], 
         [2, 4, 3, 4, 3], 
         [3, 1, 4, 3, 2], 
         [4, 3, 4, 3, 4]],
        [[1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1]],
        ]
    )
    assert np.array_equal(expected_new, b.board), print_boards(expected_new, b.board)

    assert num_eliminations == 12
    assert not is_combination_match 
    assert num_new_specials == 0
    assert num_activations == 0
    
    
    b = Board(4, 5, 4, np_random=np.random.default_rng(10))
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
    expected_new = np.array(
        [
        [[4, 1, 2, 1, 4], 
         [4, 4, 3, 4, 3], 
         [1, 1, 2, 3, 2], 
         [4, 1, 2, 3, 4]],
        [[1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 2, 1, 1, 1]],
        ]
    )
    assert np.array_equal(expected_new, b.board), print_boards(expected_new, b.board)

    b = Board(4, 5, 4, np_random=np.random.default_rng(10))
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
    expected_new = np.array(
        [
        [[4, 1, 2, 1, 4], 
         [4, 4, 1, 4, 3], 
         [3, 1, 2, 3, 2], 
         [4, 2, 2, 3, 4]],
        [[1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1]],
        ]
    )
    assert np.array_equal(expected_new, b.board), print_boards(expected_new, b.board)
    
    # swap a vertical and horizontal laser
    b = Board(4, 5, 4, np_random=np.random.default_rng(11))
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
    assert num_activations == 2
    assert is_combination_match
    expected_new = np.array(
        [
        [[2, 4, 3, 1, 3], 
         [2, 1, 2, 1, 4], 
         [4, 3, 1, 4, 3], 
         [2, 1, 2, 3, 4]],
        [[1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1]],
        ]
    )
    assert np.array_equal(expected_new, b.board), print_boards(expected_new, b.board)



    # swap two vertical lasers - it should act as if the lasers are different
    b = Board(4, 5, 5, np_random=np.random.default_rng(10))
    b.generate_board()
    b.board[0] = np.array([[4, 4, 2, 1, 4], 
                           [4, 1, 1, 4, 3], 
                           [1, 3, 2, 3, 2], 
                           [4, 1, 2, 3, 4]])
    
    b.board[1] = np.array([[1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 2, 1, 1, 1], 
                           [1, 2, 1, 1, 1]])

    num_eliminations, is_combination_match, num_new_specials, num_activations, shuffled = b.move((2, 1), (3, 1))
    assert num_eliminations == 11
    assert num_new_specials == 0
    assert num_activations == 2
    assert is_combination_match
    expected_new = np.array(
        [
        [[5, 5, 1, 2, 4], 
         [5, 3, 2, 1, 4], 
         [1, 5, 1, 4, 3], 
         [2, 4, 2, 3, 4]],
        [[1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1]],
        ]
    )
    assert np.array_equal(expected_new, b.board), print_boards(expected_new, b.board)

    # swap vertical laser and bomb
    b = Board(5, 5, 5, np_random=np.random.default_rng(10))
    b.generate_board()
    b.board[0] = np.array([[4, 4, 2, 1, 4], 
                           [4, 1, 1, 4, 3], 
                           [3, 1, 1, 4, 3], 
                           [1, 3, 2, 3, 2], 
                           [4, 1, 2, 3, 4]])
    
    b.board[1] = np.array([[1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 2, 1, 1, 1], 
                           [1, 4, 1, 1, 1]])

    num_eliminations, is_combination_match, num_new_specials, num_activations, shuffled = b.move((3, 1), (4, 1))
    expected_new = np.array(
        [
        [[3, 5, 4, 5, 1], 
         [1, 5, 5, 1, 5], 
         [4, 2, 1, 2, 2], 
         [4, 2, 3, 1, 4], 
         [5, 4, 2, 4, 3]],
        [[1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1]],
        ]
    )
    assert is_combination_match
    assert num_activations == 2
    assert np.array_equal(expected_new, b.board), print_boards(expected_new, b.board)

    
    # cookie +
    b = Board(5, 5, 5, np_random=np.random.default_rng(10))
    b.generate_board()
    b.board[0] = np.array([[4, 4, 2, 1, 4], 
                           [4, 1, 3, 4, 3], 
                           [3, 1, 1, 4, 3], 
                           [1, 3, 2, 3, 2], 
                           [4, 1, 2, 3, 4]])
    
    b.board[1] = np.array([[1, 1, -1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1]])

    num_eliminations, is_combination_match, num_new_specials, num_activations, shuffled = b.move((0, 2), (0, 3))
    expected_new = np.array(
        [
        [[3, 5, 4, 5, 4], 
         [4, 1, 1, 4, 3], 
         [4, 5, 3, 4, 3], 
         [3, 4, 2, 3, 2], 
         [4, 3, 2, 3, 4]],
        [[1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1]],
        ]
    )
    assert np.array_equal(expected_new, b.board), print_boards(expected_new, b.board)
    assert is_combination_match
    assert num_activations == 1

    # vertical laser activation in big board
    b = Board(8, 8, 9, np_random=np.random.default_rng(11))
    b.generate_board()
    b.board[0] = np.array([[4, 4, 2, 9, 2, 3, 1, 4], 
                           [5, 3, 1, 8, 7, 1, 4, 3], 
                           [1, 1, 3, 7, 6, 6, 2, 4], 
                           [3, 2, 3, 3, 4, 1, 4, 3], 
                           [4, 2, 1, 9, 3, 2, 2, 7],
                           [3, 4, 1, 3, 1, 1, 4, 3], 
                           [1, 3, 2, 3, 2, 5, 3, 2], 
                           [4, 1, 1, 5, 2, 2, 3, 4]])
                                              
    b.board[1] = np.array([[1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, 2, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1, 1, 1, 1]])
    num_eliminations, is_combination_match, num_new_specials, num_activations, shuffled = b.move((4, 3), (4, 4))
    expected_new = np.array(
        [
        [[4, 4, 2, 1, 2, 3, 1, 4], 
         [5, 3, 1, 1, 7, 1, 4, 3], 
         [1, 1, 3, 9, 6, 6, 2, 4], 
         [3, 2, 3, 9, 4, 1, 4, 3], 
         [4, 2, 1, 3, 9, 2, 2, 7], 
         [3, 4, 1, 4, 1, 1, 4, 3], 
         [1, 3, 2, 3, 2, 5, 3, 2], 
         [4, 1, 1, 3, 2, 2, 3, 4]],
        [[1, 1, 1, 1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1, 1, 1, 1], 
         [1, 1, 1, 1, 1, 1, 1, 1], 
         [1, 1, 1, 2, 1, 1, 1, 1]],
        ]
    )
    assert not is_combination_match
    assert num_new_specials == 1

    assert np.array_equal(expected_new, b.board), print_boards(expected_new, b.board)
