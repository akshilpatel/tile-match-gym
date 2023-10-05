import numpy as np
import pytest

from tile_match_gym.board import Board



def test_get_colour_lines():
    b = Board(num_rows=3, num_cols=4, num_colours=3)

    # Colourless specials should not be included
    b.board = np.array([[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]])
    assert b._get_colour_lines() == []

    b.board = np.array([[1, 4, 4, 1],
                        [1, 1, 1, 1],
                        [1, 4, 4, 1]])
    assert b._get_colour_lines() == []

    # No lines
    b.board = np.array([[4, 3, 4, 3],
                        [5, 4, 5, 4], 
                        [4, 3, 4, 2]])

    assert b._get_colour_lines() == [], b._get_colour_lines()

    # Single vertical line on top edge
    b.board[1, 0] = 4
    assert [(0, 0), (1, 0), (2, 0)] in b._get_colour_lines()
    assert len(b._get_colour_lines()) == 1

    # Single horizontal line on bottom edge.
    b.board[1, 0] = 2
    b.board[2, 1] = 4
    assert  [(2, 0), (2, 1), (2, 2)] in b._get_colour_lines(), b._get_colour_lines()
    assert len(b._get_colour_lines()) == 1

    # Different board shape.
    b2 = Board(num_rows=5, num_cols=3, num_colours=4)
    b2.board = np.array([[3, 4, 4], 
                         [5, 5, 2], 
                         [4, 4, 3], 
                         [5, 6, 2], 
                         [3, 5, 2]])
    assert b2._get_colour_lines() == []

    # Single horizontal line on left and top edge
    b2.board[0, 0] = 4
    assert [(0, 0), (0, 1), (0, 2)] in b2._get_colour_lines()
    assert len(b2._get_colour_lines()) == 1

    # Single horizontal line on bottom edge
    b2.board[0, 2] = 6
    b2.board[4, 0] = 5
    b2.board[4, 2] = 5
    assert [(4, 0), (4, 1), (4, 2)] in b2._get_colour_lines()
    assert len(b2._get_colour_lines()) == 1

    # Two horizontal lines on different lines
    b2.board = np.array([[3, 4, 4], 
                         [5, 5, 5], 
                         [4, 4, 3], 
                         [3, 3, 3], 
                         [3, 5, 2]])

    assert len(b2._get_colour_lines()) == 1, b2._get_colour_lines()
    assert [(3, 0), (3, 1), (3, 2)] in b2._get_colour_lines()

    # Separate horizontal and vertical lines on separate rows.
    b2.board = np.array([[3, 4, 4], 
                         [5, 4, 5], 
                         [4, 4, 3], 
                         [3, 3, 3], 
                         [3, 5, 2]])

    assert len(b2._get_colour_lines()) == 1
    assert  [(3, 0), (3, 1), (3, 2)] in b2._get_colour_lines()

    # Separate vertical lines on same row
    b2.board = np.array([[3, 2, 4], 
                         [5, 4, 5], 
                         [4, 4, 3], 
                         [4, 2, 3], 
                         [4, 5, 3]])

    assert len(b2._get_colour_lines()) == 2
    assert [(2, 0), (3, 0), (4, 0)] in b2._get_colour_lines()
    assert [(2, 2), (3, 2), (4, 2)] in b2._get_colour_lines()

    # Line of length > 3
    b2.board = np.array([[3, 2, 3], 
                         [2, 4, 3], 
                         [4, 4, 3], 
                         [3, 2, 3], 
                         [2, 5, 2]])
    assert len(b2._get_colour_lines()) == 1
    assert [(0, 2), (1, 2), (2, 2), (3, 2)] in b2._get_colour_lines()

    # Separate vertical lines on same row of different lengths.
    b2.board = np.array([[3, 2, 3], 
                         [4, 4, 3],
                         [4, 4, 3],
                         [4, 2, 3],
                         [2, 5, 2]])
    
    assert len(b2._get_colour_lines()) == 2
    assert [(1, 0), (2, 0), (3, 0)] in b2._get_colour_lines() 
    assert [(0, 2), (1, 2), (2, 2), (3, 2)] in b2._get_colour_lines()

    # Separate horizontal lines on same row
    b3 = Board(num_rows=4, num_cols=8, num_colours=5)
    b3.board = np.array([[2, 3, 2, 4, 4, 2, 3, 3],
                         [4, 2, 5, 2, 3, 4, 2, 3],
                         [3, 3, 3, 3, 5, 2, 2, 2],
                         [2, 3, 2, 4, 4, 2, 3, 3]])

    assert len(b3._get_colour_lines()) == 2, b3._get_colour_lines()
    assert [(2, 0), (2, 1), (2, 2), (2, 3)] in b3._get_colour_lines() 
    assert [(2, 5), (2, 6), (2, 7)] in b3._get_colour_lines()

    b4 = Board(num_rows=10, num_cols=4, num_colours=5)
    b4.board = np.array([[5, 5, 4, 5],
                     [3, 3, 5, 6],
                     [3, 6, 3, 3],
                     [5, 5, 4, 3],
                     [5, 5, 3, 5],
                     [3, 5, 2, 6],
                     [5, 6, 6, 5],
                     [4, 4, 4, 5],
                     [2, 4, 2, 2],
                     [5, 4, 5, 6]])


    output_lines = b4._get_colour_lines()
    assert len(output_lines) == 1, output_lines
    assert [(7, 1), (8, 1), (9, 1)] in output_lines
    




