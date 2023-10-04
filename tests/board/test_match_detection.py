import numpy as np
import pytest

from tile_match_gym.board import Board


def test_get_colour_lines():
    b = Board(num_rows=3, num_cols=4, num_colours=3)

    # Colourless specials should not be included
    b.board = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    assert b._get_colour_lines() == []

    b.board = np.array(
        [
            [1, 4, 4, 1],
            [1, 1, 1, 1],
            [1, 4, 4, 1],
        ]
    )
    assert b._get_colour_lines() == []

    # No lines
    b.board = np.array([[4, 3, 4, 3], [5, 4, 5, 4], [4, 3, 4, 2]])

    assert b._get_colour_lines() == [], b._get_colour_lines()

    # Single vertical line on top edge
    b.board[1, 0] = 4
    assert b._get_colour_lines()[0] == [(0, 0), (1, 0), (2, 0)], b._ge_colour_lines()
    assert len(b._get_colour_lines()) == 1

    # Single horizontal line on bottom edge.
    b.board[1, 0] = 2
    b.board[2, 1] = 4
    assert b._get_colour_lines()[0] == [(2, 0), (2, 1), (2, 2)], b._get_colour_lines()
    assert len(b._get_colour_lines()) == 1

    # Different board shape.
    b2 = Board(num_rows=5, num_cols=3, num_colours=4)
    b2.board = np.array([[3, 4, 4], [5, 5, 2], [4, 4, 3], [5, 6, 2], [3, 5, 2]])
    assert b2._get_colour_lines() == []

    # Single horizontal line on left and top edge
    b2.board[0, 0] = 4
    assert b2._get_colour_lines()[0] == [(0, 0), (0, 1), (0, 2)]
    assert len(b2._get_colour_lines()) == 1

    # Single horizontal line on bottom edge
    b2.board[0, 2] = 6
    b2.board[4, 0] = 5
    b2.board[4, 2] = 5
    assert b2._get_colour_lines()[0] == [(4, 0), (4, 1), (4, 2)]
    assert len(b2._get_colour_lines()) == 1

    # Two horizontal lines on different lines
    b2.board = np.array([[3, 4, 4], [5, 5, 5], [4, 4, 3], [3, 3, 3], [3, 5, 2]])

    assert len(b2._get_colour_lines()) == 1, b2._get_colour_lines()
    assert b2._get_colour_lines()[0] == [(3, 0), (3, 1), (3, 2)]

    # Separate horizontal and vertical lines on separate rows.
    b2.board = np.array([[3, 4, 4], [5, 4, 5], [4, 4, 3], [3, 3, 3], [3, 5, 2]])

    assert len(b2._get_colour_lines()) == 1
    assert b2._get_colour_lines()[0] == [(3, 0), (3, 1), (3, 2)]

    # Separate vertical lines on same row
    b2.board = np.array([[3, 2, 4], [5, 4, 5], [4, 4, 3], [4, 2, 3], [4, 5, 3]])

    assert len(b2._get_colour_lines()) == 2
    assert b2._get_colour_lines()[0] == [(2, 0), (3, 0), (4, 0)]
    assert b2._get_colour_lines()[1] == [(2, 2), (3, 2), (4, 2)]

    # Line of length > 3
    b2.board = np.array([[3, 2, 3], [2, 4, 3], [4, 4, 3], [3, 2, 3], [2, 5, 2]])
    assert len(b2._get_colour_lines()) == 1
    assert b2._get_colour_lines()[0] == [(0, 2), (1, 2), (2, 2), (3, 2)]

    # Separate vertical lines on same row of different lengths.
    b2.board = np.array([[3, 2, 3], [4, 4, 3], [4, 4, 3], [4, 2, 3], [2, 5, 2]])
    assert len(b2._get_colour_lines()) == 2
    assert b2._get_colour_lines()[0] == [(1, 0), (2, 0), (3, 0)]
    assert b2._get_colour_lines()[1] == [(0, 2), (1, 2), (2, 2), (3, 2)]
