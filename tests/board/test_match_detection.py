import numpy as np
import random
import pytest

from tile_match_gym.board import Board


def line_in_lines(line, lines):
    for l in lines:
        if set(line) == set(l):
            return True
    return False


def test_get_colour_lines():
    random.seed(0)
    np.random.seed(0)

    b = Board(num_rows=3, num_cols=4, num_colours=3)

    # Colourless specials should not be included
    b.board[0] = np.zeros((b.num_rows, b.num_cols))
    b.board[1] = np.full_like(b.board[0], -1)

    assert b.get_colour_lines() == []

    b.board[1] = np.array([[-1, 1, 1, -1],
                           [-1, -1, -1, -1],
                           [-1, 1, 1, -1]])
    
    b.board[0] = np.array([[0, 4, 4, 0],
                           [0, 0, 0, 0],
                           [0, 4, 4, 0]])
    assert b.get_colour_lines() == []

    # No lines
    b.board[0] = np.array([[4, 3, 4, 3],
                           [5, 4, 5, 4], 
                           [4, 3, 4, 2]])
    b.board[1] = np.ones_like(b.board[0])

    assert b.get_colour_lines() == [], b.get_colour_lines()

    # Single vertical line on top edge
    b.board[0, 1, 0,] = 4
    # assert [(0, 0), (1, 0), (2, 0)] in b.get_colour_lines()
    assert(line_in_lines([(0, 0), (1, 0), (2, 0)], b.get_colour_lines()))
    assert len(b.get_colour_lines()) == 1

    # Single horizontal line on bottom edge.
    b.board[0, 1, 0] = 2
    b.board[0, 2, 1] = 4
    assert  [(2, 0), (2, 1), (2, 2)] in b.get_colour_lines(), b.get_colour_lines()
    assert len(b.get_colour_lines()) == 1


    # Different board shape.
    b2 = Board(num_rows=5, num_cols=3, num_colours=7)
    b2.board[0] = np.array([[3, 4, 4], 
                            [5, 5, 2], 
                            [4, 4, 3], 
                            [5, 6, 2], 
                            [3, 5, 2]])
    
    assert b2.get_colour_lines() == []

    # Single horizontal line on left and top edge
    b2.board[0, 0, 0] = 4
    assert [(0, 0), (0, 1), (0, 2)] in b2.get_colour_lines()
    assert len(b2.get_colour_lines()) == 1

    # Single horizontal line on bottom edge
    b2.board[0, 0, 2] = 6
    b2.board[0, 4, 0] = 5
    b2.board[0, 4, 2] = 5
    assert [(4, 0), (4, 1), (4, 2)] in b2.get_colour_lines()
    assert len(b2.get_colour_lines()) == 1

    # Two horizontal lines on different lines
    b2.board[0] = np.array([[3, 4, 4], 
                            [5, 5, 5], 
                            [4, 4, 3], 
                            [3, 3, 3], 
                            [3, 5, 2]])

    assert len(b2.get_colour_lines()) == 1, b2.get_colour_lines()
    assert [(3, 0), (3, 1), (3, 2)] in b2.get_colour_lines()

    # Separate horizontal and vertical lines on separate rows.
    b2.board[0] = np.array([[3, 4, 4], 
                            [5, 4, 5], 
                            [4, 4, 3], 
                            [3, 3, 3], 
                            [3, 5, 2]])

    assert len(b2.get_colour_lines()) == 1
    assert  [(3, 0), (3, 1), (3, 2)] in b2.get_colour_lines()

    # Separate vertical lines on same row
    b2.board[0] = np.array([[3, 2, 4], 
                            [5, 4, 5], 
                            [4, 4, 3], 
                            [4, 2, 3], 
                            [4, 5, 3]])

    assert len(b2.get_colour_lines()) == 2
    assert [(2, 0), (3, 0), (4, 0)] in b2.get_colour_lines()
    assert [(2, 2), (3, 2), (4, 2)] in b2.get_colour_lines()

    # Line of length > 3
    b2.board[0] = np.array([[3, 2, 3], 
                                  [2, 4, 3], 
                                  [4, 4, 3],
                                  [3, 2, 3], 
                                  [2, 5, 2]])
    
    assert len(b2.get_colour_lines()) == 1
    assert [(0, 2), (1, 2), (2, 2), (3, 2)] in b2.get_colour_lines()

    # Separate vertical lines on same row of different lengths.
    b2.board[0] = np.array([[3, 2, 3], 
                                  [4, 4, 3],
                                  [4, 4, 3],
                                  [4, 2, 3],
                                  [2, 5, 2]])
    
    assert len(b2.get_colour_lines()) == 2
    assert [(1, 0), (2, 0), (3, 0)] in b2.get_colour_lines() 
    assert [(0, 2), (1, 2), (2, 2), (3, 2)] in b2.get_colour_lines()

    # Separate horizontal lines on same row
    b3 = Board(num_rows=4, num_cols=8, num_colours=5)
    b3.board[0] = np.array([[2, 3, 2, 4, 4, 2, 3, 3],
                                  [4, 2, 5, 2, 3, 4, 2, 3],
                                  [3, 3, 3, 3, 5, 2, 2, 2],
                                  [2, 3, 2, 4, 4, 2, 3, 3]])

    assert len(b3.get_colour_lines()) == 2, b3.get_colour_lines()
    assert [(2, 0), (2, 1), (2, 2), (2, 3)] in b3.get_colour_lines() 
    assert [(2, 5), (2, 6), (2, 7)] in b3.get_colour_lines()

    b4 = Board(num_rows=10, num_cols=4, num_colours=5)
    b4.board[0] = np.array([[5, 5, 4, 5],
                            [3, 3, 5, 6],
                            [3, 6, 3, 3],
                            [5, 5, 4, 3],
                            [5, 5, 3, 5],
                            [3, 5, 2, 6],
                            [5, 6, 6, 5],
                            [4, 4, 4, 5],
                            [2, 4, 2, 2],
                            [5, 4, 5, 6]])


    output_lines = b4.get_colour_lines()
    assert len(output_lines) == 2, output_lines
    assert [(7,1),(8,1),(9,1)] in output_lines
    assert [(7,0),(7,1),(7,2)] in output_lines
    # assert [(7, 1), (8, 1), (9, 1)] in output_lines

    # not on the bottom vertical line
    b6 = Board(num_rows=4, num_cols=4, num_colours=7)
    b6.board[0] = np.array([[2, 3, 4, 3],
                           [3, 1, 3, 2],
                           [3, 1, 3, 2],
                           [4, 1, 2, 1]])
    assert  [(1, 1), (2, 1), (3, 1)] in b6.get_colour_lines(), str([(1, 1), (2, 1), (3, 1)]) + " should be " + str(b6.get_colour_lines())
    assert len(b6.get_colour_lines()) == 1

    # T shape
    np.random.seed(1)
    random.seed(1)
    print("T_SHAPE")
    bo =  np.array([[2, 3, 4, 3],
                    [1, 1, 1, 2],
                    [3, 1, 3, 2],
                    [4, 1, 2, 1]])
    bb = np.ones_like(bo)
    b8 = Board(num_rows=4, num_cols=4, num_colours=7) #, board=np.array([bo, bb]), seed=1)
    b8.board[0] = np.array([[2, 3, 4, 3],
                           [1, 1, 1, 2],
                           [3, 1, 3, 2],
                           [4, 1, 2, 1]])
    lines = b8.get_colour_lines()
    # assert  [(1,0),(1,1),(1,2),(2,1),(3,1)] in lines, lines
    assert [(1,1),(2,1),(3,1)] in lines, lines
    assert [(1,0),(1,1),(1,2)] in lines, lines
    assert len(b6.get_colour_lines()) == 1


    # upside down T shape
    b8 = Board(num_rows=4, num_cols=4, num_colours=7) #, board=np.array([bo, bb]), seed=1)
    b8.board[0] = np.array([[2, 3, 4, 3],
                           [4, 1, 2, 2],
                           [3, 1, 3, 2],
                           [1, 1, 1, 3]])
    lines = b8.get_colour_lines()
    # assert  [(1,0),(1,1),(1,2),(2,1),(3,1)] in lines, lines
    assert [(1,1),(2,1),(3,1)] in lines, lines
    assert [(3,0),(3,1),(3,2)] in lines, lines
    assert len(b6.get_colour_lines()) == 1

    # L
    b8 = Board(num_rows=4, num_cols=4, num_colours=7) #, board=np.array([bo, bb]), seed=1)
    b8.board[0] = np.array([[2, 3, 4, 3],
                           [4, 1, 2, 2],
                           [3, 1, 3, 2],
                           [3, 1, 1, 1]])
    lines = b8.get_colour_lines()
    # assert  [(1,0),(1,1),(1,2),(2,1),(3,1)] in lines, lines
    assert [(1,1),(2,1),(3,1)] in lines, lines
    assert [(3,1),(3,2),(3,3)] in lines, lines
    assert len(b6.get_colour_lines()) == 1
    
def test_process_colour_lines():

    # No lines
    # Match where the colours are different
    coordinates, match_types, match_colors = get_match_details(
        [[3, 1, 2, 2],
         [1, 3, 2, 3],
         [3, 1, 1, 2]])

    assert len(coordinates) == 0
    assert len(match_types) == 0
    assert len(match_colors) == 0


    # Single vertical line
    coordinates, match_types, match_colors = get_match_details(
        np.array([[2, 3, 4, 3],
                  [3, 1, 3, 2],
                  [3, 1, 3, 2],
                  [4, 1, 2, 1]]))
    expected_coordinates = [(1, 1), (2, 1), (3, 1)]
    expected_types = ['normal']
    assert all([t == e for t, e in zip(match_types, expected_types)]), str(match_types) + " should be " + str(expected_types)
    assert match_colors == [1]
    assert len(coordinates) == 1
    assert len(coordinates[0]) == 3
    assert all([c in expected_coordinates for c in coordinates[0]])

    # Single horizontal line
    coordinates, match_types, match_colors = get_match_details(
        np.array([[2, 3, 3, 4, 3],
                  [3, 2, 4, 3, 2],
                  [4, 1, 1, 1, 3],
                  [3, 4, 2, 3, 2]]))
    expected_coordinates = [(2, 1), (2, 2), (2, 3)]
    assert len(coordinates) == 1
    assert len(coordinates[0]) == 3
    assert all([c in expected_coordinates for c in coordinates[0]])
    expected_types = ['normal']
    assert all([t == e for t, e in zip(match_types, expected_types)])
    assert match_colors == [1]

    # T 
    coordinates, match_types, match_colors = get_match_details(
        np.array([[2, 3, 3, 4, 3],
                  [4, 1, 1, 1, 3],
                  [3, 2, 1, 3, 2],
                  [3, 4, 1, 3, 2]]))
    expected_coordinates = [(1, 1), (1, 2), (1, 3), (2, 2), (3, 2)]
    assert len(coordinates) == 1
    assert len(coordinates[0]) == 5
    assert all([c in expected_coordinates for c in coordinates[0]])
    expected_types = ['bomb']
    assert all([t == e for t, e in zip(match_types, expected_types)])
    assert match_colors == [1]

    # L

    # Disjoint lines should not be merged.
    coordinates, match_types, match_colors = get_match_details(
        np.array([[2, 3, 3, 2, 3],
                  [4, 1, 4, 1, 3],
                  [3, 1, 3, 1, 2],
                  [3, 1, 4, 1, 2]]))
    assert len(coordinates) == 2
    assert [(1,1),(2,1),(3,1)] in coordinates
    assert [(1,3),(2,3),(3,3)] in coordinates
    expected_types = ['normal', 'normal']
    assert all([t == e for t, e in zip(match_types, expected_types)])
    assert match_colors == [1, 1]

    # Lines > 3.
    coordinates, match_types, match_colors = get_match_details(
        np.array([[2, 1, 3, 2, 3],
                  [4, 1, 4, 1, 3],
                  [3, 1, 3, 2, 2],
                  [3, 1, 4, 1, 2]]))
    assert len(coordinates) == 1
    assert [(0,1),(1,1),(2,1),(3,1)] in coordinates
    expected_types = ['vertical_laser']
    assert all([t == e for t, e in zip(match_types, expected_types)]), str(match_types) + " should be " + str(expected_types)
    assert match_colors == [1]

    # Lines > 3 where the board config doesn't include the corresponding special.
    coordinates, match_types, match_colors = get_match_details(
        np.array([[2, 1, 3, 2, 3],
                  [4, 1, 4, 1, 3],
                  [2, 1, 3, 2, 2],
                  [4, 1, 4, 1, 3],
                  [3, 1, 3, 2, 2],
                  [3, 1, 4, 1, 2]]))
    assert len(coordinates) == 1
    assert [(0,1),(1,1),(2,1),(3,1),(4,1)] in coordinates
    expected_types = ['cookie']
    assert all([t == e for t, e in zip(match_types, expected_types)]), str(match_types) + " should be " + str(expected_types)
    assert match_colors == [0]


def get_match_details(grid, type_grid=None, num_colours=3):
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
    return tile_coords, tile_names, tile_colours


if __name__ == "__main__":
    b4 = Board(num_rows=10, num_cols=4, num_colours=5)
    b4.board[0] = np.array([[5, 5, 4, 5],
                            [3, 3, 5, 6],
                            [3, 6, 3, 3],
                            [5, 5, 4, 3],
                            [5, 5, 3, 5],
                            [3, 5, 2, 6],
                            [5, 6, 6, 5],
                            [4, 4, 4, 5],
                            [2, 4, 2, 2],
                            [5, 4, 5, 6]])


    output_lines = b4.get_colour_lines()
    assert len(output_lines) == 1, output_lines
    assert [(3, 1), (4, 1), (5, 1)] in output_lines
