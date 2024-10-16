import numpy as np
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

def test_resolve_colour_matches():

    # No lines
    # Match where the colours are different
    new_board = run_resolve_colour_matches(
        np.array([[3, 1, 2, 2],
         [1, 3, 2, 3],
         [3, 1, 1, 2]]
        ))

    expected = np.array([
        [[3, 1, 2, 2],
         [1, 3, 2, 3],
         [3, 1, 1, 2]],
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
        ])
    assert np.array_equal(new_board, expected), new_board


    # Single vertical line
    new_board = run_resolve_colour_matches(
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
    assert np.array_equal(new_board, expected), (new_board, expected)

    # Single horizontal line
    new_board = run_resolve_colour_matches(
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
    new_board = run_resolve_colour_matches(
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
    new_board = run_resolve_colour_matches(
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
    new_board = run_resolve_colour_matches(
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
    new_board = run_resolve_colour_matches(
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

    new_board = run_resolve_colour_matches(
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




def run_resolve_colour_matches(grid, type_grid=None, num_colours=3):
    """
    Helper function to setup a board with a given grid.
    """
    # b = Board(num_rows=len(grid), num_cols=len(grid[0]), num_colours=num_colours)
    # b.board[0] = np.zeros((b.num_rows, b.num_cols))
    # b.board[1] = np.ones_like(b.board[0])
    # b.board[0] = np.array(grid)

    b = Board(num_rows=grid.shape[0], num_cols=grid.shape[1], num_colours=num_colours, board=grid)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    # if type_grid is not None:
    #     b.board[1] = np.array(type_grid)

    tile_coords, tile_names, tile_colours = b.detect_colour_matches()

    b.resolve_colour_matches(tile_coords, tile_names, tile_colours)

    # return tile_coords, tile_names, tile_colours
    return b.board

