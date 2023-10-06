from tests.utils import create_alternating_array, wipe_coords, create_board_from_array, get_special_locations
import numpy as np

def test_refill():
    arr = create_alternating_array(5, 7)
    board = create_board_from_array(arr)
    board.num_colours = 4
    wipe_coords(board, [(0, 0), (0, 1), (0, 3), (0, 6)])
    board.refill()
    assert 0 not in board.board
    assert set(board.board.flatten().tolist()).issubset(
        set(range(board.num_colourless_specials + 1, board.num_colourless_specials + board.num_colours))
    ), board.board
    assert np.all(board.board > 0)
    for i in range(board.num_rows):
        for j in range(board.num_cols):
            assert not board.tile_translator.is_special(board.board[i,j])


    arr = create_alternating_array(5, 7)
    board = create_board_from_array(arr)
    board.num_colours = 6
    wipe_coords(board, [(0, 0), (0, 1), (0, 2), (0, 5), (1, 5), (2, 5)])
    board.refill()
    assert np.all(board.board > 0)
    for i in range(board.num_rows):
        for j in range(board.num_cols):
            assert not board.tile_translator.is_special(board.board[i,j])

    arr = create_alternating_array(5, 7)
    board = create_board_from_array(arr)  #
    board.num_colours = 13
    wipe_coords(board, [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6)])
    board.refill()
    assert np.all(board.board > 0)
    for i in range(board.num_rows):
        for j in range(board.num_cols):
            assert not board.tile_translator.is_special(board.board[i,j])
