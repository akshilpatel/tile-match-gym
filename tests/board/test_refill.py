from tests.utils import create_alternating_array, wipe_coords, create_board_from_array, get_special_locations
import numpy as np
from tile_match_gym.board import Board
def test_refill():    
    b = Board(5, 7, 4, seed=1)
    wipe_coords(b, [(0, 0), (0, 1), (0, 3), (0, 6)])
    b.refill()
    assert np.all(b.board[0] > 0)
    assert np.all(b.board[0] <= b.num_colours)
    assert np.all(b.board[1] == 1)

    b = Board(5, 7, 6, seed=2)
    wipe_coords(b, [(0, 0), (0, 1), (0, 2), (0, 5), (1, 5), (2, 5)])

    b.refill()

    assert np.all(b.board[0] > 0)
    assert np.all(b.board[0] <= b.num_colours)
    assert np.all(b.board[1] == 1)

    b = Board(4, 7, 5, seed=1)

    b.board[1] = np.array([[1, 2, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, -1],
                        [1, 3, 1, 5, -1, 1, 2],
                        [1, 1, 3, 1, 2, 1, 1]])
    wipe_coords(b, [(0, 6), (1, 6), (2, 6), (3, 6), (3, 2)])

    b.refill()
    type_arr = b.board[1] = np.array([[1, 2, 1, 1, 1, 1, 1], 
                                    [1, 1, 1, 1, 1, 1, 1], 
                                    [1, 3, 1, 5, -1, 1, 1], 
                                    [1, 1, 1, 1, 2, 1, 1]])

    assert np.all(b.board[0] > 0)
    assert np.all(b.board[0] <= b.num_colours)
    assert np.array_equal(b.board[1], type_arr)