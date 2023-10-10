from tests.utils import create_alternating_array, wipe_coords, create_board_from_array, get_special_locations
import numpy as np

def test_refill():
    arr = create_alternating_array(5, 7)
    b = create_board_from_array(arr)
    b.num_colours = 4
    wipe_coords(b, [(0, 0), (0, 1), (0, 3), (0, 6)])
    b.refill()
    assert np.all(b.board[:, :, 0] > 0)
    assert np.all(b.board[:, :, 1] == 1)


    arr = create_alternating_array(5, 7)
    b = create_board_from_array(arr)
    b.num_colours = 6
    wipe_coords(b, [(0, 0), (0, 1), (0, 2), (0, 5), (1, 5), (2, 5)])
    b.refill()
    
    assert np.all(b.board[:, :, 0] > 0)
    assert np.all(b.board[:, :, 1] == 1)

    arr = create_alternating_array(5, 7)
    b = create_board_from_array(arr)  #
    b.num_colours = 13
    wipe_coords(b, [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6)])
    b.refill()
    assert np.all(b.board[:, :, 0] > 0)
    assert np.all(b.board[:, :, 1] == 1)