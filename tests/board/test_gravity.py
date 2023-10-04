import numpy as np

from tests.utils import create_alternating_board, wipe_coords
from copy import deepcopy


# Test that gravity pushes down tiles
def test_gravity():
    # Case 1

    board = create_alternating_board(4, 3)
    board.board = wipe_coords(board, [(0, 0), (2, 0), (2, 2), (2, 1), (3, 2)])

    board.gravity()
    assert np.array_equal(board.board, np.array([[0, 0, 0], [0, 3, 0], [3, 2, 2], [3, 2, 3]]))

    # Case 2
    board = create_alternating_board(height=8, width=7)
    old_board = deepcopy(board.board)
    board.gravity()
    # Board should be unchanged
    assert np.array_equal(board.board, old_board)

    board.board = wipe_coords(board, [(0, 0), (1, 0), (3, 4), (4, 4), (5, 4), (3, 2), *[(7, i) for i in range(7)]])

    board.gravity()
    assert np.array_equal(
        board.board,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 3, 0, 3, 0, 3, 2],
                [0, 2, 2, 2, 0, 2, 3],
                [2, 3, 3, 3, 0, 3, 2],
                [3, 2, 2, 2, 2, 2, 3],
                [2, 3, 2, 3, 3, 3, 2],
                [3, 2, 3, 2, 2, 2, 3],
                [2, 3, 2, 3, 2, 3, 2],
            ]
        ),
    )


if __name__ == "__main__":
    test_gravity_falls()
