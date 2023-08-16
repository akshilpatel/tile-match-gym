import numpy as np

from tests.utils import create_board_from_array, create_alternating_array, wipe_coords
from copy import deepcopy


# Test that gravity pushes down tiles
def test_gravity_falls():
    # Case 1

    arr = create_alternating_array(4, 3)
    board = create_board_from_array(arr)
    board.board = wipe_coords(board, [(0, 0), (2, 0), (2, 2), (2, 1), (3, 2)])

    board.gravity()
    assert np.array_equal(board.board, np.array([[0, 0, 0], [0, 3, 0], [3, 2, 2], [3, 2, 3]])), board.board

    # Case 2
    arr = create_alternating_array(height=8, width=7)
    board = create_board_from_array(arr)
    old_board = deepcopy(board.board)
    board.gravity()
    # Board should be unchanged
    assert np.array_equal(board.board, old_board), (arr, old_board.board)

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


def test_gravity_updates_activation_queue():
    # We will never need to do an activation consisting of two specials since this is only caused by swiping two special candies together.
    # The corresponding activation would happen immediately and so gravity would not be called with this type of activation in it.

    # Case 1
    arr = create_alternating_array(4, 3)
    board = create_board_from_array(arr)
    board.board = wipe_coords(board, [(0, 0), (2, 0), (2, 2), (2, 1), (3, 2)])

    board.board[(0, 1)] = 3
    board.board[(1, 2)] = 3
    board.board[(3, 1)] = 3

    board.activation_q.append({"coord": (0, 1), "activation_type": None})
    board.activation_q.append({"coord": (1, 2), "activation_type": 3})  # Vertical laser
    board.activation_q.append({"coord": (3, 1), "activation_type": 1})  # Horizontal laser

    board.gravity()
    activation0 = board.activation_q[0]
    activation1 = board.activation_q[1]
    activation2 = board.activation_q[2]

    assert activation0["coord"] == (1, 1), activation0
    assert activation1 == {"coord": (3, 2), "activation_type": 3}, activation1
    assert activation2 == {"coord": (3, 1), "activation_type": 1}, activation2

    # Case 2
    arr = create_alternating_array(5, 6)
    board = create_board_from_array(arr)
    board.board = wipe_coords(board, [(1, 0), (2, 2), (3, 1), (4, 3), (3, 5), (3, 3), (2, 3)])

    board.activation_q.append({"coord": (4, 2), "activation_type": None})
    board.activation_q.append({"coord": (4, 5), "activation_type": 3})  # Vertical laser
    board.activation_q.append({"coord": (0, 3), "activation_type": 1})  # Horizontal laser

    board.gravity()
    activation0 = board.activation_q[0]
    activation1 = board.activation_q[1]
    activation2 = board.activation_q[2]

    assert activation0["coord"] == (4, 2), activation0
    assert activation1 == {"coord": (4, 5), "activation_type": 3}, activation1
    assert activation2 == {"coord": (3, 3), "activation_type": 1}, activation2


if __name__ == "__main__":
    test_gravity_falls()
