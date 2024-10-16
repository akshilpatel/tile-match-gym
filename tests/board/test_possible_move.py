
from tile_match_gym.board import Board
import numpy as np
from copy import deepcopy

# TODO: Add tests for combination matches flagging as possible move
def test_possible_move():
    """
        All combinations of 3 in a row
        ___   __1   ___   _1_   ___   1__   ___   ___
        11_1  11__  11__  1_1_  1_1_  _11_  _11_  1_11
        ___   ___   __1   ___   _1_   ___   1__   ___


        All combinations of 3 in a col:

        _1_  _1_  _1_  _1_  _1_  1__  __1  _1_
        _1_  _1_  _1_  __1  1__  _1_  _1_  ___
        ___  __1  1__  _1_  _1_  _1_  _1_  _1_
         1    _    _    _    _    _    _    1
    """

    combinations = [
        [[1, 1, 1, 1], [0, 0, 1, 0], [1, 1, 1, 1]],
        [[1, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 0, 1]],
        [[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 1, 1]],
        [[0, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 0, 0, 1], [0, 1, 1, 1]],
        [[1, 1, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1]],
    ]

    x = np.array(
        [
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            [2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
            [3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            [4, 1, 2, 3, 4, 1, 2, 3, 4, 1],
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
            [2, 3, 4, 1, 2, 3, 4, 1, 2, 3]
        ]
    )

    bd = np.array([x.copy(), np.zeros_like(x)])

    # bm = Board(6, 10, 4, board = bd)
    bm = Board(6, 10, 4)
    bm.generate_board()
    for c in combinations:
        print("shape of board = ", bm.board.shape)
        print("shape of board[0, 1:4, 1:5] is ", bm.board[0, 1:4, 1:5].shape, "shape of c is ", np.array(c).shape)
        bm.board[0, 1:4, 1:5] *= c
        bm.board[0] += 1
        bm.num_colours = 5
        # print(bm.possible_move())
        out = bm.possible_move()
        assert out, "There is a move \n" + str(bm.board)
        print("passed")
        bm.board = bd.copy()

    # do rotation of the combinations
    for c in combinations:
        bm.board[0, 1:5, 1:4] *= np.rot90(c)
        # print(bm.board)
        assert bm.possible_move(), "There is a move \n" + str(bm.board)
        print("passed")
        bm.board = bd.copy()
    assert not bm.possible_move(), "There is no possible move \n" + str(bm.board)

    combinations = [
        [[1, 1, 1, 1], [0, 1, 1, 0], [1, 0, 1, 1]],
        [[0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]],
    ]
    for c in combinations:
        bm.board[0, 1:4, 1:5] *= c
        # print(bm.board)
        assert not bm.possible_move(), "There is no possible move \n" + str(bm.board)
        bm.board = bd.copy()

    for i in range(100):
        b = Board(num_rows=4, num_cols=4, num_colours=3, colour_specials=["vertical_laser", "horizontal_laser", "bomb"],
                  colourless_specials=["cookie"], np_random=np.random.default_rng(i))
        b.generate_board()
        old_board = deepcopy(b.board)
        b.possible_move()
        assert np.array_equal(b.board, old_board)

    b = Board(num_rows=5, num_cols=5, num_colours=4, colour_specials=[], colourless_specials=[],
              np_random=np.random.default_rng(2))
    b.generate_board()
    b.board[0] = np.array(
        [
            [1, 4, 4, 1, 2],
            [2, 2, 4, 3, 3],
            [2, 2, 1, 1, 2],
            [1, 4, 3, 2, 3],
            [1, 1, 3, 4, 1]
        ]
    )
    assert not b.possible_move()

    b.board[0] = np.array(
        [
            [1, 3, 4, 1, 3],
            [2, 2, 4, 3, 2],
            [1, 2, 3, 1, 2],
            [3, 4, 1, 4, 1],
            [4, 3, 2, 2, 3]
        ]
    )
    assert b.possible_move()

    b.board[0] = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 4, 4, 2, 3],
            [1, 3, 2, 3, 4],
            [2, 3, 4, 1, 2],
            [4, 1, 2, 1, 2]
        ]
    )
    assert not b.possible_move()
