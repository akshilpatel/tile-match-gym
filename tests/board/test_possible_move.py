
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
            [[1,1,1,1],[0,0,1,0],[1,1,1,1]],
            [[1,1,0,1],[0,0,1,1],[1,1,1,1]],
            [[1,1,1,1],[0,0,1,1],[1,1,0,1]],
            [[1,0,1,1],[0,1,0,1],[1,1,1,1]],
            [[1,1,1,1],[0,1,0,1],[1,0,1,1]],
            [[0,1,1,1],[1,0,0,1],[1,1,1,1]],
            [[1,1,1,1],[1,0,0,1],[0,1,1,1]],
            [[1,1,1,1],[0,1,0,0],[1,1,1,1]],
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
        print("shape of board[0, 1:4, 1:5] is ", bm.board[0,1:4, 1:5].shape, "shape of c is ", np.array(c).shape)
        bm.board[0, 1:4, 1:5] *= c
        # print(bm.board)
        # print(bm.possible_move())
        assert bm.possible_move() == True, "There is a move \n"+str(bm.board)
        print("passed")
        bm.board = bd.copy()
        
    # do rotation of the combinations
    for c in combinations:
        bm.board[0, 1:5, 1:4] *= np.rot90(c)
        # print(bm.board)
        assert bm.possible_move() == True, "There is a move \n"+str(bm.board)
        print("passed")
        bm.board = bd.copy()
    assert bm.possible_move() == False, "There is no possible move \n"+str(bm.board)

    combinations = [
            [[1,1,1,1],[0,1,1,0],[1,0,1,1]],
            [[0,1,1,0],[0,1,1,0],[1,1,1,1]],
            [[1,1,1,1],[0,1,0,1],[1,1,1,1]],
            ]
    for c in combinations:
        bm.board[0, 1:4, 1:5] *= c
        # print(bm.board)
        assert bm.possible_move() == False, "There is no possible move \n"+str(bm.board)
        bm.board = bd.copy()


    for i in range(100):
                
        b = Board(num_rows=4, num_cols=4, num_colours=3, colour_specials= ["vertical_laser", "horizontal_laser", "bomb"], colourless_specials=["cookie"], np_random=np.random.default_rng(i))
        b.generate_board()
        old_board = deepcopy(b.board)
        b.possible_move()
        assert np.array_equal(b.board, old_board)
