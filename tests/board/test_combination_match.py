
from tile_match_gym.board import Board
import numpy as np

# Take boards from json
def test_combination_match():
    # Cookie + cookie should clear the board.
    b = Board(6, 3, 4)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[4, 3, 3],
                           [2, 2, 1],
                           [4, 0, 2],
                           [4, 0, 2],
                           [3, 3, 1],
                           [2, 2, 1]])


    b.board[1] = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, -1, 1],
                           [1, -1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

    b.combination_match((2, 1), (3, 1))
    assert np.array_equal(b.board, np.zeros((2, 6, 3)))

    # Cookie + normal (no specials). Should remove normals of the same color, and activate specials of same colour
    b = Board(4, 6, 3)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[2, 2, 3, 2, 3, 2], 
                           [1, 3, 2, 3, 2, 1], 
                           [2, 2, 3, 0, 1, 3], 
                           [2, 3, 3, 2, 3, 2]])

    b.board[1] = np.array([[1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, -1, 1, 1], 
                           [1, 1, 1, 1, 1, 1]])

    b.combination_match((2, 3), (2, 4))
    assert np.array_equal(b.board[0], np.array([[2, 2, 3, 2, 3, 2],
                                                [0, 3, 2, 3, 2, 0],
                                                [2, 2, 3, 0, 0, 3],
                                                [2, 3, 3, 2, 3, 2]]))

    assert np.array_equal(b.board[1], np.array([[1, 1, 1, 1, 1, 1],
                                                [0, 1, 1, 1, 1, 0],
                                                [1, 1, 1, 0, 0, 1],
                                                [1, 1, 1, 1, 1, 1]]))

    # Cookie + normal (1 of each special activated)
    b = Board(6, 4, 4)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[3, 1, 1, 2],
                           [2, 3, 0, 2],
                           [2, 1, 2, 3],
                           [4, 3, 4, 1],
                           [3, 1, 2, 4],
                           [2, 1, 3, 2]])

    b.board[1] = np.array([[1, 1, 1, 1],
                           [1, 1, -1, 1],
                           [4, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 2, 1],
                           [3, 1, 1, 1]])
    b.combination_match((1, 3), (1, 2))

    assert np.array_equal(b.board[0], np.array([[3, 1, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 3],
                                                [0, 0, 0, 1],
                                                [3, 1, 0, 4],
                                                [0, 0, 0, 0]]))

    assert np.array_equal(b.board[1], np.array([[1, 1, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 1],
                                                [0, 0, 0, 1],
                                                [1, 1, 0, 1],
                                                [0, 0, 0, 0]]))

    # Cookie + bomb should  convert all of same colour to bomb
    b = Board(5, 5, 4)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[3, 1, 2, 1, 3],
                           [3, 1, 3, 1, 4],
                           [1, 3, 4, 4, 3],
                           [2, 1, 1, 2, 2],
                           [4, 2, 3, 0, 4]])


    b.board[1] = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, -1, 4]])


    b.combination_match((4, 3), (4, 4))
    
    assert np.array_equal(b.board[0], np.array([[3, 1, 2, 0, 0],
                                                [3, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 3, 0, 0]]))

    assert np.array_equal(b.board[1], np.array([[1, 1, 1, 0, 0],
                                                [1, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0]]))
    # Cookie + bomb (other preexisting specials)
    b.board[0] = np.array([[3, 1, 2, 1, 3],
                           [3, 1, 3, 1, 4],
                           [1, 3, 4, 4, 3],
                           [2, 1, 1, 2, 2],
                           [4, 2, 3, 0, 4]])


    b.board[1] = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 3],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [2, 1, 1, -1, 4]])

    b.combination_match((4, 3), (4, 4))
    assert np.array_equal(b.board[0], np.array([[0, 1, 2, 1, 3],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 2, 3, 0, 0]]))

    assert np.array_equal(b.board[1], np.array([[0, 1, 1, 1, 1],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0],
                                                [0, 1, 1, 0, 0]]))

    # Cookie + v laser
    b = Board(6, 4, 5)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[2, 3, 1, 4],
                           [3, 3, 4, 2],
                           [1, 4, 3, 5],
                           [3, 3, 2, 5],
                           [0, 5, 1, 1],
                           [3, 2, 5, 4]])

    b.board[1] = np.array([[1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [-1, 2, 1, 1],
                           [1, 1, 1, 1]])

    b.combination_match((4, 0), (4, 1))
    assert np.array_equal(b.board[0],  np.array([[2, 0, 0, 0],
                                                 [3, 0, 0, 0],
                                                 [1, 0, 0, 0],
                                                 [3, 0, 0, 0],
                                                 [0, 0, 0, 0],
                                                 [3, 0, 0, 0]]))

    assert np.array_equal(b.board[1],  np.array([[1, 0, 0, 0],
                                                 [1, 0, 0, 0],
                                                 [1, 0, 0, 0],
                                                 [1, 0, 0, 0],
                                                 [0, 0, 0, 0],
                                                 [1, 0, 0, 0]]))


    # Cookie + h laser
    b = Board(5, 6, 3)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[3, 3, 1, 2, 2, 1],
                           [1, 3, 3, 2, 1, 1],
                           [1, 2, 1, 3, 1, 3],
                           [2, 1, 2, 2, 3, 1],
                           [2, 2, 3, 2, 2, 3]])

    b.board[1] = np.array([[1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, -1, 3, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1]])

    b.combination_match((2, 2), (2, 1))
    assert np.array_equal(b.board[0],  np.array([[0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [2, 2, 3, 2, 2, 3]]))


    assert np.array_equal(b.board[1],  np.array([[0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [1, 1, 1, 1, 1, 1]]))



    # vertical laser + horizontal laser
    b = Board(4, 4, 6)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[4, 3, 6, 2],
                           [2, 5, 5, 1],
                           [6, 2, 5, 3],
                           [3, 1, 1, 5]])

    b.board[1] = np.array([[1, 1, 1, 1],
                           [1, 1, 2, 3],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1]])

    b.combination_match((1, 3), (1, 2))

    assert np.array_equal(b.board[0], np.array([[4, 3, 0, 2],
                                                [0, 0, 0, 0],
                                                [6, 2, 0, 3],
                                                [3, 1, 0, 5]])), b.board[0]

    assert np.array_equal(b.board[1], np.array([[1, 1, 0, 1],
                                                [0, 0, 0, 0],
                                                [1, 1, 0, 1],
                                                [1, 1, 0, 1]]))

    # v_laser + v_laser
    b = Board(5, 6, 7)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[6, 4, 6, 2, 7, 3],
                           [4, 7, 4, 7, 3, 1],
                           [1, 3, 7, 3, 2, 2],
                           [2, 4, 3, 6, 7, 7],
                           [2, 1, 6, 1, 5, 1]])

    b.board[1] = np.array([[1, 1, 1, 1, 4, 1],
                           [1, 3, 3, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 2, 1],
                           [1, 1, 1, 3, 1, 1]])


    b.combination_match((1, 1), (1, 2))
    assert np.array_equal(b.board[0], np.array([[6, 0, 6, 2, 7, 3],
                                                [0, 0, 0, 0, 0, 0],
                                                [1, 0, 7, 3, 2, 2],
                                                [2, 0, 3, 6, 7, 7],
                                                [2, 0, 6, 1, 5, 1]])) 

    assert np.array_equal(b.board[1], np.array([[1, 0, 1, 1, 4, 1],
                                                [0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 1, 1, 1],
                                                [1, 0, 1, 1, 2, 1],
                                                [1, 0, 1, 3, 1, 1]]))


    # h_laser x 2
    b.board[0] = np.array([[6, 4, 6, 2, 7, 3],
                           [4, 7, 4, 7, 3, 1],
                           [1, 3, 7, 3, 2, 2],
                           [2, 4, 3, 6, 7, 7],
                           [2, 1, 6, 1, 5, 1]])

    b.board[1] = np.array([[1, 1, 1, 1, 4, 1],
                           [1, 3, 3, 1, 1, 1],
                           [1, 1, 1, 1, 2, 1],
                           [1, 1, 1, 1, 2, 1],
                           [1, 1, 1, 3, 1, 1]])

    b.combination_match((2, 4), (3, 4))


    assert np.array_equal(b.board[0], np.array([[6, 4, 6, 0, 0, 0],
                                                [4, 7, 4, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0],
                                                [2, 4, 3, 6, 0, 7],
                                                [2, 1, 6, 1, 0, 1]])) 

    assert np.array_equal(b.board[1], np.array([[1, 1, 1, 0, 0, 0],
                                                [1, 3, 3, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 0, 1],
                                                [1, 1, 1, 3, 0, 1]]))


    
    # bomb + bomb
    b = Board(6, 7, 6)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[2, 3, 5, 1, 1, 2, 6],
                           [5, 1, 5, 4, 4, 5, 2],
                           [1, 2, 4, 5, 6, 1, 4],
                           [4, 3, 1, 1, 3, 4, 2],
                           [1, 3, 6, 2, 1, 2, 6],
                           [5, 1, 6, 1, 4, 1, 5]])

    b.board[1] = np.array([[1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 4, 1, 1, 1, 1, 1],
                           [1, 4, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1]])
    
    b.combination_match((3, 1), (4, 1))
    assert np.array_equal(b.board[0], np.array([[2, 3, 5, 1, 1, 2, 6],
                                                [0, 0, 0, 0, 4, 5, 2],
                                                [0, 0, 0, 0, 6, 1, 4],
                                                [0, 0, 0, 0, 3, 4, 2],
                                                [0, 0, 0, 0, 1, 2, 6],
                                                [0, 0, 0, 0, 4, 1, 5]]))
    
    assert np.array_equal(b.board[1], np.array([[1, 1, 1, 1, 1, 1, 1],
                                                [0, 0, 0, 0, 1, 1, 1],
                                                [0, 0, 0, 0, 1, 1, 1],
                                                [0, 0, 0, 0, 1, 1, 1],
                                                [0, 0, 0, 0, 1, 1, 1],
                                                [0, 0, 0, 0, 1, 1, 1]]))
    
    
    # bomb + v_laser
    b = Board(7, 7, 7)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[3, 2, 6, 6, 1, 4, 7],
                        [5, 5, 2, 1, 4, 7, 7],
                        [2, 7, 2, 6, 4, 5, 5],
                        [6, 6, 7, 2, 2, 7, 5],
                        [1, 6, 6, 3, 7, 4, 7],
                        [2, 4, 1, 6, 6, 1, 3],
                        [1, 6, 4, 4, 7, 1, 7]])

    b.board[1] = np.array([[1, 1, 1, 3, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 4, 1, 1],
                        [4, 1, 1, 1, 2, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1]])


    b.combination_match((3, 4), (4, 4))
    assert np.array_equal(b.board[0], np.array([[0, 0, 0, 0, 0, 0, 0],
                                                [5, 5, 2, 0, 0, 0, 7],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0, 3],
                                                [1, 6, 4, 0, 0, 0, 7]]))

    assert np.array_equal(b.board[1], np.array([[0, 0, 0, 0, 0, 0, 0],
                                                [1, 1, 1, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0, 1],
                                                [1, 1, 1, 0, 0, 0, 1]]))

    # bomb + h_laser
    b = Board(7, 7, 7)
    b.num_specials_activated = 0
    b.num_new_specials = 0
    b.board[0] = np.array([[3, 2, 6, 6, 1, 4, 7],
                           [5, 5, 2, 1, 4, 7, 7],
                           [2, 7, 2, 6, 4, 5, 5],
                           [6, 6, 7, 2, 2, 7, 5],
                           [1, 6, 6, 3, 7, 4, 7],
                           [2, 4, 1, 6, 6, 1, 3],
                           [1, 6, 4, 4, 7, 1, 7]])

    b.board[1] = np.array([[1, 1, 1, 3, 1, 1, 1],
                           [1, 1, 1, 4, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 4, 1, 1],
                           [4, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1]])


    b.combination_match((1, 3), (0, 3))
    assert np.array_equal(b.board[0], np.array([[0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [2, 7, 0, 0, 0, 0, 5],
                                                [6, 6, 0, 0, 0, 0, 5],
                                                [1, 6, 0, 0, 0, 0, 7],
                                                [2, 4, 0, 0, 0, 1, 3],
                                                [1, 6, 0, 0, 0, 1, 7]]))

    assert np.array_equal(b.board[1], np.array([[0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 1, 0, 0, 0, 0, 1],
                                                [1, 1, 0, 0, 0, 0, 1],
                                                [4, 1, 0, 0, 0, 0, 1],
                                                [1, 1, 0, 0, 0, 1, 1],
                                                [1, 1, 0, 0, 0, 1, 1]]))
