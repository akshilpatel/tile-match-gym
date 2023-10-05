import numpy as np
from tile_match_gym.board import Board
from copy import deepcopy


# def test_check_move_validity():
#     # 3 in a row
#     board = Board(10, 10, 5, seed=1)  # cookie , and vertical lase, horizontal laser, bomb
#     board.board[4, 3] = board.tile_translator.get_tile_encoding("normal", 0)
#     board.board[4, 4] = board.tile_translator.get_tile_encoding("normal", 0)
#     board.board[4, 5] = board.tile_translator.get_tile_encoding("normal", 1)
#     board.board[4, 6] = board.tile_translator.get_tile_encoding("normal", 0)
#     old_board = deepcopy(board.board)
# 
#     assert not board.check_move_validity((2, 1), (2, 2))
#     assert not board.check_move_validity((2, 1), (2, 2))
#     assert board.check_move_validity((4, 5), (4, 6))
#     assert np.array_equal(board.board, old_board), board.board
# 
#     # 3 in a col
#     board.board[6, 2] = board.tile_translator.get_tile_encoding("normal", 3)
#     board.board[8, 2] = board.tile_translator.get_tile_encoding("normal", 3)
#     board.board[9, 2] = board.tile_translator.get_tile_encoding("normal", 3)
#     board.board[7, 3] = board.tile_translator.get_tile_encoding("normal", 3)
#     board.board[7, 2] = board.tile_translator.get_tile_encoding("normal", 1)
#     old_board = deepcopy(board.board)
#     assert board.check_move_validity((6, 2), (7, 2))
#     assert np.array_equal(board.board, old_board)
#     assert board.check_move_validity((7, 2), (7, 3))
#     assert np.array_equal(board.board, old_board)
# 
#     # 3 in a row with special.
#     board.board[4, 6] = board.tile_translator.get_tile_encoding("vertical_laser", 0)
#     old_board = deepcopy(board.board)
#     assert board.check_move_validity((4, 5), (4, 6))
#     assert np.array_equal(board.board, old_board)
# 
#     # 4 in a row with special
#     assert board.check_move_validity((4, 5), (5, 5))
#     assert np.array_equal(board.board, old_board)
# 
#     # 4 in a column
#     board = Board(8, 7, 7, seed=2)  # cookie , and vertical lase, horizontal laser, bomb
#     board.board[0, 0] = board.tile_translator.get_tile_encoding("normal", 2)
#     board.board[1, 0] = board.tile_translator.get_tile_encoding("normal", 2)
#     board.board[3, 0] = board.tile_translator.get_tile_encoding("normal", 2)
#     board.board[2, 1] = board.tile_translator.get_tile_encoding("normal", 2)
#     board.board[2, 0] = board.tile_translator.get_tile_encoding("normal", 3)
#     old_board = deepcopy(board.board)
#     assert board.check_move_validity((2, 1), (2, 0))
#     assert np.array_equal(board.board, old_board)
# 
#     # 5 in a col
#     board = Board(5, 7, 7, seed=2)  # cookie , and vertical lase, horizontal laser, bomb
#     board.board[0, 0] = board.tile_translator.get_tile_encoding("normal", 3)
#     board.board[1, 0] = board.tile_translator.get_tile_encoding("normal", 3)
#     board.board[2, 0] = board.tile_translator.get_tile_encoding("normal", 2)
#     board.board[3, 0] = board.tile_translator.get_tile_encoding("normal", 3)
#     board.board[4, 0] = board.tile_translator.get_tile_encoding("normal", 3)
#     board.board[2, 1] = board.tile_translator.get_tile_encoding("normal", 3)
#     old_board = deepcopy(board.board)
#     assert board.check_move_validity((2, 1), (2, 0))
#     assert np.array_equal(board.board, old_board)
# 
#     # 4 in a col with 2 specials
#     board = Board(5, 7, 10, seed=2)  # cookie , and vertical lase, horizontal laser, bomb
#     board.board[0, 0] = board.tile_translator.get_tile_encoding("bomb", 3)
#     board.board[1, 0] = board.tile_translator.get_tile_encoding("vertical_laser", 3)
#     board.board[2, 0] = board.tile_translator.get_tile_encoding("normal", 2)
#     board.board[3, 0] = board.tile_translator.get_tile_encoding("normal", 3)
#     board.board[2, 1] = board.tile_translator.get_tile_encoding("normal", 3)
#     old_board = deepcopy(board.board)
#     assert board.check_move_validity((2, 1), (2, 0))
#     assert np.array_equal(board.board, old_board)
# 
#     # 2 specials in a row
#     board = Board(6, 4, 4, seed=2)
#     board.board[3, 2] = board.tile_translator.get_tile_encoding("bomb", 3)
#     board.board[3, 3] = board.tile_translator.get_tile_encoding("bomb", 0)
#     assert board.check_move_validity((3, 2), (3, 3))
# 
#     # 2 specials in a col
#     board = Board(4, 5, 4, seed=3)
#     board.board[2, 2] = board.tile_translator.get_tile_encoding("vertical_laser", 3)
#     board.board[3, 2] = board.tile_translator.get_tile_encoding("bomb", 1)
#     assert board.check_move_validity((3, 2), (3, 3))
# 
#     # 2 specials in a row
#     board = Board(4, 5, 4, seed=3)
#     board.board[2, 2] = board.tile_translator.get_tile_encoding("vertical_laser", 3)
#     board.board[3, 2] = board.tile_translator.get_tile_encoding("bomb", 1)
#     assert board.check_move_validity((3, 2), (3, 3))
# 
#     # Cookie + special
#     board = Board(24, 5, 12, seed=3)
#     board.board[2, 2] = board.tile_translator.get_tile_encoding("bomb", 3)
#     board.board[3, 2] = board.tile_translator.get_tile_encoding("cookie", 1)
#     assert board.check_move_validity((3, 2), (3, 3))
# 
#     # Cookie + normal
#     board = Board(12, 7, 8, seed=3)
#     board.board[2, 2] = board.tile_translator.get_tile_encoding("normal", 3)
#     board.board[2, 3] = board.tile_translator.get_tile_encoding("cookie", 0)
#     assert board.check_move_validity((2, 2), (2, 3))
# 
#     # Cookie + cookie
#     board = Board(21, 12, 10, seed=3)
#     board.board[12, 2] = board.tile_translator.get_tile_encoding("cookie", 0)
#     board.board[13, 2] = board.tile_translator.get_tile_encoding("cookie", 0)
#     assert board.check_move_validity((12, 2), (13, 2))
