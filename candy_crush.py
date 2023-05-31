import numpy as np
import time

class Board:
    def __init__(self, width, height, num_types: int= 6):
        self.board = np.random.randint(1, num_types, size=(width, height))
        self.height = height
        self.width = width
        self.num_types = num_types

    def set_board(self, board):
        self.board = board
        self.height = len(board)
        self.width = len(board[0])

    """ brute force check if there are any lines of 3 or more"""
    def check_board_brute(self):
        # go through the rows to check for lines of 3 or more
        # go through the colums to check for lines of 3 or more
        # remove any lines found and repeat until clean sweeps
        hit = False
        mask = np.ones(self.board.shape, dtype=int)
        while not hit:
            hit = False
            
            # need to do from indexes so that can apply the mask
            for row in range(self.height):
                for i in range(2, self.width):
                    if self.board[row][i-2] == self.board[row][i-1] == self.board[row][i] and self.board[row][i] != -1:
                        start = i-2
                        while self.board[row][i] == self.board[row][i-1]:
                            i+=1
                            if i >= self.width:
                                break
                        mask[row][start:i] = 0
                        hit = True
            for col in range(self.width):
                for i in range(2, self.height):
                    if self.board[i-2][col] == self.board[i-1][col] == self.board[i][col] and self.board[i][col] != -1:
                        start = i-2
                        while self.board[i][col] == self.board[i-1][col]:
                            i+=1
                            if i >= self.height:
                                break
                        mask[start:i, col] = 0
                        hit = True
        # apply the mask
        self.board = self.board * mask
    
    """ moves all the zeros in a column up to the top """
    def cascade(self):
        for col in self.board.T:
            zero_count = 0
            for i in range(len(col)-1, -1, -1):
                if col[i] == 0:
                    zero_count += 1
                elif zero_count != 0:
                    col[i+zero_count] = col[i]
                    col[i] = 0
    
    """ fills in the missing spaces with random numbers """
    def fill_missing(self):
        for col in self.board.T:
            for i in range(len(col)):
                if col[i] == 0:
                    col[i] = np.random.randint(1, self.num_types) # TODO: change this thing
                else:
                    break

def alternating_board(w,h):
    board = np.zeros((w,h), np.int32)
    board[::2,::2] = 1
    board[1::2,1::2] = 1
    return board + 1

def test_horiz():
    board = Board(8,8)
    x = alternating_board(8,8)
    x[2][1:4] = 3
    board.set_board(x)
    board.check_board_brute()
    print("board = \n", board.board)
    x[2][1:4] = 0
    assert np.array_equal(x, board.board)

def test_vert():
    board = Board(8,8)
    x = alternating_board(8,8)
    x[:,3][2:5] = 3
    print("x = \n", x)
    board.set_board(x)
    board.check_board_brute()
    print("board = \n", board.board)
    x[:,3][2:5] = 0
    assert np.array_equal(x, board.board)

def test_cross():
    board = Board(8,8)
    x = alternating_board(8,8)
    x[:,2][1:4] = 3
    x[2][1:4] = 3
    print("x = \n", x)
    board.set_board(x)
    board.check_board_brute()
    print("board = \n", board.board)
    x[:,2][1:4] = 0
    x[2][1:4] = 0
    assert np.array_equal(x, board.board)

if __name__ == '__main__':
    board = Board(8,8)

    print(board.board)

    board.check_board_brute()
    print(board.board)

    test_horiz()
    test_vert()
    test_cross()
    
    print("#"*80)
    
    board = Board(20,20)
    y = board.board.copy()

    s = time.time()

    board.check_board_brute()
    print("time taken:", time.time() - s)
    print("before:\n",y)
    print("after:\n",board.board)
    board.cascade()
    print("after cascade:\n",board.board)
    board.fill_missing()
    print("after filling zeros:\n",board.board)

    print("#"*80)

    board = Board(200,200)
    y = board.board.copy()

    s = time.time()

    board.check_board_brute()
    print("time taken:", time.time() - s)
    print("before:\n",y)
    print("after:\n",board.board)
