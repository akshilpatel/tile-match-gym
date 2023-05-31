import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Union, Tuple
from gymnasium.spaces import Box, Discrete


### Scoring ### 
# For one match, add +


class CandyCrush(gym.Env):
    def __init__(self, height:int, width:int, num_candy_types:int, num_blockers:int, seed:Optional[int] = None):

        super().__init__()
        self.height = height
        self.width = width
        self.num_candy_types = num_candy_types
        self.num_blockers = num_blockers
        if seed is None:
            seed = np.random.randint(0, 1000000000)
        self.np_random = np.random.default_rng(seed)
        self.flat_size = int(self.width * self.height)
        self.observation_space = Box(low = 1, high = num_candy_types + 1, shape=(height, width))
        self.action_space = Discrete(int(2 * self.flat_size))

    def reset(self, seed:Optional[int] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.board = self.np_random.integers(1, self.num_candy_types + 1, size = self.flat_size).reshape(self.height, self.width)
        has_matched_candies = True
        while has_matched_candies:
            has_matched_candies = self.eliminate_matched_candies()
            self.cascade_board()
        return self.board.copy(), {}

    def step(self, action:int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Actions correspond to 2 for each tile position. One for a down swipe and another for a right swipe. 
        # 1) Find the coordinate and the up or right coord. Check for invalid actions.
        is_vertical = action <= self.flat_size
        coord = np.unravel_index(action, self.board.shape)
        coord2 = coord[0] + is_vertical, coord[1] + (1-is_vertical)
        # Invalid action 
        if not (0 <= coord2[0] < self.height) or not (0 <= coord[1] < self.width):
            return self.board.copy(), self.default_reward, False, False, self.get_info()
        # 2) Swap the coords unless there is a block. TODO:Unless one of the candy types is special. In this case, activate other effects.
        else:
            self.board[coord[0], coord[1]], self.board[coord2[0], coord2[1]] = self.board[coord2[0], coord2[1]], self.board[coord[0], coord[1]]
        has_matched_candies = self.eliminate_matched_candies()        
        while has_matched_candies:
            self.cascade_board()
            has_matched_candies = self.eliminate_matched_candies()
        # TODO: Compute score from eliminated candies.
        return self.board.copy()
    def cascade_board(self) -> None:
        self.push_zeros_up()
        self.fill_missing()

    def eliminate_matched_candies(self) -> bool:
        # go through the rows to check for lines of 3 or more
        # go through the colums to check for lines of 3 or more
        # remove any lines found and repeat until clean sweeps
        mask = np.ones(self.board.shape, dtype=int)
        hit = False
        # Check all horizontal matches by finding 
        for row in range(self.height):
            for i in range(2, self.width):
                # If the current and previous 2 are matched
                if self.board[row, i-2] == self.board[row, i-1] == self.board[row, i]:
                    hit = True
                    start = i
                    # Iterate through to find the full number of matched candies. 
                    while self.board[row, i] == self.board[row, i-1]:
                        i+=1
                        if i >= self.width:
                            break
                    # Set all to zero.
                    mask[row, start-2:i] = 0
        for col in range(self.width):
            for i in range(2, self.height):
                if self.board[i-2, col] == self.board[i-1, col] == self.board[i, col]:
                    hit = True
                    start = i
                    while self.board[i, col] == self.board[i-1, col]:
                        i+=1
                        if i >= self.height:
                            break
                    mask[start-2:i, col] = 0
        
        self.board *= mask
        return hit

    def push_zeros_up(self):
        """Push empty slots to the top."""
        for col in self.board.T:
            zero_count = 0
            for i in range(len(col)-1, -1, -1):
                if col[i] == 0:
                    zero_count += 1
                elif zero_count != 0:
                    col[i + zero_count] = col[i]
                    col[i] = 0

    def fill_missing(self):
        """
        Search top to bottom in each column and break if you hit something that isn't zero.
        Since the board should
        """
        for col in self.board.T:
            for i in range(len(col)):
                if col[i] == 0:
                    col[i] = self.np_random.integers(1, self.num_candy_types + 1, size=1)
                else:
                    break

    def close(self):
        super().close()

    def render(self):
        pass

##############


# def alternating_board(w,h):
#     board = np.zeros((w,h), np.int32)
#     board[::2,::2] = 1
#     board[1::2,1::2] = 1
#     return board + 1

# def test_horiz():
#     board = Board(8,8)
#     x = alternating_board(8,8)
#     x[2][1:4] = 3
#     board.set_board(x)
#     board.check_board_brute()
#     print("board = \n", board.board)
#     x[2][1:4] = 0
#     assert np.array_equal(x, board.board)

# def test_vert():
#     board = Board(8,8)
#     x = alternating_board(8,8)
#     x[:,3][2:5] = 3
#     print("x = \n", x)
#     board.set_board(x)
#     board.check_board_brute()
#     print("board = \n", board.board)
#     x[:,3][2:5] = 0
#     assert np.array_equal(x, board.board)

# def test_cross():
#     board = Board(8,8)
#     x = alternating_board(8,8)
#     x[:,2][1:4] = 3
#     x[2][1:4] = 3
#     print("x = \n", x)
#     board.set_board(x)
#     board.check_board_brute()
#     print("board = \n", board.board)
#     x[:,2][1:4] = 0
#     x[2][1:4] = 0
#     assert np.array_equal(x, board.board)

# if __name__ == '__main__':
#     board = Board(8,8)

#     print(board.board)

#     board.check_board_brute()
#     print(board.board)

#     test_horiz()
#     test_vert()
#     test_cross()
    
#     print("#"*80)
    
#     board = Board(20,20)
#     y = board.board.copy()

#     s = time.time()

#     board.check_board_brute()
#     print("time taken:", time.time() - s)
#     print("before:\n",y)
#     print("after:\n",board.board)
#     board.cascade()
#     print("after cascade:\n",board.board)
#     board.fill_missing()
#     print("after filling zeros:\n",board.board)

#     print("#"*80)

#     board = Board(200,200)
#     y = board.board.copy()

#     s = time.time()

#     board.check_board_brute()
#     print("time taken:", time.time() - s)
#     print("before:\n",y)
#     print("after:\n",board.board)



if __name__ == "__main__":
    env = CandyCrush(5, 4, 3, 0, 1)
    print(env.reset())
    print(env.step(13))