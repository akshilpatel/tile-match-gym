from tile_match_gym.board import Board
from itertools import product
import numpy as np
def compute_num_states(num_rows, num_cols, num_colours):
    num_valid_states = 0
    num_potential_next_obs = 0
    board = Board(num_rows, num_cols, num_colours, [], [], 0)
    board.board = np.ones((2, num_rows, num_cols), dtype=np.int32)
    num_flat = int(num_rows*num_cols)
    potential_boards = list(product(range(1, num_colours+1), repeat=num_flat))
    
    for b in potential_boards:
        board.board[0] = np.array(b).reshape(num_rows, num_cols)
        
        line_matches = board.get_colour_lines()
        has_poss_move = board.possible_move()
        has_no_colour_matches = len(line_matches) == 0
        if has_poss_move and has_no_colour_matches:
            num_valid_states += 1

        if has_no_colour_matches:
            num_potential_next_obs += 1

    return num_valid_states, num_potential_next_obs



if __name__ == "__main__":
    print(compute_num_states(4, 4, 3))