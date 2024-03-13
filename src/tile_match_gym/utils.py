from tile_match_gym.board import Board
from itertools import product
import numpy as np
import multiprocessing as mp

def compute_num_states(num_rows, num_cols, num_colours, num_processes):

    board = Board(num_rows, num_cols, num_colours, [], [], 0)
    board.board = np.ones((2, num_rows, num_cols), dtype=np.int32)
    num_flat = int(num_rows*num_cols)
    potential_boards = list(product(range(1, num_colours+1), repeat=num_flat))

    all_args = [(num_rows, num_cols, board, b) for b in potential_boards]
    with mp.Pool(num_processes) as p:
        results = p.starmap(process_combo, all_args)
    
    r_arr = np.array(results)

    return r_arr[:, 0].sum(), r_arr[:, 1].sum()

def process_combo(num_rows, num_cols, board, b):
    board.board[0] = np.array(b).reshape(num_rows, num_cols)
    line_matches = board.get_colour_lines()
    has_poss_move = board.possible_move()
    has_no_colour_matches = len(line_matches) == 0
    return has_poss_move and has_no_colour_matches, has_no_colour_matches
        



if __name__ == "__main__":
    print(compute_num_states(4, 4, 3, mp.cpu_count()))