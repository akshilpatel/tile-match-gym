import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import deque

from tile_match_gym.tile_translator import TileTranslator
from tile_match_gym.utils.print_board_diffs import highlight_board_diff

"""
    tile_TYPES = {
        1:      cookie,
        2:      tile1 
        2:      tile2,
        ...
        n+1:      tilen,
        n+2:    vlaser_tile1,
        ...,
        2n+1:     vlaser_tilen,
        2n+2:   hlaser_tile1,
        ...,
        3n+1:     hstripe_tilen,
        3n+2:   bomb_tile1,
        ...,
        4n+1:     bomb_tilen,
        
    }
"""

# Base class that only does match 3
# Subclasses that add on functionality - add specials.


class Board:
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        num_colours: int,
        colourless_specials: List[str] = ["cookie"],
        colour_specials: List[str] = ["vertical_laser", "horizontal_laser", "bomb"],
        seed: Optional[int] = None,
        board: Optional[np.ndarray] = None,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_colours = num_colours

        self.flat_size = int(self.num_cols * self.num_rows)
        self.num_actions = self.num_rows * (self.num_cols - 1) + self.num_cols * (self.num_rows - 1)

        self.colourless_specials = colourless_specials
        self.colour_specials = colour_specials
        self.num_colour_specials = len(self.colour_specials)
        self.num_colourless_specials = len(self.colourless_specials)

        self.specials = set(self.colourless_specials + self.colour_specials)

        self.tile_translator = TileTranslator(num_colours, (num_rows, num_cols), self.colourless_specials, self.colour_specials)

        if seed is None:
            seed = np.random.randint(0, 1000000000)
        self.np_random = np.random.default_rng(seed)

        # handle the case where we are given a board
        if board is not None:
            self.board = board
            self.num_rows = len(board)
            self.num_cols = len(board[0])
        else:
            self.generate_board()

        self.indices = np.array([[(r, c) for r in range(self.num_cols)] for c in range(self.num_rows)])
        self.activation_q = []

    def generate_board(self):
        self.board = self.np_random.integers(
            self.num_colourless_specials + 1, self.num_colourless_specials + self.num_colours + 1, size=self.flat_size
        ).reshape(self.num_rows, self.num_cols)
        line_matches = self._get_colour_lines()
        while len(line_matches) > 0:
            self.remove_colour_lines(line_matches)
            line_matches = self._get_colour_lines()

    def remove_colour_lines(self, line_matches: List[List[Tuple[int, int]]]) -> None:
        """Given a board and list of lines where each line is a list of coordinates where the colour of the tiles at each coordinate in one line is the same, changes the board such that none of the

        Args:
            line_matches (List[List[Tuple[int, int]]]): _description_
        """
        ordinary_min = self.num_colourless_specials
        ordinary_max = ordinary_min + self.num_colours - 1
        while len(line_matches) > 0:
            l = line_matches.pop()
            coord = self.np_random.choice(l).tolist()
            new_enc = self.np_random.integers(ordinary_min + 1, ordinary_max + 2)
            while new_enc == self.board[coord[0], coord[1]]:
                new_enc = self.np_random.integers(ordinary_min + 1, ordinary_max + 2)
            self.board[coord[0], coord[1]] = new_enc

    def detect_colour_matches(self) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
        """
        Returns the types and locations of tiles involved in the bottom-most colour matches.
        """
        lines = self._get_colour_lines()
        if len(lines) == 0:
            return [], []
        else:
            tile_coords, tile_names = self._process_colour_lines(lines)
            return tile_coords, tile_names

    def _get_colour_lines(self) -> List[List[Tuple[int, int]]]:
        """
        Starts from the top and checks for 3 or more in a row vertically or horizontally.
        Returns contiguous lines of 3 or more tiles.
        """
        lines = []
        vertical_line_coords = set()
        horizontal_line_coords = set()
        found_line = False
        for row in range(self.num_rows - 1, -1, -1):
            if found_line:
                break  # Only get lowest lines.
            for col in range(self.num_cols):
                # Vertical lines
                if 1 < row and (row, col) not in vertical_line_coords:
                    curr_tile = self.board[row, col]
                    if not self.tile_translator.is_colourless_special(curr_tile):
                        if self.tile_translator.is_same_colour(curr_tile, self.board[row - 1, col]):
                            line_start = row - 1
                            line_end = row
                            while line_start > 0:
                                if self.tile_translator.is_same_colour(curr_tile, self.board[line_start - 1, col]):
                                    line_start -= 1
                                else:
                                    break
                            if line_end - line_start >= 2:
                                found_line = True
                                line = [(i, col) for i in range(line_start, line_end + 1)]      
                                vertical_line_coords.update(line)
                                lines.append(line)

                # Horizontal lines
                if col < self.num_cols - 2 and (row, col) not in horizontal_line_coords:
                    curr_tile = self.board[row, col]
                    if not self.tile_translator.is_colourless_special(curr_tile):
                        if self.tile_translator.is_same_colour(curr_tile, self.board[row, col + 1]):
                            line_start = col
                            line_end = col + 1
                            while line_end < self.num_cols - 1:
                                if self.tile_translator.is_same_colour(curr_tile, self.board[row, line_end + 1]):
                                    line_end += 1
                                else:
                                    break
                            if line_end - line_start >= 2:
                                found_line = True
                                line = [(row, i) for i in range(line_start, line_end + 1)]
                                horizontal_line_coords.update(line)
                                lines.append(line)
        
        return lines
        
    def gravity(self) -> None:
        """
        Given a board with zeros, push the zeros to the top of the board.
        If an activation queue of coordinates is passed in, then the coordinates in the queue are updated as gravity pushes the coordinates down.
        """
        mask_T = self.board.T == 0
        non_zero_mask_T = ~mask_T

        for j, col in enumerate(self.board.T):
            self.board[:, j] = np.concatenate([col[mask_T[j]], col[non_zero_mask_T[j]]])


    def refill(self) -> None:
        """Replace all empty tiles."""
        zero_mask = self.board == 0
        num_zeros = zero_mask.sum()
        if num_zeros > 0:
            rand_vals = self.np_random.integers(len(self.colourless_specials) + 1, self.num_colours + len(self.colourless_specials) + 1, size=num_zeros)
            self.board[zero_mask] = rand_vals

    # TODO: Make this faster.
    def check_move_validity(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
        """
        This function checks if the action actually does anything.
        First it checks if both coordinates are on the board. Then it checks if the action achieves some form of matching.

        Args:
            coord (tuple): The first coordinate on grid corresponding to the action taken. This will always be above or to the left of the second coordinate below.
            coord2 (tuple): coordinate on grid corresponding to the action taken.

        Returns:
            bool: True iff action has an effect on the environment.
        """
        ## Check both coords are on the board. ##
        if not (0 <= coord1[0] < self.num_rows and 0 <= coord1[1] < self.num_cols):
            return False
        if not (0 <= coord2[0] < self.num_rows and 0 <= coord2[1] < self.num_cols):
            return False
        # Check coords are next to each other.
        if not (coord1[0] == coord2[0] or coord1[1] == coord2[1]):
            return False

        # Checks if both are special.
        if self.tile_translator.is_special(self.board[coord1]) and self.tile_translator.is_special(self.board[coord2]):
            return True, "both special"

        if self.tile_translator.is_colourless_special(self.board[coord1]) or self.tile_translator.is_colourless_special(self.board[coord2]):
            return True, "colourless_special"

        # Extract a minimal grid around the coords to check for at least 3 match. This covers checking for Ls or Ts.
        r_min = max(0, min(coord1[0] - 2, coord2[0] - 2))
        r_max = min(self.num_rows, max(coord1[0] + 3, coord2[0] + 3))
        c_min = max(0, min(coord1[1] - 2, coord2[1] - 2))
        c_max = min(self.num_cols, max(coord1[1] + 3, coord2[1] + 3))

        # Swap the coordinates to see what happens.
        self.board[coord1], self.board[coord2] = self.board[coord2], self.board[coord1]
        for r in range(r_min, r_max):
            for c in range(c_min + 2, c_max):
                # If the current and previous 2 are matched and that they are not cookies.
                if self.tile_translator.is_same_colour(self.board[r, c - 2], self.board[r, c - 1], self.board[r, c]):
                    self.board[coord1], self.board[coord2] = self.board[coord2], self.board[coord1]
                    return True

        for r in range(r_min + 2, r_max):
            for c in range(c_min, c_max):
                if self.tile_translator.is_same_colour(self.board[r - 2, c], self.board[r - 1, c], self.board[r, c]):
                    self.board[coord1], self.board[coord2] = self.board[coord2], self.board[coord1]
                    return True

        self.board[coord1], self.board[coord2] = self.board[coord2], self.board[coord1]
        return False

    def _process_colour_lines(self, lines: List[List[Tuple[int, int]]]) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
        """
        Given list of contiguous lines, this function detects the match type from the bottom up, merging any lines that share a coordinate if.
        It greedily extracts the maximum match from the bottom up. So first look at what the most powerful thing you can extract from the bottom up.

        Note: concurrent groups can be matched at the same time.
        """
        tile_names = []
        tile_coords = []

        # lines = sorted([sorted(i, key=lambda x: (x[0],x[1])) for i in lines], key=lambda y: (y[0][0]), reverse=True)
        lines = sorted([sorted(i, key=lambda x: (x[0], x[1])) for i in lines], key=lambda y: (y[0][0]), reverse=False)

        while len(lines) > 0:
            line = lines.pop(0)
            # check for cookie
            if len(line) >= 5 and "cookie" in self.specials:
                tile_names.append("cookie")
                tile_coords.append(line[:5])
                if len(line[5:]) > 2:
                    lines.append(line[5:])  # TODO - should just not pop the line rather than removing and adding again.
            # check for laser
            elif len(line) == 4:
                if line[0][0] == line[1][0] and "horizontal_laser" in self.specials:
                    tile_names.append("horizontal_laser")
                    tile_coords.append(line)
                elif "vertical_laser" in self.specials:
                    tile_names.append("vertical_laser")
                    tile_coords.append(line)
            # check for bomb (coord should appear in another line)
            elif "bomb" in self.specials:
                if any([coord in l for coord in line for l in lines]):  # TODO - REMOVE THIS AS SLOW AND IS DONE TWICE
                    for l in lines:
                        shared = [c for c in line if c in l]
                        if any(shared):
                            shared = shared[0]
                            # Add the closest three coordinates from both lines.
                            sorted_closest = sorted(l, key=lambda x: (abs(x[0] - shared[0]) + abs(x[1] - shared[1])))
                            tile_coords.append(
                                [p for p in line] + [p for p in sorted_closest[:3] if p not in line]
                            )  # TODO: Change this to also only extract 3 closest to intersection from line.
                            tile_names.append("bomb")
                            if len(l) < 6:  # Remove the other line if shorter than 3 after extracting bomb.
                                lines.remove(l)
                            else:
                                for c in sorted_closest[:3]:  # Remove the coordinates that were taken for the bomb
                                    l.remove(c)
                            break  # Stop searching after finding one intersection. This should break out of the for loop.
            # Check for normals. This happends even if the lines are longer than 3 but there are no matching specials.
            elif len(line) >= 3:
                tile_names.append("norm")
                tile_coords.append(line)
            # check for no match
            else:
                tile_names.append("ERR")
                tile_coords.append(line)


# 1. Currently automatch always return true indicating the board always has a match. Instead, two options are available: First, automatch should call gravity and refill before checking if there are matches left to detect/resolve on the board, and then return the. This avoids returning True for a match and then gravity making the match no longer exist.
# 3. Rewrite gravity to not use transpose.
# 4. The activation queue should be emptied all at once. Instead, by doing one activation and then gravity and refill, we might end up with activations being called where there no longer exists a map due to delay?
    def possible_move(self, grid=None):
        """
        checks if any 3 in a row can be made in the current grid
        if grid does not exist then take self.board

        *Maybe incorporate this into the check for matches function*
        
        - 3 in a row already (WONT BE CALLED WHILE THERE ARE MATCHES)

        conditions:
            - 2/3 the same with a neighbouring color at the odd one out
            - 2 in a row with a space and then the same color


        look at the next four in each direction if there is a gap, check
        if anything in the missing axis is the same

        if there is a 2 in a row, check if the space is in position 1 or 2

        All combininations of 3 in a row:
        
        ___   __1   ___   _1_   ___   1__   ___   ___                                                       
        11_1  11__  11__  1_1_  1_1_  _11_  _11_  1_11                                                          
        ___   ___   __1   ___   _1_   ___   1__   ___                                                       

        
        All combinations of 3 in a col:

        _1_  _1_  _1_  _1_  _1_  1__  __1  _1_                                                        
        _1_  _1_  _1_  __1  1__  _1_  _1_  ___                                                       
        ___  __1  1__  _1_  _1_  _1_  _1_  _1_                                                        
         1    _    _    _    _    _    _    1                                                      

                                                                                
        If there is 2 in a row:
            check diagonally up and down for the same color
            check 2 before and 2 after
        else:
            check neighbours of gap for same colour
                                                                                
        """
        rows, cols = self.num_rows, self.num_cols

        exists = lambda c: c[0] >= 0 and c[1] >= 0 and c[0] < rows and c[1] < cols

        if grid is None:
            grid = self.board
        
        for i in range(2):
            if i == 1:
                grid= np.rot90(grid)
                rows, cols = self.num_cols, self.num_rows
                # rows, cols = self.cols, self.rows
            for r in range(rows - 2):
                for c in range(cols - 2):

                    # if 2/3 of the values in the next 3 are the same
                    if len(set(grid[r][c:c + 3])) == 2:
                        # check the possible combiniations
                        if exists([r,c+2]) and grid[r][c] == grid[r][c + 2]: # gap in the middle 1_1
                            for possible in [[r + 1, c + 1], [r - 1, c + 1]]: # triangles
                                if exists(possible) and grid[possible[0]][possible[1]] == grid[r][c]:
                                    return True

                        cn = c
                        # if the second two are the same 011 shift logic up 1 column
                        if grid[r][c+1] == grid[r][c + 2]:
                            cn = c+1

                        for possible in [[r, cn+3], [r-1,cn+2], [r+1, cn+2],[r-1,cn-1], [r+1,cn-1]]: # combinations around _11_
                            if exists(possible) and grid[possible[0]][possible[1]] == grid[r][cn]:
                                print(grid)
                                print("HIT HERE")
                                return True

        return False

def temp_test_matches():
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
    bm = Board(0, 0, 4, board=x.copy())

    for c in combinations:
        bm.board[1:4, 1:5] *= c
        # print(bm.board)
        # print(bm.possible_move())
        assert bm.possible_move() == True, "There is a move \n"+str(bm.board)
        print("passed")
        bm.board = x.copy()
    # do rotation of the combinations
    for c in combinations:
        bm.board[1:5, 1:4] *= np.rot90(c)
        # print(bm.board)
        assert bm.possible_move() == True, "There is a move \n"+str(bm.board)
        print("passed")
        bm.board = x.copy()
    assert bm.possible_move() == False, "There is no possible move \n"+str(bm.board)
    print("passed")


    combinations = [
            [[1,1,1,1],[0,1,1,0],[1,0,1,1]],
            [[0,1,1,0],[0,1,1,0],[1,1,1,1]],
            [[1,1,1,1],[0,1,0,1],[1,1,1,1]],
            ]
    for c in combinations:
        bm.board[1:4, 1:5] *= c
        # print(bm.board)
        assert bm.possible_move() == False, "There is no possible move \n"+str(bm.board)
        print("passed")
        bm.board = x.copy()





if __name__ == "__main__":

    # REMOVE THIS AND MOVE TO THE TESTING SECTION
    # temp_test_matches()
    # x = np.array(
    #     [
    #         [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
    #         [2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
    #         [3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
    #         [4, 1, 2, 3, 4, 1, 2, 3, 4, 1],
    #         [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
    #         [2, 3, 4, 1, 2, 3, 4, 1, 2, 3]
    #     ]
    # )
    # bm = Board(0, 0, 4, board=np.array(x), seed=3)
    # exit()

    import json

    sort_l1 = lambda l: sorted(l, key=lambda x: (x[0], x[1]))
    sort_coords = lambda l: sorted([sort_l1(i) for i in l])
    coords_match = lambda l1, l2: sort_coords(l1) == sort_coords(l2)
    format_test = lambda r, e: "result: \t" + str(r) + "\nexpected: \t" + str(e) + "\n"

    boards = json.load(open("boards.json", "r"))["boards"]

    for i, board in enumerate(boards):
        print("testing board: ", board["name"])

        bm = Board(0, 0, 3, board=np.array(board["board"]), seed=i)
        # bm.rows, bm.cols = np.array(board["board"]).shape
        matches = bm.get_lines()
        tile_coords, tile_names = bm.get_matches(matches)

        print("BOARD::::::")
        bm.print_board()

        expected_matches = [[tuple(coord) for coord in line] for line in board["matches"]]
        expected_tile_coords = [[tuple(coord) for coord in line] for line in board["tile_locations"]]
        expected_activation_q = [tuple(coord) for coord in board["activation_q"]]
        expected_tile_names = board["tile_names"]
        expected_first_activation = np.array(board["first_activation"])
        expected_post_activation = np.array(board["post_activation"])
        expected_post_gravity = np.array(board["post_gravity"])
        expected_post_refill = np.array(board["post_refill"])

        assert len(matches) == len(board["matches"]), "incorrect number of matches found\n" + format_test(matches, expected_matches)
        assert coords_match(matches, expected_matches), "incorrect matches found\n" + format_test(sort_coords(matches), sort_coords(expected_matches))

        assert coords_match(tile_coords, expected_tile_coords), "incorrect tile coords found\n" + format_test(
            sort_coords(tile_coords), sort_coords(expected_tile_coords)
        )

        # make sure that the tiles collected are in the same order
        ordered_matches1 = [sort_l1(t) for t in tile_coords]
        ordered_matches2 = [sort_l1(t) for t in expected_tile_coords]
        assert all([c1 == c2 for c1, c2 in zip(ordered_matches1, ordered_matches2)]), "incorrect match order found\n" + format_test(
            ordered_matches1, ordered_matches2
        )
        # make sure that the tiles collected are correct and in the same order
        # print(tile_names, expected_tile_names)
        assert all([name == expected_name for name, expected_name in zip(tile_names, expected_tile_names)]), "incorrect tile names found\n" + format_test(
            tile_names, expected_tile_names
        )

        # activation tests
        bm.print_board()
        if len(tile_coords) > 0:
            bm.activate_match(tile_coords[0], tile_names[0])
            tile_coords.pop(0)
        bm.print_board()
        assert np.array_equal(bm.board, expected_first_activation), "incorrect board after activation\n" + highlight_board_diff(
            bm.board, expected_first_activation
        )

        # activation queue tests
        print("Activation Queue", bm.activation_q)

        assert len(bm.activation_q) == len(expected_activation_q), "incorrect activation queue length\n" + format_test(bm.activation_q, expected_activation_q)
        assert all([bm.activation_q[i] == a for i, a in enumerate(expected_activation_q)]), "incorrect activation queue\n" + format_test(
            bm.activation_q, expected_activation_q
        )

        # handle activations test
        if len(bm.activation_q) > 0:
            activation = bm.activation_q.pop()
            bm.apply_activation(activation)
            bm.print_board()

        assert np.array_equal(bm.board, expected_post_activation), "incorrect board after activation\n" + highlight_board_diff(
            bm.board, expected_post_activation
        )

        # Gravity test
        bm.gravity()
        assert np.array_equal(bm.board, expected_post_gravity), "incorrect board after gravity\n" + highlight_board_diff(bm.board, expected_post_gravity)

        non_zero_mask = bm.board != 0
        zero_mask = ~non_zero_mask
        old_board = bm.board[non_zero_mask]
        # Refill test
        bm.refill()

        assert np.all(bm.board > 0)
        assert np.all(bm.board[non_zero_mask] == old_board), (bm.board, old_board)
        assert np.all(bm.board[zero_mask] > bm.num_colourless_specials) and np.all(bm.board[zero_mask] <= (bm.num_colourless_specials + bm.num_colours))
        assert np.array_equal(bm.board, expected_post_refill), "incorrect board after refill\n" + highlight_board_diff(bm.board, expected_post_refill)

        print("PASSED")
        print("----")

    bm.colour_check()
