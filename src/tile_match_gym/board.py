import numpy as np

from typing import Optional, List, Tuple, Dict
from collections import Counter

from tile_match_gym.tile_translator import TileTranslator
from tile_match_gym.utils.print_board_diffs import highlight_board_diff

"""
tile_colours = {
    0: Colourless
    1: colour1,
    2: ...
}
coloured_tile_types = {
    1:      normal,
    2:      v_laser,
    3:      h_laser,
    4:      bomb,
}

colourless_tile_types = {
    -1:      cookie,

}

empty_tile: [0, 0] if tile_

"""





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
        self.normal_tile_range = self.tile_translator.get_normal_tile_range()

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
        self.board = np.ones((2, self.num_rows, self.num_cols), dtype=int)
        self.board[0] = self.np_random.integers(1, self.num_colours+1, self.flat_size).reshape(self.num_rows, self.num_cols)

        line_matches = self.get_colour_lines()
        num_line_matches = len(line_matches)

        while not self.possible_move() or num_line_matches > 0:
            if num_line_matches > 0:
                self.remove_colour_lines(line_matches)
            else:
                shuffled_idcs = np.arange(self.num_rows * self.num_cols)
                self.np_random.shuffle(shuffled_idcs)
                shuffled_idcs = shuffled_idcs.reshape(self.num_rows, self.num_cols)
                self.board = self.board[:, shuffled_idcs // self.num_cols, shuffled_idcs % self.num_cols]

            line_matches = self.get_colour_lines()
            num_line_matches = len(line_matches)
            
        assert self.possible_move()
        assert self.get_colour_lines() == []
            
    def remove_colour_lines(self, line_matches: List[List[Tuple[int, int]]]) -> None:
        """Given a board and list of lines where each line is a list of coordinates where the colour of the tiles at each coordinate in one line is the same, changes the board such that none of the
            This is only used for generating the board. This function does not touch the type of tiles in the board, only the colours.
        Args:
            line_matches (List[List[Tuple[int, int]]]): List of lines where each line is colour match.
        """
        while len(line_matches) > 0:
            l = line_matches[0]
            row = min(self.num_rows - 1, l[0][0] + 1)
            self.board[0, :row+1, :] = self.np_random.integers(1, self.num_colours+1, int((row+1) * self.num_cols)).reshape(-1, self.num_cols)
            line_matches = self.get_colour_lines()
            
    def detect_colour_matches(self) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
        """
        Returns the types and locations of tiles involved in the bottom-most colour matches.
        """
        lines = self.get_colour_lines()
        if len(lines) == 0:
            return [], []
        else:
            tile_coords, tile_names = self.process_colour_lines(lines)
            return tile_coords, tile_names

    def get_colour_lines(self) -> List[List[Tuple[int, int]]]:
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
                    if self.board[1, row, col] > 0 : # Not Colourless special
                        if self.board[0, row, col] == self.board[0, row-1, col]: # Don't have to check the other one isn't a colourless special since colourless specials should be 0 in first axis.
                            line_start = row - 1
                            line_end = row
                            while line_start > 0:
                                if self.board[0, row, col] == self.board[0, line_start - 1, col]:
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
                    if self.board[1, row, col] > 0 : # Not Colourless special
                        if self.board[0, row, col] == self.board[0, row, col + 1]: # Don't have to check the other one isn't a colourless special since colourless specials should be 0 in first axis.
                            line_start = col
                            line_end = col + 1
                            while line_end < self.num_cols - 1:
                                if self.board[0, row, col] == self.board[0, row, line_end + 1]:
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
        Given a board with zeroes, push the zeroes to the top of the board.
        If an activation queue of coordinates is passed in, then the coordinates in the queue are updated as gravity pushes the coordinates down.
        """

        colour_zero_mask_T = self.board[0].T == 0
        
        type_zero_mask_T = self.board[1].T == 0
        zero_mask_T = colour_zero_mask_T & type_zero_mask_T
        non_zero_mask_T = ~zero_mask_T
        print(zero_mask_T.shape)
        
        for j in range(self.num_cols):
            self.board[0][:, j] = np.concatenate([self.board[0][:, j][zero_mask_T[j]], self.board[0][:, j][non_zero_mask_T[j]]])
            self.board[1][:, j] = np.concatenate([self.board[1][:, j][zero_mask_T[j]], self.board[1][:, j][non_zero_mask_T[j]]])
            

    def refill(self) -> None:
        """Replace all empty tiles."""
        zero_mask_colour = self.board[0] == 0
        zero_mask_type = self.board[1] == 0
        zero_mask = zero_mask_colour & zero_mask_type
        # print(zero_mask.shape)
        num_zeros = zero_mask.sum()
        if num_zeros > 0:
            rand_vals = self.np_random.integers(1, self.num_colours + 1, size=num_zeros)
            self.board[0, zero_mask] = rand_vals
            self.board[1, zero_mask] = 1


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

        # Checks if both are special
        if self.board[1, coord1[0], coord1[1]] != 0 and self.board[1, coord2[0], coord2[1]] != 0:
            return True

        # At least one colourless special.
        if self.board[1, coord1[0], coord1[1]] < 0 or self.board[1, coord2[0], coord2[1]] < 0:
            return True

        # Extract a minimal grid around the coords to check for at least 3 match. This covers checking for Ls or Ts.
        r_min = max(0, min(coord1[0] - 2, coord2[0] - 2))
        r_max = min(self.num_rows, max(coord1[0] + 3, coord2[0] + 3))
        c_min = max(0, min(coord1[1] - 2, coord2[1] - 2))
        c_max = min(self.num_cols, max(coord1[1] + 3, coord2[1] + 3))

        # Swap the coordinates to see what happens.
        self.board[:, coord1], self.board[:, coord2] = self.board[:, coord2], self.board[:, coord1]
        for r in range(r_min, r_max):
            for c in range(c_min + 2, c_max):
                # If the current and previous 2 are matched and that they are not cookies.
                if self.board[1, r, c] > 0: # Check it isn't a colourless special or empty.
                    if self.board[0, r, c - 2] == self.board[0, r, c - 1] == self.board[0, r, c]:
                        self.board[:, coord1], self.board[:, coord2] = self.board[:, coord2], self.board[:, coord1]
                        return True

        for r in range(r_min + 2, r_max):
            for c in range(c_min, c_max):
                if self.board[1, r, c] > 0: 
                    if self.board[0, r - 2, c] == self.board[0, r - 1, c] == self.board[0, r, c]:
                        self.board[:, coord1], self.board[:, coord2] = self.board[:, coord2], self.board[:, coord1]
                        return True

        self.board[:, coord1], self.board[:, coord2] = self.board[:, coord2], self.board[:, coord1]
        return False

    def process_colour_lines(self, lines: List[List[Tuple[int, int]]]) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
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
                tile_names.append("normal")
                tile_coords.append(line)
            # check for no match
            else:
                tile_names.append("ERR")
                tile_coords.append(line)

    def move(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> None:
        
        if not self.check_move_validity(coord1, coord2):
            return
        
        # Swap the coordinates.
        self.board[:, coord1], self.board[:, coord2] = self.board[:, coord2], self.board[:, coord1]

        ## Combination match ##

        # If there are two special coords. Add one activation with both coords to the activation queue.
        has_two_specials = self.board[1, coord1[0], coord1[1]] != 0 and self.board[1, coord2[0], coord2[1]] != 0
        has_one_colourless_special = self.board[1, coord1[0], coord1[1]] < 0 or self.board[1, coord2[0], coord2[1]] < 0
        # Combination match of two specials or one colourless special and any other tile.
        if has_two_specials or has_one_colourless_special:
            self.combination_match(coord1, coord2)
            self.gravity()
            self.refill()

        ## Colour matching ##
        has_match = True
        while has_match:
            match_locs, match_types = self.detect_colour_matches()
            if len(match_locs) == 0:
                has_match = False
            else:
                self.resolve_colour_matches(match_locs, match_types)
                self.gravity()
                self.refill()
    
    def resolve_colour_matches(self, match_locs: List[List[Tuple[int, int]]], match_types: List[str]) -> None:
        """The main loop for processing a batch of colour matches. This function eliminates tiles, activates specials and creates new specials.

        Args:
            match_locs (List[List[Tuple[int, int]]]): List of match locations. Each match location is a list of coordinates that are part of the match.
            match_types (List[str]): List of match types ordered in the same way as match_locs.
        """
        self.activation_q_coords = set()
        self.activation_q = []
        for i in range(len(match_locs)):
            match_coords = match_locs[i]
            match_type = match_types[i]

            for coord in match_coords:
                tile = self.board[coord]
                if tile[1] != 0:
                    self.activation_q_coords.add(coord)
                    self.activation_q.append((coord, *tile))
                self.board[coord] = 0
            
            for coord, tile_type, tile_colour in self.activation_q:
                self.activate_special(coord, tile_type, tile_colour)

            if match_type != "normal":
                self.create_special(match_coords, match_type)

    def activate_special(self, coord, tile_type, tile_colour):
        special_r, special_c = coord

        # Delete special.
        self.board[:, special_r, special_c] = 0
        
        if tile_type == "vertical_laser":
            for row in range(self.num_rows):
                if row == special_r: continue

                elif (self.board[1, row, special_c] not in [0, 1]) and (row, special_c) not in self.activation_q_coords:
                    self.activation_q_coords.add((row, special_c))
                    self.activation_q.append((coord, *self.board[row, special_c]))
                self.board[:, row, special_c] = 0

        elif tile_type == "horizontal_laser":
            for col in range(self.num_cols):
                if col == special_c: continue
                elif self.board[1, special_r, col] not in [0, 1] and (special_r, col) not in self.activation_q_coords:
                    self.activation_q_coords.add((special_r, col))
                    self.activation_q.append((coord, *self.board[special_r, col]))
                self.board[:, special_r, col] = 0
            
        elif tile_type == "bomb":
            for i in range(coord[0] - 1, coord[0] + 2):
                for j in range(coord[1] - 1, coord[1] + 2):
                    if (i, j) == coord: continue
                    elif self.board[1, i, j] not in [0, 1]:
                        self.activation_q.append(((i, j), *self.board[1, i, j]))
                    self.board[:, i, j] = 0

        elif tile_type == "cookie":
            
            colours = [self.board[0, coord[0] + i, coord[1] + j]  for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
            most_common_colour = Counter(colours).most_common(1)[0]
            # get the most common neighbour and remove all variants of that tile
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    colour = self.board[0, i, j]
                    if colour == most_common_colour:
                        if self.board[1, i, j] not in [0, 1]:
                            self.activation_q.append((i, j))
                        else:
                            self.board[:, i, j] = 0
        else: 
            raise ValueError(f"{tile_type} is an invalid special tile type.")

    def get_special_creation_pos(self, coords: List[Tuple[int, int]], straight=True) -> Tuple[int, int]:
        """
        Given a set of coordinates return the position of the special tile that should be placed
        The position should be as close to the center as possible but should not already be special.
        """

        if not straight:
            # get the corner coords
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            corner = (max(xs, key=xs.count), max(ys, key=ys.count))

            std = [c for c in coords if not self.board[1, c[0], c[1]] not in [0, 1]]
            if corner in std:
                return corner
            else:
                return sorted(std, key=lambda x: (x[0] - corner[0]) ** 2 + (x[1] - corner[1]) ** 2)[0]

        sorted_coords = sorted([c for c in coords if not self.board[1, c[0], c[1]] not in [0, 1]], key=lambda x: (x[0], x[1]))
        
        if len(sorted_coords) % 2 == 0:
            return sorted_coords[len(sorted_coords) // 2 - 1]
        return sorted_coords[len(sorted_coords) // 2]


    def possible_move(self, grid=None):
        """
        Checks if any 3 in a row can be made in the current grid
        If the grid does not exist then take self.board

        If 2/3 in a row are the same color then either a gap 1_1 or 2 in a row 11_1 or 1_11 exist.

        Check combinations of diagonal neighbours to determine if a match is possible

        """
        rows, cols = self.num_rows, self.num_cols

        exists = lambda c: c[0] >= 0 and c[1] >= 0 and c[0] < rows and c[1] < cols

        if grid is None:
            grid = self.board[0, :, :]
        
        for i in range(2): # check both orientations
            if i == 1:
                grid = np.rot90(grid)
                rows, cols = self.num_cols, self.num_rows
            for r in range(rows - 2):
                for c in range(cols - 2):

                    # if 2/3 of the values in the next 3 are the same
                    if len(set(grid[r, c:c + 3])) == 2:
                        # check the possible combiniations
                        if exists([r,c+2]) and grid[r, c] == grid[r, c + 2]: # gap in the middle 1_1
                            for possible in [[r + 1, c + 1], [r - 1, c + 1]]: # triangles
                                if exists(possible) and grid[possible[0], possible[1]] == grid[r, c]:
                                    return True

                        cn = c
                        # if the second two are the same 011 shift logic up 1 column
                        if grid[r, c+1] == grid[r, c + 2]:
                            cn = c+1

                        for possible in [[r, cn+3], [r-1,cn+2], [r+1, cn+2],[r-1,cn-1], [r+1,cn-1]]: # combinations around _11_
                            if exists(possible) and grid[possible[0], possible[1]] == grid[r, cn]:
                                return True

        return False # there are no ways to make a move and get 3 in a row
    

    def create_special(self, match_coords: List[Tuple[int, int ]], match_type: str) -> None :
        """This function creates a special tile at the location specified by the match_coords.

        Args:
            match_coords (List[Tuple[int, int ]]): Coordinates to pick from.
            match_type (str): The type of new special tile to create
        """

        raise NotImplementedError()
