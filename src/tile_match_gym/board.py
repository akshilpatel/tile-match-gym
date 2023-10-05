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
        self.board[coord1], self.board[coord2] = self.board[coord2], self.board[coord1]

        
        ## Combination match ##

        # If there are two special coords. Add one activation with both coords to the activation queue.
        coord1_special = self.tile_translator.is_special(self.board[coord1])
        coord2_special = self.tile_translator.is_special(self.board[coord2])
        has_one_colourless_special = self.tile_translator.is_colourless_special(self.board[coord1]) or self.tile_translator.is_colourless_special(self.board[coord1])
        # Combination match of two specials or one colourless special and any other tile.
        if (coord1_special and coord2_special) or has_one_colourless_special:
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
        # For each match
            # For each coord in match:
                # 1. If the match_coord is an existing special then add activation to local activation_queue
                # 2. Delete coord.

            # For each activation in local queue:
                # Activate the special at the coordinate.
            
            # if match_type != normal:
                # Create special(match_type, self.np_random.choice(match_coords))
        self.activation_q_coords = set()

        self.activation_q = []
        for i in range(len(match_locs)):
            match_coords = match_locs[i]
            match_type = match_types[i]

            for coord in match_coords:
                tile = self.board[coord]
                if self.tile_translator.is_special(tile):
                    self.activation_q_coords.add(coord)
                    self.activation_q.append((coord, *self.tile_translator.get_type_colour(tile)))
                self.board[coord] = 0
            
            for coord, tile_type, tile_colour in self.activation_q:
                self.activate_special(coord, tile_type, tile_colour)

            if match_type != "normal":
                self.create_special(match_type, self.np_random.choice(match_coords))

    def activate_special(self, coord, tile_type, tile_colour):
        special_r, special_c = coord
        if tile_type == "vertical_laser":
            for row in range(self.num_rows):
                if self.tile_translator.is_special(self.board[row, special_c]) and (row, special_c) not in self.activation_q_coords:
                    self.activation_q_coords.add((row, special_c))
                    self.activation_q.append((coord, *self.tile_translator.get_type_colour(self.board[row, special_c])))
                self.board[row, special_c] = 0
        elif tile_type == "horizontal_laser":
            for col in range(self.num_cols):
                if self.tile_translator.is_special(self.board[special_r, col]) and (special_r, col) not in self.activation_q_coords:
                    self.activation_q_coords.add((special_r, col))
                    self.activation_q.append((coord, *self.tile_translator.get_type_colour(self.board[special_r, col])))
                self.board[special_r, col] = 0
            
        elif tile_type == "bomb":
            for i in range(coord[0] - 1, coord[0] + 2):
                for j in range(coord[1] - 1, coord[1] + 2):
                    if self.tile_translator.is_special(self.board[i, j]):
                        self.activation_q.append((i, j))
                    else:
                        self.board[i, j] = 0
        elif tile_type == "cookie":
            self.cookie(coord, tile_colour)
        

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
            print("corner = ", corner)
            std = [c for c in coords if not self.tile_translator.is_special(self.board[c])]
            print("std = ", std)
            if corner in std:
                return corner
            else:
                return sorted(std, key=lambda x: (x[0] - corner[0]) ** 2 + (x[1] - corner[1]) ** 2)[0]

        print("coords = ", coords)
        sorted_coords = sorted([c for c in coords if not self.tile_translator.is_special(self.board[c])], key=lambda x: (x[0], x[1]))
        print("sorted_coords = ", sorted_coords)
        print(len(sorted_coords))
        if len(sorted_coords) % 2 == 0:
            return sorted_coords[len(sorted_coords) // 2 - 1]
        return sorted_coords[len(sorted_coords) // 2]

# TODO: Rewrite gravity to not use transpose.
