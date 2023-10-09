import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import deque

# from tile_match_gym.tile_translator import TileTranslator
# from tile_match_gym.utils.print_board_diffs import highlight_board_diff

# temp while testing
from tile_translator import TileTranslator  # temp while testing
from utils.print_board_diffs import highlight_board_diff

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
        rows: int,
        cols: int,
        num_colours: int,
        colourless_specials: List[str] = ["cookie"],
        colour_specials: List[str] = ["vertical_laser", "horizontal_laser", "bomb"],
        seed: Optional[int] = None,
        board: Optional[np.ndarray] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.num_colours = num_colours

        self.flat_size = int(self.cols * self.rows)
        self.num_actions = self.cols * (self.cols - 1) + self.rows * (self.rows - 1)

        self.colourless_specials = colourless_specials
        self.colour_specials = colour_specials
        self.num_colour_specials = len(self.colour_specials)
        self.num_colourless_specials = len(self.colourless_specials)
        self.specials = set(self.colourless_specials + self.colour_specials)

        self.tile_translator = TileTranslator(num_colours, (rows, cols), self.colourless_specials, self.colour_specials)

        if seed is None:
            seed = np.random.randint(0, 1000000000)
        self.np_random = np.random.default_rng(seed)

        self.board = self.np_random.integers(1, self.num_colours + 1, size=self.flat_size).reshape(self.rows, self.cols)

        # handle the case where we are given a board
        if board is not None:
            self.board = board
            self.rows = len(board)
            self.cols = len(board[0])

        self.indices = np.array([[(r, c) for r in range(self.cols)] for c in range(self.rows)])
        self.activation_q = []

    def generate_board(self):
        self.board = self.np_random.integers(1, self.num_colours + 1, size=self.flat_size).reshape(self.rows, self.cols)
        has_match = True
        while has_match:
            has_match = self.automatch(scoring=False)
            self.gravity()
            self.refill()

    # Happens after all effects are done, you just put the special in place.
    def create_special(
        self,
        match_coords: List[Tuple[int, int]],
        match_type: str,
        colour_idx: Optional[int] = None,
    ) -> None:
        rand_coord = self.np_random.choice(list(filter(self.tile_translator.is_tile_ordinary, match_coords)))
        if colour_idx is None:
            colour_idx = self.tile_translator.get_tile_colour(rand_coord)
        if match_type in ["horizontal_laser", "vertical_laser"]:
            self.board[rand_coord] = int(self.num_colours * 4) + 1
        elif match_type == "horizontal_laser":
            self.board[rand_coord] = self.num_colours + colour_idx
        elif match_type == "vertical_laser":
            self.board[rand_coord] = int(2 * self.num_colours) + colour_idx
        elif match_type == "bomb":
            self.board[rand_coord] = int(3 * self.num_colours) + colour_idx
        else:
            raise NotImplementedError(f"The special type does not exist: {match_type}")

    def gravity(self) -> None:
        """
        Given a board with zeros, push the zeros to the top of the board.
        If an activation queue of coordinates is passed in, then the coordinates in the queue are updated as gravity pushes the coordinates down.
        """
        mask_T = self.board.T == 0
        non_zero_mask_T = ~mask_T
        zero_counts_T = np.cumsum(mask_T[:, ::-1], axis=1)[:, ::-1]

        for j, col in enumerate(self.board.T):
            self.board[:, j] = np.concatenate([col[mask_T[j]], col[non_zero_mask_T[j]]])

        # Update coordinates in activation queue
        if len(self.activation_q) != 0:
            for activation in self.activation_q:
                row, col = activation["coord"]
                activation["coord"] = (row + zero_counts_T[col, row], col)

    def refill(self, fixed=False) -> None:
        """Replace all empty tiles."""
        zero_mask = self.board == 0
        num_zeros = zero_mask.sum()
        if num_zeros > 0:
            rand_vals = self.np_random.integers(len(self.colourless_specials) + 1, self.num_colours + len(self.colourless_specials) + 1, size=num_zeros)
            self.board[zero_mask] = rand_vals

    
    def _check_same_colour(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
        tile1 = self.board[coord1]
        tile2 = self.board[coord2]
        return self.tile_translator.get_tile_colour(tile1) == self.tile_translator.get_tile_colour(tile2)

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
        if not (0 <= coord1[0] < self.rows and 0 <= coord1[1] < self.cols):
            return False
        if not (0 <= coord2[0] < self.rows and 0 <= coord2[1] < self.cols):
            return False
        # Check coords are next to each other.
        if not (coord1[0] == coord2[0] or coord1[1] == coord2[1]):
            return False

        # Checks if both are special.
        if self.tile_translator.is_special(self.board[coord1]) and self.tile_translator.is_special(self.board[coord2]):
            return True

        if self.tile_translator.is_colourless_special(self.board[coord1]) or self.tile_translator.is_colourless_special(self.board[coord2]):
            return True

        # Extract a minimal grid around the coords to check for at least 3 match. This covers checking for Ls or Ts.
        r_min = max(0, min(coord1[0] - 2, coord2[0] - 2))
        r_max = min(self.rows, max(coord1[0] + 3, coord2[0] + 3))
        c_min = max(0, min(coord1[1] - 2, coord2[1] - 2))
        c_max = min(self.cols, max(coord1[1] + 3, coord2[1] + 3))

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

    def move(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> None:
        if not self.check_move_validity(coord1, coord2):
            return
        self.board[coord1], self.board[coord2] = self.board[coord2], self.board[coord1]

        # If there are two special coords. Add one activation with both coords to the activation queue.
        coord1_special = self.tile_translator.is_special(self.board[coord1])
        coord2_special = self.tile_translator.is_special(self.board[coord2])


        # Combination match of two specials or one colourless special and any other tile.
        if coord1_special or coord2_special:
            if (coord1_special and coord2_special) or self.tile_translator.is_colourless_special(self.board[coord1]) or self.tile_translator.is_colourless_special(self.board[coord1]):
                self.activation_q.append({"coord": coord1, "second_special_coord": coord2})
            self.activation_loop()
            return

        has_match = True
        while has_match:
            has_match = self.automatch()

    def print_board(self) -> None:
        # 2-5 is colour1, 6-9 is colour2, 10-13 is colour3, 14-17 is colour4, 18-21 is colour5, 22-25 is colour6
        # get_colour = lambda number, tile: (number - tile - 2) // self.tile_translator.num_specials + 1
        get_colour = lambda number, tile: self.tile_translator.get_type_colour(number)[1]
        print_tile = lambda x, tile_type: "\033[1;3{}m{:>2}\033[0m".format(get_colour(x, tile_type) + 1, self.tile_translator._get_char(x))

        print(" " + "-" * (self.cols * 2 + 1))
        for row in reversed(self.board):
            print("| ", end="")
            for tile in row:
                print(print_tile(tile, 0), end=" ")
            print("|")
        print(" " + "-" * (self.cols * 2 + 1))

    def colour_check(self) -> None:
        get_char = lambda number: self.tile_translator._get_char(number)
        get_colour = lambda number, tile: self.tile_translator.get_type_colour(number)[1]
        print_tile = lambda x, tile_type: "\033[1;3{}m{:2}\033[0m".format(get_colour(x, tile_type), get_char(x))
        for i in range(0, self.num_colours + len(self.colour_specials) * self.num_colours + len(self.colourless_specials)):
            print(print_tile(i, 0), end=" ")
        print()

    def activation_loop(self) -> None:
        while len(self.activation_q) > 0:
            activation = self.activation_q.pop()
            self.apply_activation(**activation)
            self.gravity()
            self.refill()

            has_match = True
            while has_match:
                has_match = self.automatch()

    def automatch(self, scoring: Optional[bool] = False) -> bool:
        """Implements one round of automatching. Assumes and implements only one match per call.

        Args:
            scoring (Optional[bool], optional): Whether or not to accumulate the scoring function for this step. Defaults to False.

        Returns:
            bool: True iff the board has a match.
        """
        if scoring:
            raise NotImplementedError("Scoring functionality")
        matches, match_types = self.get_matches()  # List of coordinates consisting a match.
        if len(matches) == 0:
            return False
        for i, match in enumerate(matches):
            for match_coord in match:
                self.activation_q.append({"coord": match_coord})
            self.activation_loop()
            if match_types[i] in self.specials:
                self.create_special(match, match_types[i])
        return True

    ############################################################################
    ## Activation functions ##
    ############################################################################

    def apply_activation(
        self,
        coord: Tuple[int, int],
        activation_type: Optional[int] = None,
        second_special_coord: Optional[Tuple[int, int]] = None,
    ):
        """
        Should take a particular coordinate of the board.
        Get the activation  given the tile
        Update the activation queue if needed.
        Eliminate ordinary tiles if they are next in activation queue.
        If both coordinates are specials, the second_special_coord should not be None.



        ---
        If activation type is given with coord, then activate that activation type at that coord. 


        Otherwise, we have the following cases:
         
          1. Two special coords are present
          2. One colourless special is present.
          3. One coloured special coord is present.
          4. No special coords are present.


        # These first two only happen on taking a move.
        In case 1 we activate the special combination match according to which specials are present.
        In case 2, we combine the colourless special with ordinary tile and activate the combination match.

        # These last two happen either on taking a move or through automatch.
        In case 3, we activate the special coordinate and delete the rest. 
        In case 4, we delete the ordinary tiles.

        """



        # Unless a specific activation is wanted, we infer it from the existing tile, unless two special tiles are given.
        if activation_type is None:
            if second_special_coord is None:
                activation_type, _ = self.tile_translator.get_type_colour(self.board[coord])
                if self.board[coord] == 0:
                    return

        # Maximum one special in the move/activation.
        if second_special_coord is None:
            self.board[coord] = 0
            if activation_type == 1:  # v_stripe
                self.activation_q += self.indices[:, coord[1]].reshape((-1, 2)).tolist()
            elif activation_type == 2:  # h_stripe
                self.activation_q += self.indices[coord[0], :].reshape((-1, 2)).tolist()
            elif activation_type == 3:  # bomb
                min_top = max(0, coord[0] - 1)  # max of 0 and leftmost bomb
                min_left = max(coord[1] - 1, 0)  # max of 0 and topmost bomb
                max_bottom = min(self.rows, coord[0] + 2)  # min of rightmost and cols
                max_right = min(coord[1] + 2, self.cols)  # min of bottommost and rows
                coord_arr = self.indices[min_top:max_bottom, min_left:max_right].reshape((-1, 2)).tolist()
                self.activation_q += [{"coord": x for x in coord_arr}]
            # TODO: Add one clause here for if a cookie is hit.

        # This is for when two specials are combined.
        else:
            tile_type, tile_colour = self.tile_translator.get_type_colour(self.board[coord])            
            tile2_type, tile2_colour = self.tile_translator.get_type_colour(self.board[second_special_coord])
            

            if tile_type == 4:  # One cookie
                if tile2_type == 4:  # Two cookies
                    self.activation_q += self.indices.reshape(-1, 2).tolist()
                else:
                    self.board[coord] = 0
                    mask = (self.board != int(self.num_colours * 4) + 1) & (self.board % self.num_colours == tile_colour)  # Get same colour
                    self.board[mask] = self.board[second_special_coord]  # cookie

            if tile2_type == 4:  # One cookie
                self.board[coord] = 0
                mask = (self.board != int(self.num_colours * 4) + 1) & (self.board % self.num_colours == tile_colour)  # Get same colour
                self.board[mask] = self.board[coord]  # cookie

            # At least 1 bomb
            if tile_type == 3:
                # 2 bombs
                if tile2_type == 3:
                    self.board[coord] = 0
                    self.board[second_special_coord] = 0
                    if coord[0] == second_special_coord[0]:  # Horizontal match
                        base_coord = coord[0], min(coord[1], second_special_coord[1])
                        min_top = max(0, base_coord[0] - 2)
                        max_bottom = min(self.rows, base_coord[0] + 3)
                        min_left = max(base_coord[1] - 2, 0)
                        max_right = min(base_coord[1] + 4, self.cols)
                        self.activation_q.extend([{"coord": x} for x in self.indices[min_top:max_bottom, min_left:max_right].reshape((-1, 2))])
                    else:  # Vertical match
                        base_coord = coord[0], min(coord[1], second_special_coord[1])
                        min_top = max(0, base_coord[0] - 2)  # max of 0 and leftmost bomb
                        max_bottom = min(self.rows, base_coord[0] + 4)  # min of rightmost and cols
                        min_left = max(base_coord[1] - 2, 0)  # max of 0 and topmost bomb
                        max_right = min(base_coord[1] + 3, self.cols)  # min of bottommost and rows
                        self.activation_q.extend([{"coord": x} for x in self.indices[min_top:max_bottom, min_left:max_right].reshape((-1, 2))])
                elif tile2_type <= 2:  # Bomb + laser
                    self.board[coord] = 0
                    self.board[second_special_coord] = 0
                    min_left = max(0, second_special_coord[0] - 1)
                    max_right = min(self.cols, second_special_coord[0] + 2)
                    min_top = max(0, second_special_coord[1] - 1)
                    max_bottom = min(self.rows, second_special_coord[1] + 2)
                    coord_arr = np.intersect1d(
                        self.indices[min_top:max_bottom, :].reshape((-1, 2)),
                        self.indices[:, min_left:max_right].reshape((-1, 2)),
                    )
                    self.activation_q.extend([{"coord": x} for x in coord_arr])

            elif tile2_type == 3:  # Bomb + laser
                self.board[coord] = 0
                self.board[second_special_coord] = 0
                min_left = max(0, coord[0] - 1)
                max_right = min(self.cols, coord[0] + 2)
                min_top = max(0, coord[1] - 1)
                max_bottom = min(self.rows, coord[1] + 2)
                coord_arr = np.intersect1d(
                    self.indices[min_top:max_bottom, :].reshape((-1, 2)),
                    self.indices[:, min_left:max_right].reshape((-1, 2)),
                )
                self.activation_q.extend([{"coord": x} for x in coord_arr])

            elif tile_type <= 2:  # laser + laser
                self.board[coord] = 0
                self.board[second_special_coord] = 0
                coord_arr = np.intersect1d(
                    self.indices[second_special_coord[0], :].reshape((-1, 2)),
                    self.indices[:, second_special_coord[1]].reshape((-1, 2)),
                )
                self.activation_q.extend([{"coord": x} for x in coord_arr])
            else:
                raise ValueError(f"We are ridden with bugs. candy1: {tile_type} candy2: {tile2_type}")

    def apply_activation(self, coord: Tuple[int, int]) -> None:
        """Applies the activation at the given coordinates.

        Args:
            coords (Tuple[int, int]): Coordinates of the activation.
            second_special_coord (Optional[Tuple[int, int]], optional): If the activation is a special tile, the second special tile's coordinates. Defaults to None.
            
        """
        ttype, _ = self.tile_translator.get_type_colour(self.board[coord])
        # set the activated tile to 0
        self.board[coord] = 0
        if ttype == 3:  # vertical stripe
            # go through each coordinate in the column and set it to 0
            for i in range(self.rows):
                if self.tile_translator.is_special(self.board[i, coord[1]]):
                    self.activation_q.append((i, coord[1]))
                else:
                    self.board[i, coord[1]] = 0
        elif ttype == 4:  # horizontal stripe
            # go through each coordinate in the row and set it to 0
            for i in range(self.cols):
                if self.tile_translator.is_special(self.board[coord[0], i]):
                    self.activation_q.append((coord[0], i))
                else:
                    self.board[coord[0], i] = 0
        elif ttype == 5:  # bomb
            # go through each coordinate in the 3x3 square and set it to 0
            for i in range(coord[0] - 1, coord[0] + 2):
                for j in range(coord[1] - 1, coord[1] + 2):
                    if self.tile_translator.is_special(self.board[i, j]):
                        self.activation_q.append((i, j))
                    else:
                        self.board[i, j] = 0
        elif ttype == 1:  # cookie
            neighbours = [self.board[(coord[0] + i, coord[1] + j)] for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
            cols = [self.tile_translator.get_type_colour(n)[1] for n in neighbours if n != 0]
            most_common_col = max(set(cols), key=cols.count)
            # get the most common neighbour
            # remove all variants of that tile
            for i in range(self.rows):
                for j in range(self.cols):
                    col = self.tile_translator.get_type_colour(self.board[i, j])[1]
                    if col == most_common_col:
                        if self.tile_translator.is_special(self.board[i, j]):
                            self.activation_q.append((i, j))
                        else:
                            self.board[i, j] = 0

    def handle_activations(self):  # Used only for testing.
        while len(self.activation_q) > 0:
            activation = self.activation_q.pop()
            self.apply_activation(activation)
            self.print_board()
            self.gravity()
            self.refill()

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

    def activate_match(self, coords: List[Tuple[int, int]], name: str) -> None:
        """Activates a match of the given name at the given coordinates.

        Args:
            coords (Tuple[int, int]): Coordinates of the match.
            name (str): Name of the match.

        Don't add normal tiles to the activation queue.
        """
        if len(coords) == 3:
            print("3 match")
            for coord in coords:
                # TODO: Need to check this isnt special - if it is add to activation q
                if not self.tile_translator.is_special(self.board[coord]):
                    self.board[coord] = 0
                else:
                    self.activation_q.append(coord)
        elif len(coords) == 4:
            print("4 match")
            spec_coord = self.get_special_creation_pos(coords)
            self.board[spec_coord] += 1 if name == "vertical_laser" else 2
            for coord in coords:
                if coord != spec_coord:
                    # TODO: Need to check this isnt special - if it is add to activation q
                    if not self.tile_translator.is_special(self.board[coord]):
                        self.board[coord] = 0
                    else:
                        self.activation_q.append(coord)
        elif len(coords) == 5:  # cookie or bomb
            print("Cookie or bomb")
            # checks if a single line
            if all([i[0] == coords[0][0] for i in coords[1:]]) or all([i[1] == coords[0][1] for i in coords[1:]]):  # cookie
                new_spec_location = self.get_special_creation_pos(coords)
                self.board[new_spec_location] = 1
            else:  # bomb
                new_spec_location = self.get_special_creation_pos(coords, straight=False)
                self.board[new_spec_location] = 3 * self.board[new_spec_location] + 1

            for coord in coords:
                if coord != new_spec_location:
                    if not self.tile_translator.is_special(self.board[coord]):
                        self.board[coord] = 0
                    else:
                        self.activation_q.append(coord)
            # TODO: this just selects the first coordinate but need to choose randomly that is not already special
            # self.board[coords[0]] = self.tile_translator.get_tile_number(name, self.board[coords[0]])
        # self.print_board()

    ############################################################################
    ## Match functions ##
    ############################################################################

    def _sort_coords(self, l: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        return sorted([sorted(i, key=lambda x: (x[0], x[1])) for i in l])

    def get_tiles(self) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
        """
        Returns the types of tiles in the board and their locations
        """
        lines = self.get_lines()
        tile_coords, tile_names = self.get_matches(lines)
        return tile_coords, tile_names

    def get_lines(self) -> List[List[Tuple[int, int]]]:
        """
        Starts from the bottom and checks for 3 or more in a row vertically or horizontally.
        returns contiguous lines of 3 or more candies.
        Stops once it finds the lowest row with a match and only takes lines that start on that row.
        """
        lines = []
        for row in range(self.rows):
            for el in range(self.cols):
                r = row + 1
                e = el + 1

                # make sure line has not already been checked
                # if not (row > 0 and self.board[row][el] == self.board[row-1][el]):
                if not (row > 0 and self.tile_translator.is_same_colour(self.board[row][el], self.board[row - 1][el]) or self.board[row][el] == 1): # TODO: Take out  the or self.board[row][el] == 1 since we dont want to check for cookies
                    # check for vertical lines
                    while r < self.rows:
                        # if self.board[r][el] == self.board[r-1][el]:
                        # if same colour or cookie
                        if self.tile_translator.is_same_colour(self.board[r][el], self.board[r - 1][el]) or self.board[r][el] == 1:
                            r += 1
                        else:
                            break
                    if r - row >= 3:
                        lines.append([(row + i, el) for i in range(r - row)])

                # make sure line has not already been checked
                # if not (el > 0 and self.board[row][el] == self.board[row][el-1]):
                if not (el > 0 and self.tile_translator.is_same_colour(self.board[row][el], self.board[row][el - 1]) or self.board[row][el] == 1):
                    # check for horizontal lines
                    while e < self.cols:
                        # if self.board[row][e] == self.board[row][e-1]:
                        if (
                            self.tile_translator.is_same_colour(self.board[row][e], self.board[row][e - 1])
                            or self.board[row][e] == 1
                            or self.board[row][e - 1] == 1
                        ):
                            e += 1
                        else:
                            break
                    if e - el >= 3:
                        lines.append([(row, el + i) for i in range(e - el)])
        print("lines = ", lines)
        return lines

    def get_matches(self, lines: List[List[Tuple[int, int]]]) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
        """
        Detects the match type from the bottom up

        returns the match coordinates and the match type for each match in the island removed from bottom to top

        Note: concurrent groups can be matched
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
                            tile_coords.append([p for p in line] + [p for p in sorted_closest[:3] if p not in line]) # TODO: Change this to also only extract 3 closest to intersection from line.
                            tile_names.append("bomb")        
                            if len(l) < 6 : # Remove the other line if shorter than 3 after extracting bomb.
                                lines.remove(l)
                            else:
                                for c in sorted_closest[:3]: # Remove the coordinates that were taken for the bomb
                                    l.remove(c)
                            break # Stop searching after finding one intersection. This should break out of the for loop.
            # Check for normals. This happends even if the lines are longer than 3 but there are no matching specials.
            elif len(line) >= 3:
                tile_names.append("norm")
                tile_coords.append(line)
            # check for no match
            else:
                tile_names.append("ERR")
                tile_coords.append(line)

        return tile_coords, tile_names


if __name__ == "__main__":
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