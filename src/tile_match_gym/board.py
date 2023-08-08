import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import deque

from tile_match_gym.tile_translator import TileTranslator

# temp while testing
# from tile_translator import TileTranslator  # temp while testing
from tile_match_gym.utils.print_board_diffs import highlight_board_diff


"""
    tile_TYPES = {
        1:      tile1 
        2:      tile2,
        ...
        n:      tilen,
        n+1:    vlaser_tile1,
        ...,
        2n:     vlaser_tilen,
        2n+1:   hlaser_tile1,
        ...,
        3n:     hlaser_tilen,
        3n+1:   bomb_tile1,
        ...,
        4n:     bomb_tilen,
        4n+1:   cookie
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
        seed: Optional[int] = None,
        board: Optional[np.ndarray] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.num_colours = num_colours

        self.tile_translator = TileTranslator(num_colours, (rows, cols))

        if seed is None:
            seed = np.random.randint(0, 1000000000)
        self.np_random = np.random.default_rng(seed)
        self.flat_size = int(self.cols * self.rows)
        self.num_actions = self.cols * (self.cols - 1) + self.rows * (self.rows - 1)
        self._special_match_types = {"horizontal_laser", "vertical_laser", "cookie", "bomb"}
        # self.generate_board()
        self.board = self.np_random.integers(1, self.num_colours + 1, size=self.flat_size).reshape(self.rows, self.cols)
        self.activation_q = []

        self.indices = np.array([[(r, c) for r in range(0, cols)] for c in range(0, rows)])

        # handle the case where we are given a board
        if board is not None:
            self.board = board
            self.rows = len(board)
            self.cols = len(board[0])

    def generate_board(self) -> None:
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
        color_idx: Optional[int] = None,
    ) -> None:
        rand_coord = self.np_random.choice(list(filter(self.tile_translator.is_tile_ordinary, match_coords)))
        if color_idx is None:
            color_idx = self.tile_translator.get_tile_colour(rand_coord)
        if match_type in ["horizontal_laser", "vertical_laser"]:
            self.board[rand_coord] = int(self.num_colours * 4) + 1
        elif match_type == "horizontal_laser":
            self.board[rand_coord] = self.num_colours + color_idx
        elif match_type == "vertical_laser":
            self.board[rand_coord] = int(2 * self.num_colours) + color_idx
        elif match_type == "bomb":
            self.board[rand_coord] = int(3 * self.num_colours) + color_idx
        else:
            raise NotImplementedError(f"The special type does not exist: {match_type}")

    def gravity(self) -> None:
        """
        Given a board with zeros, push the zeros to the top of the board.
        If an activation queue of coordinates is passed in, then the coordinates in the queue are updated as gravity pushes the coordinates down.
        """
        # Everything is done using the transpose since numpy indexing is faster row-wise
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
                print(row, col, row + zero_counts_T[col, row])

    def refill(self) -> None:
        """Replace all empty tiles."""
        zero_mask = self.board == 0
        num_zeros = zero_mask.sum()
        if num_zeros > 0:
            rand_vals = self.np_random.integers(1, self.num_colours + 1, size=num_zeros)
            self.board[zero_mask] = rand_vals

    def apply_activation(
        self,
        coord: Tuple[int, int],
        activation_type: Optional[int] = None,
        second_special_coord: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Should take a particular coordinate of the board, get the activation given the tile, resolve activating the coordinate and update the activation queue if needed.
        For efficiency, the method elminates ordinary tiles if they are next in activation queue.
        For now we also deal with the special case of if both coordinates are specials. This happens only when two specials are swiped together.

        Args:
            coord (Tuple[int, int]): The coordinate of the tile to activate.
            activation_type (Optional[int], optional): The type of activation. We could infer this by looking at the board, but this is faster.
                                                       Also, it allows us to apply an activation in a cell when the cell no longer contains the special item. Defaults to None.
            second_special_coord (Optional[Tuple[int, int]], optional): The coordinate of the second special tile. Defaults to None.
        """

        if activation_type == None:
            activation_type = self.tile_translator.get_tile_type(self.board[coord])
            if self.board[coord] == 0:
                return

        # Maximum one special in the move/activation.
        if second_special_coord is None:
            self.board[coord] = 0
            if activation_type == 1:  # v_laser
                self.activation_q += self.indices[:, coord[1]].reshape((-1, 2)).tolist()
            elif activation_type == 2:  # h_laser
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
            tile_type = self.tile_translator.get_tile_type(self.board[coord])
            tile_colour = self.tile_translator.get_tile_colour(self.board[coord])
            tile2_type = self.tile_translator.get_tile_type(self.board[second_special_coord])
            tile_colour = self.tile_translator.get_tile_colour(self.board[second_special_coord])
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

            if tile_type == 3:  # Bomb
                if tile2_type == 3:  # Two bombs
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

    def _check_same_colour(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
        tile1 = self.board[coord1]
        tile2 = self.board[coord2]
        return self.tile_translator.get_tile_colour(tile1) == self.tile_translator.get_tile_colour(tile2)

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
            return False, None
        if not (0 <= coord2[0] < self.rows and 0 <= coord2[1] < self.cols):
            return False, None

        # Extract a 6x6 grid around the coords to check for at least 3 match. This covers checking for Ls or Ts.
        y_ranges = max(0, coord1[0] - 2), min(self.rows, coord2[0] + 3)
        x_ranges = max(0, coord1[1] - 2), min(self.cols, coord2[1] + 3)
        surround_grid = self.board[y_ranges[0] : y_ranges[1]][x_ranges[0] : x_ranges[1]][:]

        # Swap the coordinates to see what happens.
        surround_grid[coord1], surround_grid[coord2] = surround_grid[coord2], self.board[coord1]
        # Doesn't matter what type of tile it is, if the colours match then its a match
        surround_grid %= self.num_colours
        for sg in [surround_grid, surround_grid.T]:
            for j in range(sg.shape[0]):
                for i in range(2, sg.shape[1]):
                    # If the current and previous 2 are matched and that they are not cookies.
                    if sg[j, i - 2] == sg[j, i - 1] == sg[j, i]:
                        return True, 0
        return False

    def move(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> None:
        if not self.check_move_validity(coord1, coord2):
            return
        self.board[coord1], self.board[coord2] = self.board[coord2], self.board[coord1]

        # If there are two special coords. Add one activation with both coords to the activation queue.
        if not self.tile_translator.is_tile_ordinary(self.board[coord1]) or self.tile_translator.is_tile_ordinary(self.board[coord1]):
            self.activation_q.append({"coord": coord1, "second_special_coord": coord2})
            self.activation_loop()
            return

        has_match = True
        while has_match:
            has_match = self.automatch()

    def print_board(self) -> None:
        # 2-5 is color1, 6-9 is color2, 10-13 is color3, 14-17 is color4, 18-21 is color5, 22-25 is color6
        get_color = lambda number, tile: (number - tile - 2) // self.tile_translator.num_specials + 1
        print_tile = lambda x, tile_type: "\033[1;3{}m{:2}\033[0m".format(get_color(x, tile_type), x)
        print(" " + "-" * (self.cols * 2 + 1))
        for row in self.board:
            print("| ", end="")
            for tile in row:
                print(print_tile(tile, 0), end=" ")
            print("|")
        print(" " + "-" * (self.cols * 2 + 1))

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
            if match_types[i] in self._special_match_types:
                self.create_special(match, match_types[i])
        return True

    ############################################################################
    ## Activation functions ##
    ############################################################################

    def activate_match(self, coords: List[Tuple[int, int]], name: str) -> None:
        """Activates a match of the given name at the given coordinates.

        Args:
            coords (Tuple[int, int]): Coordinates of the match.
            name (str): Name of the match.
        """
        if len(coords) == 3:
            for coord in coords:
                # TODO: Need to check this isnt special - if it is add to activation q
                self.board[coord] = 0
        elif len(coords) == 4:
            for coord in coords:
                # TODO: Need to check this isnt special - if it is add to activation q
                self.board[coord] = 0

            # TODO: this just selects the first coordinate but need to choose randomly that is not already special
            self.board[coords[0]] = self.tile_translator.get_tile_number(name, self.board[coords[0]])

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
        returns contiguous lines of 3 or more candies
        """
        lines = []
        for row in range(self.rows):
            for el in range(self.cols):
                r = row + 1
                e = el + 1

                # make sure line has not already been checked
                if not (row > 0 and self.board[row][el] == self.board[row - 1][el]):
                    # check for vertical lines
                    while r < self.rows:
                        if self.board[r][el] == self.board[r - 1][el]:
                            r += 1
                        else:
                            break
                    if r - row >= 3:
                        lines.append([(row + i, el) for i in range(r - row)])

                # make sure line has not already been checked
                if not (el > 0 and self.board[row][el] == self.board[row][el - 1]):
                    # check for horizontal lines
                    while e < self.cols:
                        if self.board[row][e] == self.board[row][e - 1]:
                            e += 1
                        else:
                            break
                    if e - el >= 3:
                        lines.append([(row, el + i) for i in range(e - el)])
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
            if len(line) >= 5:
                tile_names.append("cookie")
                tile_coords.append(line[:5])
                if len(line[5:]) > 2:
                    lines.append(line[5:])  # TODO - should just not pop the line rather than removing and adding again.
            # check for laser
            elif len(line) == 4:
                if line[0][0] == line[1][0]:
                    tile_names.append("horizontal_laser")
                else:
                    tile_names.append("vertical_laser")
                tile_coords.append(line)
            # check for bomb
            elif any([coord in l for coord in line for l in lines]):
                # elif any([c in l for c in line for l in lines]): # TODO - REMOVE THIS AS SLOW AND IS DONE TWICE
                for l in lines:
                    shared = [c for c in line if c in l]
                    if any(shared):
                        shared = shared[0]
                        sorted_closest = sorted(l, key=lambda x: (abs(x[0] - shared[0]) + abs(x[1] - shared[1])))
                        tile_coords.append([p for p in line] + [p for p in sorted_closest[:3] if p not in line])
                        if len(l) <= 6:
                            lines.remove(l)
                        for c in sorted_closest[:3]:
                            l.remove(c)
                        break
                tile_names.append("bomb")
            # check for normal
            elif len(line) == 3:
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

    for board in boards:
        print("testing board: ", board["name"])
        print("board: ", board["board"])

        bm = Board(0, 0, 0, board=np.array(board["board"]))
        matches = bm.get_lines()
        tile_coords, tile_names = bm.get_matches(matches)

        expected_matches = [[tuple(coord) for coord in line] for line in board["matches"]]
        expected_tile_coords = [[tuple(coord) for coord in line] for line in board["tile_locations"]]
        expected_tile_names = board["tile_names"]
        expected_cleared_board = np.array(board["cleared_board"])

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
            bm.print_board()
            assert np.array_equal(bm.board, expected_cleared_board), "incorrect board after activation\n" + highlight_board_diff(
                bm.board, expected_cleared_board
            )

        print("PASSED")

        print("----")
