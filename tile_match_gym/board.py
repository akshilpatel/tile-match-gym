import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import deque
from tile_translator import TileTranslator

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
        3n:     hstripe_tilen,
        3n+1:   bomb_tile1,
        ...,
        4n:     bomb_tilen,
        4n+1:   cookie
    }
    """

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
        self._special_match_types = ["vertical4", "horizontal4", "vertical5", "horizonta5","bomb"]
        # self.generate_board()
        self.board = self.np_random.integers(1, self.num_colours + 1, size=self.flat_size).reshape(self.rows, self.cols)
        self.activation_q = []

        self.indices = np.array([[(r, c) for r in range(0, cols)] for c in range(0, rows)])

        # handle the case where we are given a board
        if board is not None:
            self.board = board
            self.rows = len(board)
            self.cols = len(board[0])

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
        color_idx: Optional[int] = None,
        ) -> None:
        rand_coord = self.np_random.choice(list(filter(self.tile_translator.is_tile_ordinary, match_coords)))
        if color_idx is None:
            color_idx = self.tile_translator.get_tile_colour(rand_coord)
        if match_type in ["horizontal5", "vertical5"]:
            self.board[rand_coord] = int(self.num_colours * 4) + 1
        elif match_type == "horizontal4":
            self.board[rand_coord] = self.num_colours + color_idx
        elif match_type == "verticall4":
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
        # zero_counts = np.zeros((self.rows, self.cols))
        # for j, col in enumerate(self.board.T):
        #     zero_count = 0
        #     for i in range(len(col)-1, -1, -1):
        #         if col[i] == 0:
        #             zero_count += 1
        #         elif zero_count != 0:
        #             col[i + zero_count] = col[i]
        #             col[i] = 0

        #         zero_counts[i, j] = zero_count
        mask_T = self.board.T==0
        non_zero_mask_T = ~mask_T
        zero_counts_T = np.cumsum(mask_T[:, ::-1], axis=1)[::-1]

        for j, col in enumerate(self.board.T):
            self.board[:, j] = np.concatenate([col[mask_T[j]], col[non_zero_mask_T[j]]])

        # Update coordinates in activation queue
        if len(self.activation_q) != 0:
            for activation in self.activation_q:
                row, col = activation["coord"]
                activation["coord"][0] += zero_counts_T[col, row]
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
        ):
        """
        Should take a particular coordinate of the board.
        Get the activation  given the tile
        Update the activation queue if needed.
        Eliminate ordinary tiles if they are next in activation queue.
        If both coordinates are specials, the second_special_coord should not be None.
        """
        if activation_type == None:
            activation_type = self.tile_translator.get_tile_type(self.board[coord])
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
                coord_arr = (self.indices[min_top:max_bottom, min_left:max_right].reshape((-1, 2)).tolist())
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

    def check_move_validity(
        self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
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
        get_col = lambda x: "\033[1;3{}m{}\033[0m".format(x, x)
        print(" " + "-" * (self.cols * 2 + 1))
        for row in self.board:
            print("| ", end="")
            for tile in row:
                print(get_col(tile), end=" ")
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
        matches = self.get_match_coords()  # List of coordinates consisting a match.
        if len(matches) == 0:
            return False
        for match in matches:
            match_type = self.get_match_type(match)
            for match_coord in match:
                self.activation_q.append({"coord": match_coord})
            self.activation_loop()
            if match_type in self._special_match_types:
                self.create_special(match, match_type)
        return True

    ## Match functions ##

    def _sort_coords(self,l: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        return sorted([sorted(i, key=lambda x: (x[0], x[1])) for i in l])

    def get_tiles(self) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
        """
        Returns the types of tiles in the board and their locations
        """
        matches = self.get_lines()
        # islands = self.get_islands(matches)
        tile_coords, tile_names = self.get_matches([], matches)
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
                if not (row > 0 and self.board[row][el] == self.board[row-1][el]):
                    # check for vertical lines
                    while r < self.rows:
                        if self.board[r][el] == self.board[r-1][el]:
                            r += 1
                        else:
                            break
                    if r - row >= 3:
                        lines.append([(row + i, el) for i in range(r - row)])
                
                # make sure line has not already been checked
                if not (el > 0 and self.board[row][el] == self.board[row][el-1]):
                    # check for horizontal lines
                    while e < self.cols:
                        if self.board[row][e] == self.board[row][e-1]:
                            e += 1
                        else:
                            break
                    if e - el >= 3:
                        lines.append([(row, el + i) for i in range(e - el)])
        return lines

    def get_matches(self, islands: List[List[Tuple[int, int]]], lines: List[List[Tuple[int, int]]]) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
        """
        Detects the match type from the bottom up

        returns the match coordinates and the match type for each match in the island removed from bottom to top

        TODO: make this more efficient and include the islands so that
        concurrent groups can be matched
        """

        tile_names = []
        tile_coords = []
        
        lines = sorted([sorted(i, key=lambda x: (x[0],x[1])) for i in lines], key=lambda y: (y[0][0]), reverse=True)

        while len(lines) > 0:
            line = lines.pop()
            # check for cookie
            if len(line) >= 5:
                tile_names.append("cookie")
                tile_coords.append(line[:5])
                if len(line[5:]) > 2:
                    lines.append(line[5:]) # TODO - should just not pop the line rather than removing and adding again.
            # check for laser
            elif len(line) == 4:
                tile_names.append("laser")
                tile_coords.append(line)
            # check for bomb
            elif any([c in l for c in line for l in lines]): # TODO - REMOVE THIS AS SLOW AND IS DONE TWICE
                for l in lines:
                    shared = [c for c in line if c in l]
                    if any(shared):
                        shared = shared[0]
                        sorted_closest = sorted(l, key=lambda x: (abs(x[0]-shared[0]) + abs(x[1]-shared[1])))
                        tile_coords.append([p for p in line]+[p for p in sorted_closest[:3] if p not in line])
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
    
    @staticmethod
    def get_islands(lines: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        """
        Returns a list of islands from a list of lines

        TODO - Currently changes 'lines' in place. Should not do this.
        """
        # This can definitely be made faster
        islands = []
        for line in lines:
            # check if line is already in an island
            in_island = False
            for island in islands:
                for coord in line:
                    if coord in island:
                        in_island = True
                        break
                if in_island:
                    for coord in line:
                        if coord not in island:
                            island.append(coord)
            if not in_island:
                islands.append(line)
        return islands


if __name__ == "__main__":
    # board = Board(8, 7, 4)
    # board.board = board.np_random.integers(
    #     1, board.num_colours + 1, size=board.flat_size
    # ).reshape(board.rows, board.cols)

    # board.board = np.array(
    #     [
    #         [4, 4, 2, 1, 3, 2, 3],  # 0
    #         [3, 1, 1, 1, 2, 3, 4],  # 1
    #         [4, 4, 2, 1, 3, 2, 3],  # 2
    #         [2, 3, 2, 2, 2, 3, 2],  # 3
    #         [1, 1, 2, 3, 3, 3, 4],  # 4
    #         [1, 3, 2, 4, 1, 2, 4],  # 5
    #         [4, 3, 1, 3, 1, 4, 4],  # 6
    #         [3, 3, 3, 1, 1, 1, 4],  # 7
    #     ]
    # )
    # print(board.get_lowest_v_match_coords())
    # assert board.get_lowest_h_match_coords() == [
    #     [(7, 0), (7, 1), (7, 2)],
    #     [(7, 3), (7, 4), (7, 5)],
    # ]
    # print("hi")

    # board.board = np.array(
    #     [
    #         [4, 4, 2, 2, 2, 2, 3],  # 0
    #         [3, 1, 1, 2, 2, 3, 4],  # 1
    #         [4, 4, 2, 1, 3, 2, 3],  # 2
    #         [2, 3, 2, 2, 1, 3, 2],  # 3
    #         [1, 1, 2, 3, 2, 3, 3],  # 4
    #         [1, 3, 2, 4, 1, 2, 4],  # 5
    #         [4, 3, 1, 3, 1, 4, 4],  # 6
    #         [3, 2, 3, 2, 1, 1, 4],  # 7
    #     ]
    # )

    # print(board.get_lowest_h_match_coords())

    # board.board = np.array(
    #     [
    #         [4, 4, 4, 3, 2, 2, 2],  # 0
    #         [3, 1, 1, 2, 2, 3, 4],  # 1
    #         [4, 4, 2, 1, 3, 2, 3],  # 2
    #         [2, 3, 2, 2, 1, 3, 2],  # 3
    #         [1, 1, 2, 3, 2, 3, 3],  # 4
    #         [1, 3, 2, 4, 1, 2, 4],  # 5
    #         [4, 3, 1, 3, 1, 4, 4],  # 6
    #         [3, 2, 3, 2, 1, 1, 4],  # 7
    #     ]
    # )

    # print(board.get_lowest_h_match_coords())

    # print("original")
    # board.print_board()

    # # board.apply_activation((1, 1), 0)

    # # print("normal activation at (1,1)")
    # # board.print_board()

    # board.apply_activation((1, 1), 1)
    # print("v_stripe activation at (1,1)")
    # board.print_board()
    # print("board activation q = ", board.activation_q)

    # board.apply_activation((1, 1), 2)

    # print("h_stripe activation at (1,1)")
    # board.print_board()

    # board.apply_activation((2,2), 3)
    # print("bomb activation at (2,2)")
    # board.print_board()
    import json

    sort_coords = lambda l:sorted([sorted(i, key=lambda x: (x[0], x[1])) for i in l])
    coords_match = lambda l1, l2: sort_coords(l1) == sort_coords(l2)
    format_test = lambda r, e: "result: \t"+str(r)+"\nexpected: \t"+str(e)+"\n"

    boards = json.load(open("boards.json", "r"))["boards"]
    
    for board in boards:
        print("testing board: ", board['name'])
        bm = Board(0,0,0, board=np.array(board['board']))
        matches = bm.get_lines()
        expected_matches = [[tuple(coord) for coord in line] for line in board['matches']]
        expected_islands = [[tuple(coord) for coord in line] for line in board['islands']]
        expected_tile_coords = [[tuple(coord) for coord in line] for line in board['tile_locations']]
        expected_tile_names = board['tile_names']

        assert len(matches) == len(board['matches']), "incorrect number of matches found\n"+format_test(matches, expected_matches)
        assert coords_match(matches, expected_matches), "incorrect matches found\n"+format_test(matches, expected_matches)
        
        #islands = bm.get_islands(matches)
        #assert coords_match(islands, expected_islands), "incorrect islands found\n"+format_test(sort_coords(islands), sort_coords(expected_islands))
    
        tile_coords, tile_names = bm.get_matches([], matches)
        assert coords_match(tile_coords, expected_tile_coords), "incorrect tile coords found\n"+format_test(sort_coords(tile_coords), sort_coords(expected_tile_coords))
            
        # make sure that the tiles collected are correct and in the same order
        print(tile_names, expected_tile_names)
        assert all(
            [
                name == expected_name
                for name, expected_name in zip(tile_names, expected_tile_names)
            ]
        ), "incorrect tile names found\n" + format_test(tile_names, expected_tile_names)
        
        print("tile_coords = ", tile_coords)
        print("tile_names = ", tile_names)
        print("PASSED")
        print("get_tiles = ", bm.get_tiles())

        print("----")


