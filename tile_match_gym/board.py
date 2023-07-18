import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import deque

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


class TileTranslator:
    def __init__(self, num_colours: int, board_shape: Tuple[int, int]):
        self.num_colours = num_colours
        self.board_shape = board_shape

    def _tile_type_str(self, tile_idx: int):
        tile_type = self.get_tile_type(tile_idx)
        if tile_type == 0:
            return "ordinary"
        elif tile_type == 1:
            return "vertical_stripe"
        elif tile_type == 2:
            return "horizontal_stripe"
        elif tile_type == 3:
            return "bomb"
        elif tile_type == 4:
            return "cookie"
        else:
            raise ValueError("Invalid tile type.")

    def is_tile_ordinary(self, tile_idx: int) -> bool:
        return (tile_idx - 1) < self.num_colours

    def get_tile_type(self, tile_idx: int) -> int:
        """
        Convert the tile index to whether the tile is ordinary, or which type of special it is.

        Args:
            tile_idx (int): Raw tile encoding.

        Returns:
            int: Index of tile type
        """
        return tile_idx // self.num_colours

    def get_tile_colour(self, tile_idx: int):
        return (tile_idx - 1) % self.num_colours

    def get_activation_effect(self, tile_idx):
        tile_type = self.get_tile_type(tile_idx)
        # Ordinary tiles are just deleted.
        if tile_type == 0:  # ordinary
            return 0
        elif tile_type == 1:  # horizontal
            return np.zeros(self.board_shape[1])
        elif tile_type == 2:  # vertical
            return np.zeros(self.board_shape[0])
        else:
            raise NotImplementedError("Have not implemented special tiles yet.")


class Board:
    def __init__(
        self,
        height: int,
        width: int,
        num_colours: int,
        seed: Optional[int] = None,
        board: Optional[np.ndarray] = None,
        ):
        self.height = height
        self.width = width
        self.num_colours = num_colours

        self.tile_translator = TileTranslator(num_colours, (height, width))

        if seed is None:
            seed = np.random.randint(0, 1000000000)
        self.np_random = np.random.default_rng(seed)
        self.flat_size = int(self.width * self.height)
        self.num_actions = self.width * (self.width - 1) + self.height * (self.height - 1)
        self._special_match_types = ["vertical4", "horizontal4", "vertical5", "horizonta5","bomb"]
        # self.generate_board()
        self.board = self.np_random.integers(1, self.num_colours + 1, size=self.flat_size).reshape(self.height, self.width)
        self.activation_q = []

        self.indices = np.array([[(r, c) for r in range(0, width)] for c in range(0, height)])

        # handle the case where we are given a board
        if board is not None:
            self.board = board
            self.height = board.shape
            self.width = board.shape[1]

    def generate_board(self):
        self.board = self.np_random.integers(1, self.num_colours + 1, size=self.flat_size).reshape(self.height, self.width)
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
        # zero_counts = np.zeros((self.height, self.width))
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
                max_bottom = min(self.height, coord[0] + 2)  # min of rightmost and width
                max_right = min(coord[1] + 2, self.width)  # min of bottommost and height
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
                        max_bottom = min(self.height, base_coord[0] + 3)  
                        min_left = max(base_coord[1] - 2, 0)  
                        max_right = min(base_coord[1] + 4, self.width)  
                        self.activation_q.extend([{"coord": x} for x in self.indices[min_top:max_bottom, min_left:max_right].reshape((-1, 2))])
                    else:  # Vertical match
                        base_coord = coord[0], min(coord[1], second_special_coord[1])
                        min_top = max(0, base_coord[0] - 2)  # max of 0 and leftmost bomb
                        max_bottom = min(self.height, base_coord[0] + 4)  # min of rightmost and width
                        min_left = max(base_coord[1] - 2, 0)  # max of 0 and topmost bomb
                        max_right = min(base_coord[1] + 3, self.width)  # min of bottommost and height
                        self.activation_q.extend([{"coord": x} for x in self.indices[min_top:max_bottom, min_left:max_right].reshape((-1, 2))])
                elif tile2_type <= 2:  # Bomb + laser
                    self.board[coord] = 0
                    self.board[second_special_coord] = 0
                    min_left = max(0, second_special_coord[0] - 1)
                    max_right = min(self.width, second_special_coord[0] + 2)
                    min_top = max(0, second_special_coord[1] - 1)
                    max_bottom = min(self.height, second_special_coord[1] + 2)
                    coord_arr = np.intersect1d(
                        self.indices[min_top:max_bottom, :].reshape((-1, 2)),
                        self.indices[:, min_left:max_right].reshape((-1, 2)),
                    )
                    self.activation_q.extend([{"coord": x} for x in coord_arr])

            elif tile2_type == 3:  # Bomb + laser
                self.board[coord] = 0
                self.board[second_special_coord] = 0
                min_left = max(0, coord[0] - 1)
                max_right = min(self.width, coord[0] + 2)
                min_top = max(0, coord[1] - 1)
                max_bottom = min(self.height, coord[1] + 2)
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
                raise ValueError(
                    f"We are ridden with bugs. candy1: {tile_type} candy2: {tile2_type}"
                )

    def get_match_coords(self) -> List[List[Tuple[int, int]]]:
        """For the current board, find the first set of matches. Go from the bottom up and find the set of matches.
        Returns:
            List[List[Tuple[int, int]]]: List of matches. Each match is specified by a list.

        # Look through the matches from bottom up and stop when you've checked the lowest row that has a match.
        # Do the same thing for vertical.
        # This currently only works for lines in one axis, i.e. we cannot detect Ls or Ts
        """
        h_matches, lowest_row_h = self.get_lowest_h_match_coords()
        v_matches, lowest_row_v = self.get_lowest_v_match_coords()
        if lowest_row_h == lowest_row_v == -1:
            return []
        # Check which matches are lowest and only return those.
        if lowest_row_h == lowest_row_v:
            # Check for bombs:
            for i, h_match in enumerate(h_matches):
                for j, v_match in enumerate(v_matches):
                    set_h_match = set(h_match)
                    set_v_match = set(v_match)            
            return h_matches + v_matches
        elif lowest_row_h > lowest_row_v:
            return h_matches
        else:
            return v_matches
        
    # def get_match_type(self, match_segment: List[Tuple[int, int]]) -> str:
    #     """String indicator of what match has occured.
    #     Args:
    #         match_islands (List[Tuple[int, int]]): Coords contained within a single match.
    #     Returns:
    #         str: Describing the match.
    #     """
    #     match_len = len(match_coords)
    #     if match_len == 3:
    #         if match_coords[0][0] == match_coords[1][0]:
    #             return "horizontal3"
    #         else:
    #             return "vertical3"

    #     if match_len >= 5:

    #     # Check for contiguous 5s in a line -> cookie. -> rip out the 
        
        
    #     # Check for contiguous 4s in a line

    #     # Else bomb.        
    #     coord_arr = np.vstack(match_coords)
    #     if len(match_coords) == 4:
    #         if np.all(coord_arr[:, 0] == coord_arr[0, 0]):
    #             return "horizontal4"
    #         elif np.all(coord_arr[:, 1] == coord_arr[0, 0]):
    #             return "vertical4"

    #     if match_len == 5:
    #         if match_coords[0][0] == match_coords[1][0]:
    #             return "horizontal5"
    #         else:
    #             return "vertical5"

    # Could use a mask to fix by setting those that have been added to a match to mask.
    def get_lowest_h_match_coords(self) -> List[List[Tuple[int, int]]]:
        h_matches = []
        lowest_row_h = -1
        # Check all horizontal matches starting from the bottom
        for row in range(self.height - 1, -1, -1):
            if lowest_row_h != -1:  # Don't need to check rows higher up.
                break
            col = 2
            while col < self.width:
                # If the current and previous 2 are matched
                if self.board[row, col - 2] == self.board[row, col - 1] == self.board[row, col]:
                    lowest_row_h = max(row, lowest_row_h)
                    start = (row, col - 2)
                    # Iterate through to find the full number of matched candies.
                    while col < self.width and self.board[row, col] == self.board[row, col - 1]:
                        col += 1
                    match = [(start[0], i) for i in range(start[1], col)]
                    h_matches.append(match)
                    col += 2
                else:
                    col += 1
        return h_matches, lowest_row_h

    # Could use a mask to fix by setting those that have been added to a match to mask.
    def get_lowest_v_match_coords(self) -> List[List[Tuple[int, int]]]:
        """
        Find the lowest vertical matches on the board starting from the bottom up.

        Returns:
            List[List[Tuple[int, int]]]: List of coordinates defining the vertical matches.
        """
        v_matches = []
        lowest_row_v = -1
        # Bottom left to top right
        row = self.height - 3
        while row >= 0:
            if lowest_row_v != -1:
                break
            for col in range(self.width):
                if self.board[row, col] == self.board[row + 1, col] == self.board[row + 2, col]:  # Found a match
                    lowest_row_v = max(row + 2, lowest_row_v)
                    match = [(row + 2, col), (row + 1, col), (row, col)]
                    m_search_row = row
                    while m_search_row > 0 and self.board[m_search_row, col] == self.board[m_search_row - 1, col]:
                        m_search_row -= 1
                        match.append((m_search_row, col))
                    v_matches.append(match)
            row -= 1
        return v_matches, lowest_row_v

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
        if not (0 <= coord1[0] < self.height and 0 <= coord1[1] < self.width):
            return False, None
        if not (0 <= coord2[0] < self.height and 0 <= coord2[1] < self.width):
            return False, None

        # Extract a 6x6 grid around the coords to check for at least 3 match. This covers checking for Ls or Ts.
        y_ranges = max(0, coord1[0] - 2), min(self.height, coord2[0] + 3)
        x_ranges = max(0, coord1[1] - 2), min(self.width, coord2[1] + 3)
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
        print(" " + "-" * (self.width * 2 + 1))
        for row in self.board:
            print("| ", end="")
            for tile in row:
                print(get_col(tile), end=" ")
            print("|")
        print(" " + "-" * (self.width * 2 + 1))

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


if __name__ == "__main__":
    board = Board(8, 7, 4)
    board.board = board.np_random.integers(
        1, board.num_colours + 1, size=board.flat_size
    ).reshape(board.height, board.width)

    board.board = np.array(
        [
            [4, 4, 2, 1, 3, 2, 3],  # 0
            [3, 1, 1, 1, 2, 3, 4],  # 1
            [4, 4, 2, 1, 3, 2, 3],  # 2
            [2, 3, 2, 2, 2, 3, 2],  # 3
            [1, 1, 2, 3, 3, 3, 4],  # 4
            [1, 3, 2, 4, 1, 2, 4],  # 5
            [4, 3, 1, 3, 1, 4, 4],  # 6
            [3, 3, 3, 1, 1, 1, 4],  # 7
        ]
    )
    print(board.get_lowest_v_match_coords())
    assert board.get_lowest_h_match_coords() == [
        [(7, 0), (7, 1), (7, 2)],
        [(7, 3), (7, 4), (7, 5)],
    ]
    print("hi")

    board.board = np.array(
        [
            [4, 4, 2, 2, 2, 2, 3],  # 0
            [3, 1, 1, 2, 2, 3, 4],  # 1
            [4, 4, 2, 1, 3, 2, 3],  # 2
            [2, 3, 2, 2, 1, 3, 2],  # 3
            [1, 1, 2, 3, 2, 3, 3],  # 4
            [1, 3, 2, 4, 1, 2, 4],  # 5
            [4, 3, 1, 3, 1, 4, 4],  # 6
            [3, 2, 3, 2, 1, 1, 4],  # 7
        ]
    )

    print(board.get_lowest_h_match_coords())

    board.board = np.array(
        [
            [4, 4, 4, 3, 2, 2, 2],  # 0
            [3, 1, 1, 2, 2, 3, 4],  # 1
            [4, 4, 2, 1, 3, 2, 3],  # 2
            [2, 3, 2, 2, 1, 3, 2],  # 3
            [1, 1, 2, 3, 2, 3, 3],  # 4
            [1, 3, 2, 4, 1, 2, 4],  # 5
            [4, 3, 1, 3, 1, 4, 4],  # 6
            [3, 2, 3, 2, 1, 1, 4],  # 7
        ]
    )

    print(board.get_lowest_h_match_coords())

    print("original")
    board.print_board()

    # board.apply_activation((1, 1), 0)

    # print("normal activation at (1,1)")
    # board.print_board()

    board.apply_activation((1, 1), 1)
    print("v_stripe activation at (1,1)")
    board.print_board()
    print("board activation q = ", board.activation_q)

    # board.apply_activation((1, 1), 2)

    # print("h_stripe activation at (1,1)")
    # board.print_board()

    # board.apply_activation((2,2), 3)
    # print("bomb activation at (2,2)")
    # board.print_board()
