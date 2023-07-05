import numpy as np
from typing import Optional, List, Tuple
from collections import deque
"""
    tile_TYPES = {
        1:      tile1 
        2:      tile2,
        ...
        n:      tilen,
        n+1:    vstripe_tile1,
        ...,
        2n:     vstripe_tilen,
        2n+1:   hstripe_tile1,
        ...,
        3n:     hstripe_tilen,
        3n+1:   bomb_tile1,
        ...,
        4n:     bomb_tilen,
        4n+1:   cookie
    }
    """

class TileTranslator:
    def __init__(self, num_colours:int, board_shape: Tuple[int, int]):
        self.num_colours = num_colours
        self.board_shape = board_shape

    def _tile_type_str(self, tile_idx:int):
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

    def get_tile_type(self, tile_idx:int) -> int:
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
        if tile_type == 0: # ordinary 
            return 0
        elif tile_type == 1: # horizontal
            return np.zeros(self.board_shape[1])
        elif tile_type == 2: # vertical
            return np.zeros(self.board_shape[0])
        else:
            raise NotImplementedError("Have not implemented special tiles yet.")

class Board:
    def __init__(self, height: int, width:int, num_colours:int, seed:Optional[int] = None, board:Optional[np.ndarray] = None):
        self.height = height
        self.width = width
        self.num_colours = num_colours

        self.tile_translator = TileTranslator(num_colours, (height, width))

        if seed is None:
            seed = np.random.randint(0, 1000000000)
        self.np_random = np.random.default_rng(seed)
        self.flat_size = int(self.width * self.height)
        self.num_actions = self.width * (self.width - 1) + self.height * (self.height - 1)
        self._special_match_types = ["vertical4", "horizontal4", "vertical5", "horizontal5", "bomb"]
        # self.generate_board()
        self.board = self.np_random.integers(1, self.num_colours + 1, size = self.flat_size).reshape(self.height, self.width)
        self.activation_q = []

        self.indices = np.array([[(r,c) for r in range(0,width)] for c in range(0,height)])

        # handle the case where we are given a board
        if board is not None:
            self.board = board
            self.height = board.shape
            self.width = board.shape[1]

    def generate_board(self):
        self.board = self.np_random.integers(1, self.num_colours + 1, size = self.flat_size).reshape(self.height, self.width)
        has_match = True
        while has_match:
            has_match = self.automatch(scoring = False)
            self.gravity()
            self.refill()
        
    def automatch(self, scoring: Optional[bool] = False) -> bool:
        """Implements one round of automatching. Assumes and implements only one match per call.

        Args:
            scoring (Optional[bool], optional): Whether or not to accumulate the scoring function for this step. Defaults to False.

        Returns:
            bool: True iff the board has a match.
        """
        if scoring:
            raise NotImplementedError("Scoring functionality")
        matches = self.get_match_coords() # List of coordinates consisting a match.
        if len(matches) > 0: # If there are matches
            for match in matches:
                match_type = self.get_match_type(match)
                if match_type in self._special_match_types:
                    self.create_special(match, match_type)
                return True
        else:
            return False
        
    # Happens after all effects are done, you just put the special in place.
    def create_special(self, match_coords: List[Tuple[int, int]], match_type: str, color_idx: Optional[int]=None) -> None: 
        rand_coord = self.np_random.choice(list(filter(self.tile_translator.is_tile_ordinary, match_coords)))
        if color_idx is None:
            color_idx = self.tile_translator.get_tile_colour(rand_coord)
        if match_type in ["horizontal5", "vertical5"]:
            self.board[rand_coord] = int(self.num_colours * 4) + 1
        elif match_type == "horizontal4": 
            self.board[rand_coord] = self.num_colours + color_idx
        elif match_type == "verticall4": 
            self.board[rand_coord] = int(2*self.num_colours) + color_idx
        elif match_type == "bomb":
            self.board[rand_coord] = int(3*self.num_colours) + color_idx
        else:
            raise NotImplementedError(f"The special type does not exist: {match_type}")

    def gravity(self, activation_queue: Optional[deque] = None) -> None:
        """
        Given a board with zeros, push the zeros to the top of the board.
        If an activation queue of coordinates is passed in, then the coordinates in the queue are updated as gravity pushes the coordinates down.
        Args:
            activation_queue: A queue of activations to be applied to the board. The coordinates in these must be updated.
        """
        board_transposed = self.board.T
        zero_mask = board_transposed == 0
        non_zero_mask = ~zero_mask

        # Compute the number of zeros pushed to the top at each position
        zero_counts = np.cumsum(zero_mask, axis=0)
        # Shift non-zero elements to their correct positions
        board_transposed[non_zero_mask] = board_transposed[non_zero_mask][zero_counts[non_zero_mask] - 1]
        board_transposed[zero_mask] = 0

        # Update coordinates in activation queue
        if activation_queue is not None:
            for activation in activation_queue:
                i, col = activation["coord"]
                activation["coord"][0] += zero_counts[i, col]
        self.board = board_transposed.T

    def refill(self) -> None:
        """Replace all empty tiles."""
        zero_mask = self.board == 0
        num_zeros = zero_mask.sum()
        if num_zeros > 0:
            rand_vals = self.np_random.integers(1, self.num_colours + 1, size=num_zeros)
            self.board[zero_mask] = rand_vals

    def apply_activation(self, coord: Tuple[int, int], activation_type: Optional[int]=None, special_coord: Optional[Tuple[int, int]]=None):
        """
        Should take a particular coordinate of the board.
        Get the activation effect given the tile
        """
        if activation_type == None:
            activation_type = self.tile_translator.get_tile_type(self.board[coord])
            if self.board[coord] == 0:
                return
        
        if special_coord is None:
            self.board[coord] = 0
            if activation_type == 1: # v_stripe
                self.activation_q += self.indices[:, coord[1]].reshape((-1,2)).tolist()
            elif activation_type == 2: # h_stripe
                self.activation_q += self.indices[coord[0], :].reshape((-1,2)).tolist()
            elif activation_type == 3: # bomb
                min_top = max(0,coord[0]-1) # max of 0 and leftmost bomb
                min_left = max(coord[1]-1, 0) # max of 0 and topmost bomb
                max_bottom = min(self.height, coord[0]+2) # min of rightmost and width
                max_right = min(coord[1]+2, self.width) # min of bottommost and height
                self.activation_q += self.indices[min_top:max_bottom, min_left:max_right].reshape((-1,2)).tolist()
        else: 
            tile_type = self.tile_translator.get_tile_type(self.board[coord])
            tile_colour = self.tile_translator.get_tile_colour(self.board[coord])
            print("special_coord", special_coord)
            print("self.board[special_boy] = ", self.board[special_coord])
            tile2_type = self.tile_translator.get_tile_type(self.board[special_coord])
            tile_colour = self.tile_translator.get_tile_colour(self.board[special_coord])
            if tile_type == 4: # One cookie
                if tile2_type == 4: # Two cookies
                    self.activation_q += self.indices.reshape(-1, 2).tolist()
                else: 
                    self.board[coord] = 0
                    mask = (self.board != int(self.num_colours * 4) + 1) & (self.board % self.num_colours == tile_colour) # Get same colour
                    self.board[mask] = self.board[special_coord] # cookie

            if tile2_type == 4: # One cookie
                self.board[coord] = 0
                mask = (self.board != int(self.num_colours * 4) + 1) & (self.board % self.num_colours == tile_colour) # Get same colour
                self.board[mask] = self.board[coord] # cookie
            
            if tile_type == 3: # Bomb
                if tile2_type == 3: # Two bombs
                    self.board[coord] = 0
                    self.board[special_coord] = 0
                    if coord[0] == special_coord[0]: # Horizontal match 
                        base_coord = coord[0], min(coord[1], special_coord[1])
                        min_top = max(0, base_coord[0]-2) # max of 0 and leftmost bomb
                        max_bottom = min(self.height, base_coord[0]+3) # min of rightmost and width
                        min_left = max(base_coord[1]-2, 0) # max of 0 and topmost bomb
                        max_right = min(base_coord[1] + 4, self.width) # min of bottommost and height
                        self.activation_q += self.indices[min_top:max_bottom, min_left:max_right].reshape((-1,2)).tolist()
                    else: # Vertical match
                        base_coord = coord[0], min(coord[1], special_coord[1])
                        min_top = max(0, base_coord[0]-2) # max of 0 and leftmost bomb
                        max_bottom = min(self.height, base_coord[0]+4) # min of rightmost and width
                        min_left = max(base_coord[1]-2, 0) # max of 0 and topmost bomb
                        max_right = min(base_coord[1] + 3, self.width) # min of bottommost and height
                        self.activation_q += self.indices[min_top:max_bottom, min_left:max_right].reshape((-1,2)).tolist()
                elif tile2_type <= 2: # Bomb + stripe
                    self.board[coord] = 0
                    self.board[special_coord] = 0
                    min_left = max(0, special_coord[0]-1)
                    max_right = min(self.width, special_coord[0]+2)
                    min_top = max(0, special_coord[1]-1)
                    max_bottom = min(self.height, special_coord[1]+2)
                    self.activation_q += np.intersect1d(self.indices[min_top:max_bottom, :].reshape((-1,2)), self.indices[:, min_left:max_right].reshape((-1,2))).tolist()
            
            elif tile2_type == 3: # Bomb + stripe
                self.board[coord] = 0
                self.board[special_coord] = 0
                min_left = max(0, coord[0]-1)
                max_right = min(self.width, coord[0]+2)
                min_top = max(0, coord[1]-1)
                max_bottom = min(self.height, coord[1]+2)
                self.activation_q += np.intersect1d(self.indices[min_top:max_bottom, :].reshape((-1,2)), self.indices[:, min_left:max_right].reshape((-1,2))).tolist()
            
            elif tile_type <= 2: # Stripe + stripe
                self.board[coord] = 0
                self.board[special_coord] = 0
                self.activation_q += np.intersect1d(self.indices[special_coord[0], :].reshape((-1,2)), self.indices[:, special_coord[1]].reshape((-1,2))).tolist()

            else:
                raise ValueError(f"We are ridden with bugs. candy1: {tile_type} candy2: {tile2_type}")
    def get_match_coords(self) -> List[List[Tuple[int, int]]]:    
        """For the current board, find the first set of matches. Go from the bottom up and find the set of matches. 

        Returns:
            List[List[Tuple[int, int]]]: List of coordinates defining the match.

        # Look through the matches from bottom up and stop when you've checked the lowest row that has a match.
        # Do the same thing for vertical.
        """
        h_matches, lowest_row_h = self.get_lowest_h_match_coords()
        v_matches, lowest_row_v = self.get_lowest_v_match_coords()
        if lowest_row_h == lowest_row_v == -1:
            return []
        # Check which matches are lowest and only return those.
        if lowest_row_h == lowest_row_v:
            return h_matches + v_matches
        elif lowest_row_h > lowest_row_v:
            return h_matches
        else:
            return v_matches

    def get_match_type(self, match_coords: List[Tuple[int, int]]) -> str:
        """String indicator of what match has occured.
        Args:
            match_coords (List[Tuple[int, int]]): Coords contained within a single match.
        Returns:
            str: Describing the match.
        """
        if len(match_coords) == 5:
            if match_coords[0][0] == match_coords[1][0]:
                return "horizontal5"
            else:
                return "vertical5"
        
        if len(match_coords) == 4:
            if match_coords[0][0] == match_coords[1][0]:
                return "horizontal4"
            else:
                return "vertical4"
        
        if len(match_coords) == 3:
            if match_coords[0][0] == match_coords[1][0]:
                return "horizontal3"
            else:
                return "vertical3"

    def _check_same_colour(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
        tile1 = self.board[coord1]
        tile2 = self.board[coord2]
        return self.tile_translator.get_tile_colour(tile1) == self.tile_translator.get_tile_colour(tile2)
    
    # Could use a mask to fix by setting those that have been added to a match to mask.
    def get_lowest_h_match_coords(self) -> List[List[Tuple[int, int]]]:
        h_matches = []
        lowest_row_h = -1
        # Check all horizontal matches starting from the bottom
        for row in range(self.height - 1, -1, -1):
            if lowest_row_h != -1: # Don't need to check rows higher up.
                break
            col = 2
            while col < self.width:
                # If the current and previous 2 are matched
                if self.board[row, col-2] == self.board[row, col-1] == self.board[row, col]:
                    lowest_row_h = max(row, lowest_row_h)
                    start = (row, col-2)
                    # Iterate through to find the full number of matched candies. 
                    while col < self.width and self.board[row, col] == self.board[row, col-1]:
                        col +=1
                    match = [(start[0], i) for i in range(start[1], col)]
                    h_matches.append(match)
                    col += 2
                else:
                    col +=1
        return h_matches

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
                if self.board[row, col] == self.board[row + 1, col] == self.board[row + 2, col]: # Found a match
                    lowest_row_v = max(row + 2, lowest_row_v)
                    match = [(row + 2, col), (row + 1, col), (row, col)]                 
                    m_search_row = row
                    while m_search_row > 0 and self.board[m_search_row, col] == self.board[m_search_row-1, col]:
                        m_search_row -= 1
                        match.append((m_search_row, col))
                    v_matches.append(match)        
            row -=1
        return v_matches

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
        if not (0 <= coord1[0] < self.height and 0 <= coord1[1] < self.width):
            return False, None
        if not (0 <= coord2[0] < self.height and 0 <= coord2[1] < self.width):
            return False, None

        # Extract a 6x6 grid around the coords to check for at least 3 match. This covers checking for Ls or Ts.
        y_ranges = max(0, coord1[0] - 2), min(self.height, coord2[0] + 3)
        x_ranges = max(0, coord1[1] - 2), min(self.width, coord2[1] + 3)
        surround_grid = self.board[y_ranges[0]: y_ranges[1]][x_ranges[0]: x_ranges[1]][:]
        
        # Swap the coordinates to see what happens.
        surround_grid[coord1], surround_grid[coord2] = surround_grid[coord2], self.board[coord1]                
        # Doesn't matter what type of tile it is, if the colours match then its a match
        surround_grid %= self.num_colours 
        for sg in [surround_grid, surround_grid.T]:
            for j in range(sg.shape[0]):
                for i in range(2, sg.shape[1]):
                    # If the current and previous 2 are matched and that they are not cookies.
                    if sg[j, i-2] == sg[j, i-1] == sg[j, i]:
                        return True, 0
        return False

    def move(self, coord1: Tuple[int, int], coord2:Tuple[int, int]) -> None:
        if not self.check_move_validity(coord1, coord2):
            return
        self.board[coord1], self.board[coord2] = self.board[coord2], self.board[coord1]
        has_match = True
        while has_match:
            has_match = self.automatch()
            self.gravity()
            self.refill()

    def print_board(self) -> None:
        get_col = lambda x: "\033[1;3{}m{}\033[0m".format(x, x)
        print(' ' + '-' * (self.width * 2 + 1))
        for row in self.board:
            print('| ', end='')
            for tile in row:
                print(get_col(tile), end=' ')
            print('|')
        print(' ' + '-' * (self.width * 2 + 1))

    def activation_loop(self, activation_queue: deque):
        while activation_queue:
            activation = activation_queue.pop()
            self.apply_activation(**activation)
            self.gravity(activation_queue)
            self.refill()

        has_match = True
        while has_match:
            has_match = self.automatch()

# def apply_activation(coord, activation_type=None, special_coords=None)

#     if activation_type is None:
#         """This function takes an activation type and applies it to board at the specified coordinate"""
#         if coord is 0:
#             return
#         if coord is normal:
#             delete_coord
#         elif coord == v_tstripe:
#             delete_coord
#             add_v_slice_to_activation_queue
#         elif coord == h_stripe:
#             delete_coord
#             add_h_slice_to_activation_queue
#         elif coord == bomb:
#             delete coord
#             add bomb slice to activation_queue	

#     # activation is cookie + special:
#     if activation_type == cookie + v_stripe + colour:
#         delete cookie_coord
#         turn all same colour into v or h stripe
#         add those coords to the activation queue in random order.

#     if activation_type == v_stripe + h_stripe:
#         delete both coords.
#         add vslice and h_slice to activation queue

#     if activation_type == stripe + bomb:
#         delete both coords.
#         add 3 vslices and 3 hslices to activation_queue

#     if activation_type == bomb + bomb:
#         delete both bombs
#         add 5x5 grid to queue

#     if activation_type == cookie + bomb:
#         delete cookie coord
#         turn all same colour into bomb
#         add those bombs to activation queue

#     if activation_type == cookie + cookie:
#         delete cookies 
#         Add all coords to activation queue



# while activation_queue:
#     activation = activation_queue.pop()
#     gravity(activation_queue) # This also updates the activation_coords in the activation_queue
#     refill()

#     update_activation_queue() # This adds new activations (resulting from the call to gravity and refill) to the activation_queue
#     # maybe has to call automatch()
#     # get_match_type -> match_type
#     # get_match_coords -> 


if __name__ == "__main__":
    board = Board(8, 7, 4)
    board.board = board.np_random.integers(1, board.num_colours + 1, size = board.flat_size).reshape(board.height, board.width)


    board.board = np.array([
        [4, 4, 2, 1, 3, 2, 3], # 0
        [3, 1, 1, 1, 2, 3, 4], # 1
        [4, 4, 2, 1, 3, 2, 3], # 2
        [2, 3, 2, 2, 2, 3, 2], # 3
        [1, 1, 2, 3, 3, 3, 4], # 4
        [1, 3, 2, 4, 1, 2, 4], # 5
        [4, 3, 1, 3, 1, 4, 4], # 6
        [3, 3, 3, 1, 1, 1, 4], # 7
    ])
    print(board.get_lowest_v_match_coords())
    assert board.get_lowest_h_match_coords() == [[(7, 0), (7, 1), (7, 2)], [(7, 3), (7, 4), (7, 5)]]
    print("hi")


    board.board = np.array([
        [4, 4, 2, 2, 2, 2, 3], # 0
        [3, 1, 1, 2, 2, 3, 4], # 1
        [4, 4, 2, 1, 3, 2, 3], # 2
        [2, 3, 2, 2, 1, 3, 2], # 3
        [1, 1, 2, 3, 2, 3, 3], # 4
        [1, 3, 2, 4, 1, 2, 4], # 5
        [4, 3, 1, 3, 1, 4, 4], # 6
        [3, 2, 3, 2, 1, 1, 4], # 7
    ])

    print(board.get_lowest_h_match_coords())
    

    board.board = np.array([
        [4, 4, 4, 3, 2, 2, 2], # 0
        [3, 1, 1, 2, 2, 3, 4], # 1
        [4, 4, 2, 1, 3, 2, 3], # 2
        [2, 3, 2, 2, 1, 3, 2], # 3
        [1, 1, 2, 3, 2, 3, 3], # 4
        [1, 3, 2, 4, 1, 2, 4], # 5
        [4, 3, 1, 3, 1, 4, 4], # 6
        [3, 2, 3, 2, 1, 1, 4], # 7
    ])

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

