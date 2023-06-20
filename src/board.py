import numpy as np
from typing import Optional, List, Tuple

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
        if tile_type > 0:
            raise NotImplementedError("Have not implemented special tiles yet.")
        if tile_type == 0:
            return "ordinary"
            
    def get_tile_type(self, tile_idx:int) -> int:
        """
        Convert the tile index to whether the tile is ordinary, or which type of special it is.

        Args:
            tile_idx (int): Raw tile encoding.

        Returns:
            int: Index of tile type
        """
        if (tile_idx - 1) >= self.num_colours:
            raise NotImplementedError("Have not implemented special tiles yet.")
        return tile_idx // self.num_colours

    def get_tile_colour(self, tile_idx: int):
        if tile_idx -1 >= self.num_colours:
            raise NotImplementedError("Have not implemented special tiles yet.")
        
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
        
        # self.generate_board()
        self.board = self.np_random.integers(1, self.num_colours + 1, size = self.flat_size).reshape(self.height, self.width)

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
        
        matches = self.get_match_coords()
        if len(matches) > 0:
            for match in matches:
                match_type = self.get_match_type(match)
                self.clear_coords(match, match_type)
                return True
        else:
            return False
        
    def gravity(self) -> None:
        """Push empty slots to the top."""
        for col in self.board.T:
            zero_count = 0
            for i in range(len(col)-1, -1, -1):
                if col[i] == 0:
                    zero_count += 1
                elif zero_count != 0:
                    col[i + zero_count] = col[i]
                    col[i] = 0


    def refill(self) -> None:
        """
        Search top to bottom in each column and break if you hit something that isn't zero.
        Since the board should
        """
        for col in self.board.T:
            for i in range(len(col)):
                if col[i] == 0:
                    col[i] = self.np_random.integers(1, self.num_colours + 1, size=1)
                else:
                    break
    
    
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
        """STring indicator of what match has occured.

        Args:
            match_coords (List[Tuple[int, int]]): Coords contained within a single match.

        Returns:
            str: Describing the match.
        """
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
    
    def clear_coords(self, match_coords: List[Tuple[int, int]], match_type: str):
        if match_type in ["horizontal3", "vertical3"]:
            self.board[np.array(match_coords)] = 0
        else:
            raise NotImplementedError("Special tiles not yet implemented")
    
    def acivate_old_specials(self, match_coords: List[Tuple[int, int]]):
        effects_masks = []
        for coord in match_coords:
            tile_type = self.tile_translator.get_tile_type(coord)
            tile_effect = self.tile_translator.get_activation_effect(coord)
            if tile_type == 0: # Ordinary
                continue
            else:
                effects_masks.append(tile_effect)

    # HAppens after all effects are done, you just put the special in place.
    def create_special(self, match_coords, match_type: str) -> None: 
        pass


    def _check_same_colour(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
        tile1 = self.board[coord1[0], coord1[1]]
        tile2 = self.board[coord2[0], coord2[1]]
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
        if not (0 <= coord1[0] < self.board_height and 0 <= coord1[1] < self.board_width):
            return False, None
        if not (0 <= coord2[0] < self.board_height and 0 <= coord2[1] < self.board_width):
            return False, None

        # Extract a 6x6 grid around the coords to check for at least 3 match. This covers checking for Ls or Ts.
        y_ranges = max(0, coord1[0] - 2), min(self.board_height, coord2[0] + 3)
        x_ranges = max(0, coord1[1] - 2), min(self.board_width, coord2[1] + 3)
        surround_grid = self.board[y_ranges[0]: y_ranges[1]][x_ranges[0]: x_ranges[1]][:]
        
        # Swap the coordinates to see what happens.
        surround_grid[coord1[0], coord1[1]], surround_grid[coord2[0], coord2[1]] = surround_grid[coord2[0], coord2[1]], self.board[coord1[0], coord1[1]]                
        # Doesn't matter what type of tile it is, if the colours match then its a match
        surround_grid %= self.num_tile_types 
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
        self.board[coord1[0], coord1[1]], self.board[coord2[0], coord2[1]] = self.board[coord2[0], coord2[1]], self.board[coord1[0], coord1[1]]
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

    board.print_board()

