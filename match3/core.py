import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Union, Tuple, List
from gymnasium.spaces import Box, Discrete
from collections import deque


### Scoring ### 
# For one match, add +



class Board(gym.Env):
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
    def __init__(self, board_height:int, board_width:int, num_tile_types:int, add_specials:bool=False, num_blockers:int=0, seed:Optional[int] = None):

        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.num_tile_types = num_tile_types
        self.num_blockers = num_blockers
        if seed is None:
            seed = np.random.randint(0, 1000000000)
        self.np_random = np.random.default_rng(seed)
        self.flat_size = int(self.board_width * self.board_height)       
         
        self._add_specials = add_specials
        if add_specials:
            high = 2 + int(4*num_tile_types)
        else:
            high = num_tile_types + 1
        
        self.observation_space = Box(low = 1, high = high, shape=(board_height, board_width)) # Normal, v stripe, h stripe, wrapped, and 1 bomb. # TODO: Add blockers for next version of environment. 
        self.action_space = Discrete(int(2 * self.flat_size))

        self.special_translator = {
            "normal": 0,
            "v_stripe": 1,
            "h_stripe": 2,
            "bomb": 3,
            "cookie": 4
        }


    def reset(self, seed:Optional[int] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.board = self.np_random.integers(1, self.num_tile_types + 1, size = self.flat_size).reshape(self.board_height, self.board_width)
        has_matched_candies = True
        while has_matched_candies:
            has_matched_candies = self.eliminate_matched_candies()
            self.cascade_board()
        return self.board.copy(), {}
    
    def _action_to_coords(self, action: int) -> Tuple[Tuple[int, int]]:
        is_vertical = action <= self.flat_size
        coord = np.unravel_index(action, self.board.shape)
        coord2 = coord[0] + is_vertical, coord[1] + (1-is_vertical)
        return coord, coord2
    
    def _check_valid_action(self, coord:Tuple[int], coord2:Tuple[int]) -> bool:
        """This function checks if the action actually does anything. First it checks if both coordinates are on the board. Then it checks if the action achieves some form of matching.

        Args:
            coord (tuple): The first coordinate on grid corresponding to the action taken. This will always be above or to the left of the second coordinate below.
            coord2 (tuple): coordinate on grid corresponding to the action taken.

        Returns:
            bool: True iff action has an effect on the environment.
        """
        ## Check both coords are on the board. ##
        if not (0 <= coord[0] < self.board_height and 0 <= coord[1] < self.board_width):
            return False, None
        if not (0 <= coord2[0] < self.board_height and 0 <= coord2[1] < self.board_width):
            return False, None

        if self._add_specials:
            ## Check if the candies at both coordinates are both special.
            if self.board[coord[0], coord[1]] > self.num_tile_types and self.board[coord2[0], coord2[1]] > self.num_tile_types:
                return True, 2
            
            ## Check if one tile is a multi-coloured bomb.
            if self.board[coord[0], coord[1]] == int(4*self.num_tile_types + 1) or self.board[coord2[0], coord2[1]] == int(4*self.num_tile_types + 1):
                return True, 1
        
        # Extract a 6x6 grid around the coords to check for at least 3 match. This covers checking for Ls or Ts.
        y_ranges = max(0, coord[0] - 2), min(self.board_height, coord2[0] + 3)
        x_ranges = max(0, coord[1] - 2), min(self.board_width, coord2[1] + 3)
        surround_grid = self.board[y_ranges[0]: y_ranges[1]][x_ranges[0]: x_ranges[1]][:]
        
        # Swap the coordinates to see what happens.
        surround_grid[coord[0], coord[1]], surround_grid[coord2[0], coord2[1]] = surround_grid[coord2[0], coord2[1]], self.board[coord[0], coord[1]]        
        # Set to zero for multi-coloured bombs. This does nothing when not using specials. 
        surround_grid[surround_grid == int(4*self.num_tile_types + 1)] = -1 
        
        # Doesn't matter what type of tile it is, if the colours match then its a match
        surround_grid %= self.num_tile_types 
        for sg in [surround_grid, surround_grid.T]:
            for j in range(sg.shape[0]):
                for i in range(2, sg.shape[1]):
                    # If the current and previous 2 are matched and that they are not cookies.
                    if sg[j, i-2] == sg[j, i-1] == sg[j, i] and sg[j, i]> 0:
                        return True, 0
        return False, 0
    
    def _get_tile_type(self, coord: Tuple[int, int]) -> str:        
        return self.special_translator[self.board[coord[0], coord[1]] // self.num_tile_types]
    
    def _check_same_colour(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
        if self.special_translator[self.board[coord1[0], coord1[1]]] == "cookie":
            return False
        else:
            tile_colour1 = self.board[coord1[0], coord1[1]] % self.num_tile_types
            tile_colour2 = self.board[coord2[0], coord2[1]] % self.num_tile_types
            return tile_colour1 == tile_colour2

    def _get_activation_area(self, affected_coords_list: List[Tuple[int, int]], num_special=0) -> List[Tuple[int, int]]:
        """
        Given a list of root coordinates, this function finds the island of candies of the same colour.
        The function works using a depth first search using the neighbours of a coordinate. 

        Args:
            affected_coords_list (List[Tuple[int, int]]): List of coordinates to start searching from.
            num_special (int, optional): If there are knwo. Defaults to 0.

        Returns:
            List[Tuple[int, int]]: _description_
        """
        # Determine if the neighbours of each cell are part of the affected group i.e. if there is a match of 3 starting from the bottom up.
        if num_special == 0:
            cardinal_dirs = [(0,1), (1, 0), (-1, 0), (0, -1)]   
            visited = set()
            nbh_visited = set()
            while len(visited) != len(affected_coords_list):
                for coord in affected_coords_list > 0:
                    for cardinal_dir in cardinal_dirs:
                        nbh_coord = coord[0] + cardinal_dir[0], coord[1] + cardinal_dir[1]
                        if nbh_coord in visited: 
                            continue
                        else:
                            nbh_visited.add(nbh_coord)
                            if self._check_same_colour(coord, nbh_coord):
                                affected_coords_list.append(nbh_coord)
                    visited.add(coord)
            return visited.union(nbh_visited)
    

        # Cookie involved TODO: Make this work for specials.
        if num_special == 1:
            [coord, coord2] = affected_coords_list
            tile_type = self._get_tile_type(coord)
            if tile_type == self.special_translator["cookie"]:
                return np.ravel_multi_index(np.flatnonzero(self.board == self.board[coord2[0], coord2[1]]), self.board.shape)
            elif tile_type == self.special_translator["v_stripe"]:
                pass
            elif tile_type == self.special_translator["h_stripe"]:
                pass
            elif tile_type == self.special_translator["bomb"]:
                pass
        # Two cookies, cookie and stripe, v stripe and vstripe, stripe and bomb, bomb and bomb.
        raise NotImplementedError()
            

    def _filter_target_area(self, target_area: List[List[Tuple[int, int]]]):
        """Given a board and target area, this function determines what cells actually are activated.It determines what cells are blown up first.
        Args:
            next_affected (List[List[Tuple[int, int]]]): _description_
        """
        # Find longest lines in any directions. If you have one outright winner, then choose that. 

        # Activate 1 special.
        if len(target_area) == 1:
            return target_area

        v_sort = sorted(target_area, key = lambda x : x[0])
        h_sort = sorted(target_area, key = lambda x : x[1])
        
        min_x, max_x = h_sort[0][1], h_sort[-1][1]
        min_y, max_y = v_sort[0][0], h_sort[-1][0]

        surround_grid = self.board[min_y: min(max_y + 1, self.board_height), min_x: min(max_x + 1, self.board_width)]
        
        max_line_len = 0
        max_line_candidates = set() # Trakk all max lines to track for Ts and Ls.
        for sg in [surround_grid, surround_grid.T]:
            for i in range(sg.shape[0]):
                start_col = 0
                end_col = 0
                curr_line_len = 0
                while end_col < sg.shape[1]:
                    if self._get_tile_type(sg[end_col]) == self._get_tile_type(sg[end_col]):
                        curr_line_len += 1
                    else:
                        if max_line_len < curr_line_len:
                            max_line_start = (i, start_col)
                            max_line_end = (i, end_col-1)
                        elif max_line_len == curr_line_len == 3: # 4s and 5s take priority.
                            max_line_candidates.add((max_line_start, max_line_end))
                        else:
                            start_col = end_col
                    end_col += 1

        
        if max_line_len > 3:
            # Horizontal line
            if max_line_start[0] == max_line_end[0]: 
                return [(max_line_start[0], i + max_line_start[1]) for i in range(max_line_end[1] - max_line_start[1])]
            # Vertical Line
            else:
                return [(max_line_start[0] + i, max_line_start[1]) for i in range(max_line_end[0] - max_line_start[0])]
        else: # Has to be 3.
            three_coords = set()
            for start, end in max_line_candidates:
                for y in range(end[1] - start[1]):
                    for x in range(end[0] - start[0]):
                        three_coords.add((x, y))
            return three_coords

    def _activation_pass(self, filtered_activation_coords: List[Tuple[int, int]], skip_coords:List[Tuple[int, int]] = []):
        """ 
        Cases:
            3 in a row. 
            4 in a row
            5 in a row
            T 
            L 
            L with specials
            T with specials
            3 in a row with specials
            4 in a ro with specials
            5 in a row with specials.
            1 special.
        """
        # If there are 5 or more, get the centre one, turn it into a cookie, activate the rest. the rest.
        # If it's 4, get the centre one, turn it into a stripe delete the rest.
        # If 
        for coord in filtered_activation_coords:
            if coord not in skip_coords:
                self._activate_cell(coord)

    def _get_tile_value(self, tile_type:str, tile_colour: int):
        return self.special_translator[tile_type] * self.num_tile_types + tile_colour

    def _activate_cell(self, coord:Tuple[int, int]):
        # Only call this when the cell is hit by another thing. Not for specials.
        # Bunch of if statements for which the board changes.
        tile_type = self._get_tile_type(coord)
        self.board[coord[0], coord[1]] = 0
        if tile_type == "horizontal_stripe":
            map(self._activate_cell, [(coord[0], i) for i in range(self.board_width)])
        elif tile_type == "vertical_stripe":
            map(self._activate_cell, [(i, coord[1]) for i in range(self.board_height)])
        elif tile_type == "bomb":
            coords = [(coord[0] + i, coord[1] + j) for i in range(-1, 2) for j in range(-1, 2)]
            map(self._activate_cell, coords)
        elif tile_type == "cookie":
            raise NotImplementedError()
            # Choose a random cell next to it and 

    def _creation_pass(self, activation_coords_list: List[Tuple[int, int]]):
        """This function takes a pass over the activation coordinates and checks if there are 4s, 5s or Ls or Ts.
          If so, inserts special. 

        Args:
            activation_coords_list (List[List[Tuple[int, int]]]): List containing lists of coordinates to consider for activation next.
        """
        # Loop through each list of coordinates. Get the length of the list. If it's 5 then cookie, elif f
        # Could be more than one 
        skip_coords = set()
        if len(activation_coords_list) == 1:
            return skip_coords
        else:
            # Horizontal line given
            is_horizontal = activation_coords_list[0][0] == activation_coords_list[1][0]
            coord_list = sorted(activation_coords_list, key = lambda x: x[is_horizontal])
            if len(coord_list) >= 5:
                special_coord = coord_list[0] + int(is_horizontal*2), coord_list[1] + int((1-is_horizontal)*3)
                self.board[special_coord[0], special_coord[1]] = self._get_tile_value("cookie", 0)
                skip_coords.add(special_coord)
            elif len(coord_list) == 4:
                # horizontal stripe if the line is horizontal. 
                stripe_type = "horizontal_stripe" if is_horizontal else "vertical_stripe"
                special_coord = coord_list[0] + int(is_horizontal*2), coord_list[1] + int((1-is_horizontal)*3)
                self.board[coord_list[0] + int(is_horizontal), coord_list[1] + int((1-is_horizontal))] = self._get_tile_value(stripe_type, 0) # TODO: Adaptively set coordinate of where to put striped tile depending on where the move takes place.
            # Matched 3
            # TODO: Add functionality for bombs.
                # Put a 
        return skip_coords

    def _execute_pass(self, filtered_activation_coords):
        skip_coords = self._creation_pass(filtered_activation_coords) # Creates 
        # Remove all match 3s, carry out effects of specials.
        self._activation_pass(filtered_activation_coords, skip_coords)


    def step(self, action:int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Actions correspond to 2 for each tile position. One for a down swipe and another for a right swipe. 
        # 1) Find the coordinate and the up or right coord. Check for invalid actions.
        coord, coord2 = self._action_to_coords(action)
        is_valid, num_special = self._check_valid_action(coord, coord2)
        
        if not is_valid:
            return self.board.copy(), self.default_reward, False, False, self._get_info()
        else:
            # These are the cells that need to be activated together. What happens to them is determined in _activate_cells.
            next_affected_cells = self._get_next_activated_cells([coord, coord2], num_special=num_special > 0)  
            while len(next_affected_cells) > 0:
                self._activate_cells(next_affected_cells) # Figure out what needs to happen to the region depending on its shape. 
                self._cascade()
                next_affected_cells = self._get_next_activated_cells()

        # Invalid action
            if not has_matched_candies:
                self.board[coord[0], coord[1]], self.board[coord2[0], coord2[1]] = self.board[coord2[0], coord2[1]], self.board[coord[0], coord[1]]
            else:            
                while has_matched_candies:
                    self.cascade_board()
                    has_matched_candies = self.eliminate_matched_candies()
                # TODO: Compute score from eliminated candies.
        return self.board.copy()
    
    
    def cascade_board(self) -> None:
        self.push_zeros_up()
        self.fill_missing()

    def eliminate_matched_candies(self) -> bool:
        # go through the rows to check for lines of 3 or more
        # go through the colums to check for lines of 3 or more
        # remove any lines found and repeat until clean sweeps
        mask = np.ones(self.board.shape, dtype=int)
        hit = False
        # Check all horizontal matches by finding 
        for row in range(self.board_height):
            for i in range(2, self.board_width):
                # If the current and previous 2 are matched
                if self.board[row, i-2] == self.board[row, i-1] == self.board[row, i]:
                    hit = True
                    start = i
                    # Iterate through to find the full number of matched candies. 
                    while self.board[row, i] == self.board[row, i-1]:
                        i+=1
                        if i >= self.board_width:
                            break
                    # Set all to zero.
                    mask[row, start-2:i] = 0
        for col in range(self.board_width):
            for i in range(2, self.board_height):
                if self.board[i-2, col] == self.board[i-1, col] == self.board[i, col]:
                    hit = True
                    start = i
                    while self.board[i, col] == self.board[i-1, col]:
                        i+=1
                        if i >= self.board_height:
                            break
                    mask[start-2:i, col] = 0
        
        self.board *= mask
        return hit

    def push_zeros_up(self):
        """Push empty slots to the top."""
        for col in self.board.T:
            zero_count = 0
            for i in range(len(col)-1, -1, -1):
                if col[i] == 0:
                    zero_count += 1
                elif zero_count != 0:
                    col[i + zero_count] = col[i]
                    col[i] = 0

    def fill_missing(self):
        """
        Search top to bottom in each column and break if you hit something that isn't zero.
        Since the board should
        """
        for col in self.board.T:
            for i in range(len(col)):
                if col[i] == 0:
                    col[i] = self.np_random.integers(1, self.num_tile_types + 1, size=1)
                else:
                    break

    def close(self):
        super().close()

    def render(self):
        pass

##############


# Check valid move
# If specials - activate specials. 
# If 5s make a cookie at the coordinate of the move or at the centre of the line, activate the rest of the cells.
# If 4s make a stripe at the coordinate of the move or at the centre of the line, 
# If L or T make a bomb at the coordinate of the move or at the centre of the line.
# If 3 disappear.

# def alternating_board(w,h):
#     board = np.zeros((w,h), np.int32)
#     board[::2,::2] = 1
#     board[1::2,1::2] = 1
#     return board + 1

# def test_horiz():
#     board = Board(8,8)
#     x = alternating_board(8,8)
#     x[2][1:4] = 3
#     board.set_board(x)
#     board.check_board_brute()
#     print("board = \n", board.board)
#     x[2][1:4] = 0
#     assert np.array_equal(x, board.board)

# def test_vert():
#     board = Board(8,8)
#     x = alternating_board(8,8)
#     x[:,3][2:5] = 3
#     print("x = \n", x)
#     board.set_board(x)
#     board.check_board_brute()
#     print("board = \n", board.board)
#     x[:,3][2:5] = 0
#     assert np.array_equal(x, board.board)

# def test_cross():
#     board = Board(8,8)
#     x = alternating_board(8,8)
#     x[:,2][1:4] = 3
#     x[2][1:4] = 3
#     print("x = \n", x)
#     board.set_board(x)
#     board.check_board_brute()
#     print("board = \n", board.board)
#     x[:,2][1:4] = 0
#     x[2][1:4] = 0
#     assert np.array_equal(x, board.board)

# if __name__ == '__main__':
#     board = Board(8,8)

#     print(board.board)

#     board.check_board_brute()
#     print(board.board)

#     test_horiz()
#     test_vert()
#     test_cross()
    
#     print("#"*80)
    
#     board = Board(20,20)
#     y = board.board.copy()

#     s = time.time()

#     board.check_board_brute()
#     print("time taken:", time.time() - s)
#     print("before:\n",y)
#     print("after:\n",board.board)
#     board.cascade()
#     print("after cascade:\n",board.board)
#     board.fill_missing()
#     print("after filling zeros:\n",board.board)

#     print("#"*80)

#     board = Board(200,200)
#     y = board.board.copy()

#     s = time.time()

#     board.check_board_brute()
#     print("time taken:", time.time() - s)
#     print("before:\n",y)
#     print("after:\n",board.board)



if __name__ == "__main__":
    env = tileCrush(5, 4, 3, 0, 1)
    print(env.reset())
    print(env.step(13))