import numpy as np
from typing import List, Tuple
import json

class BoardMatching:

    def get_lines(self, board) -> List[List[Tuple[int, int]]]:
        """
        Get the coordinates of all the lines on the board that have at least 3
        matching candies.

        Args:
            board (np.array): The board to search for lines.

        Returns:
            list of lists of tuples: A list of lines where each line is a list of
            tuple coordinates.
        """
        h_lines = self.get_h_lines(board)
        v_lines = self.get_v_lines(board)
        return h_lines + v_lines

    def get_h_lines(self, board) -> List[List[Tuple[int, int]]]:
        """
        Get the coordinates of all the horizontal lines on the board that have at
        least 3 matching candies.
        """
        h_lines = []
        # Check all horizontal lines starting from the bottom
        for row in range(board.shape[0] - 1, -1, -1):
            col = 2
            while col < board.shape[1]:
                # If the current and previous 2 are matched
                if (board[row, col - 2] == board[row, col - 1] == board[row, col]):
                    start = (row, col - 2)
                    # Iterate through to find the full number of matched candies.
                    while (col < board.shape[1] and board[row, col] == board[row, col - 1]):
                        col += 1
                    h_lines.append([(start[0], i) for i in range(start[1], col)])
                    col += 2
                else:
                    col += 1
        return h_lines

    def get_v_lines(self, board) -> List[List[Tuple[int, int]]]:
        """
        Get the coordinates of all the vertical lines on the board that have at 
        least 3 matching candies.
        """
        v_lines = []
        # Bottom left to top right
        row = board.shape[0] - 3
        while row >= 0:
            for col in range(board.shape[1]):
                if (board[row, col] == board[row + 1, col] == board[row + 2, col]):
                    match = [(row + 2, col), (row + 1, col), (row, col)]
                    m_search_row = row
                    while m_search_row > 0 and board[m_search_row, col] == board[m_search_row - 1, col]:
                        m_search_row -= 1
                        match.append((m_search_row, col))
                    v_lines.append(match)
            row -= 1
        return v_lines

    def get_match_coords(self, board) -> List[List[Tuple[int, int]]]:
            """For the current board, find the first set of matches. Go from the bottom up and find the set of matches.
            Returns:
                List[List[Tuple[int, int]]]: List of matches. Each match is specified by a list.

            # Look through the matches from bottom up and stop when you've checked the lowest row that has a match.
            # Do the same thing for vertical.
            # This currently only works for lines in one axis, i.e. we cannot detect Ls or Ts
            """
            h_matches, lowest_row_h = self.get_lowest_h_match_coords(board)
            v_matches, lowest_row_v = self.get_lowest_v_match_coords(board)
            if lowest_row_h == lowest_row_v == -1:
                return []
            # Check which matches are lowest and only return those.
            if lowest_row_h == lowest_row_v:            
                return h_matches + v_matches
            elif lowest_row_h > lowest_row_v:
                return h_matches
            else:
                return v_matches
        
    # Could use a mask to fix by setting those that have been added to a match to mask.
    def get_lowest_v_match_coords(self, board) -> List[List[Tuple[int, int]]]:
        """
        Find the lowest vertical matches on the board starting from the bottom up.

        Returns:
            List[List[Tuple[int, int]]]: List of coordinates defining the vertical matches.
        """
        v_matches = []
        lowest_row_v = -1
        # Bottom left to top right
        row = board.shape[0] - 3
        while row >= 0:
            if lowest_row_v != -1:
                break
            for col in range(board.shape[1]):
                if (board[row, col] == board[row + 1, col] == board[row + 2, col]
                ):  # Found a match
                    lowest_row_v = max(row + 2, lowest_row_v)
                    match = [(row + 2, col), (row + 1, col), (row, col)]
                    m_search_row = row
                    while m_search_row > 0 and board[m_search_row, col] == board[m_search_row - 1, col]:
                        m_search_row -= 1
                        match.append((m_search_row, col))
                    v_matches.append(match)
            row -= 1
        return v_matches, lowest_row_v
    
    def get_lowest_h_match_coords(self, board) -> List[List[Tuple[int, int]]]:
        h_matches = []
        lowest_row_h = -1
        # Check all horizontal matches starting from the bottom
        for row in range(board.shape[0] - 1, -1, -1):
            if lowest_row_h != -1:  # Don't need to check rows higher up.
                break
            col = 2
            while col < board.shape[1]:
                # If the current and previous 2 are matched
                if (board[row, col - 2] == board[row, col - 1] == board[row, col]
                ):
                    lowest_row_h = max(row, lowest_row_h)
                    start = (row, col - 2)
                    # Iterate through to find the full number of matched candies.
                    while (col < board.shape[1] and board[row, col] == board[row, col - 1]):
                        col += 1
                    match = [(start[0], i) for i in range(start[1], col)]
                    h_matches.append(match)
                    col += 2
                else:
                    col += 1
        return h_matches, lowest_row_h

def merge_matches(matches: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    Merge matches that share coordinates and return the merged result.

    Args:
        matches (list): A list of matches where each match is a list of tuple coordinates.

    Returns:
        list: A list of lines where each line is a list of tuple coordinates. 
              Each line has an overlap with at least one other line.

    """
    merged_matches = []  # List to store the merged matches
    merged_indices = set()  # Set to track the merged indices
    for i in range(len(matches)):
        if i in merged_indices:
            continue  # Skip this match if it has already been merged

        match = matches[i]
        merged_match = [match]
        intersection_coords = set()

        for j in range(i + 1, len(matches)):
            if j in merged_indices:
                continue  # Skip this match if it has already been merged
            other_match = matches[j]
            
            has_intersect = False
            for coord in match:
                for coord2 in other_match:
                    if coord == coord2:
                        has_intersect = True
                        break
            if has_intersect:
                # Merge the matches if they share any coordinates
                merged_match.append(other_match)
                merged_indices.add(j)
                intersection_coords.add(coord)
        
        merged_matches.append((merged_match, intersection_coords))
        
    return merged_matches # List of islands and intersection 




def clear_intersected_coord(island, line_idx, intersection_coords):
    # Remove intersected coords from other lines, remove current line and remove any invalid lines.
    line = island[line_idx]
    for coord in intersection_coords:
        if coord in line:
            for other_line in island:
                if coord in other_line:
                    if len(other_line) <=3:
                        del other_line
                    else:
                        other_line.remove(coord)
    for line in island:
        if not is_valid_line(line):
            island.remove(line)

def is_valid_line(line):
    if len(line) < 3:
        return False
    is_horizontal = line[0][0] == line[1][0]
    sorted_line = sorted(line, key=lambda x: x[is_horizontal])
    end = 1
    while end < len(sorted_line):
        if sorted_line[end][is_horizontal] != sorted_line[end - 1][is_horizontal] + 1:
            return False
    return True
        
def detect_match_types(island: List[Tuple[int, int]], intersection_coords: List[Tuple[int, int]]):
    """1. Detect cookies
       2. Detect lasers
       3. Detect bombs
       4. Detect ordinar

    Args:
        island (List[Tuple[int, int]]): _description_
        intersection_coords (List[Tuple[int, int]]): _description_

    Returns:
        _type_: _description_
    """
    island = sorted(island, key = lambda x: len(x), reverse=True)
    matches = []

    cookie_matches, island = detect_cookie_matches(island, intersection_coords, matches)
    laser_matches, island = detect_laser_matches(island, intersection_coords, matches)
    bomb_matches, island = detect_bomb_matches(island, intersection_coords, matches)
    ordinary_matches, island = detect_ordinary_matches(island, intersection_coords, matches)

    return cookie_matches + laser_matches + bomb_matches + ordinary_matches   
    
    
def detect_cookie_matches(island, intersection_coords, matches):
    for line_idx, line in enumerate(island):
        if len(line) >= 5:
            matches.append({
                "coords": line,
                "type": "cookie"})
            # Remove intersected coords from other lines, remove current line and remove any invalid lines.
            clear_intersected_coord(island, line_idx, intersection_coords)    
    return matches, island

def detect_laser_matches(island, intersection_coords, matches):
    for line_idx, line in enumerate(island):
        if len(line) == 4:
            if line[0][0] == line[1][0]:
                match_type = "horizontal"
            else:
                match_type = "vertical"
            matches.append({
                "coords": line,
                "type": match_type})
            
            clear_intersected_coord(island, line_idx, intersection_coords)
    return matches, island

def detect_ordinary_matches(island, intersection_coords, matches):
    # Check each line in island.
    for line_idx, line in enumerate(island):
        if len(line) == 3:
            matches.append({
                "coords": line,
                "type": "horizontal" if line[0][0] == line[1][0] else "vertical"
                })
            clear_intersected_coord(island, line_idx, intersection_coords)

    return matches, island
    
def detect_bomb_matches(island, intersection_coords, matches, coord_to_lines: dict[Tuple[int,int]: List[List[Tuple[int, int]]]]):
    for coord in intersection_coords:
        matches.append({
                "coords": list(set(*coord_to_lines[coord])),
                "type": "bomb"
                })
        for line in coord_to_lines[coord]:    
            clear_intersected_coord(island, line_idx, intersection_coords)
        
    return matches, island


# TODO: Change get_match_coords to check same colour instead of same number.



if __name__ == "__main__":
    matcher = BoardMatching()
    
    boards = json.load(open("boards.json", "r"))["boards"]

    t = [[str(x) for x in line] for line in boards[0]['board']]
    t[2][0] = "X"
    [print(line) for line in t]
    
    for board in boards:
        bd = np.array(board['board'])
        lns = board['matches']
        match_coords = matcher.get_lines(bd)
        print("matched coords = ", match_coords)
        print("expected coords = ", board['matches'])
        # ensure the correct number of matches are found
        # ensure the coordinates are the same
        r = [[list(coord) for coord in sublist] for sublist in match_coords]
        print("bd = ", bd)
        assert len(match_coords) == len(board['matches'])
        for l in range(len(r)):
            for coord in r[l]:
                assert coord in board['matches'][l]
        print("----")
