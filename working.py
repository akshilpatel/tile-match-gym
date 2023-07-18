import numpy as np
from typing import List, Tuple
import json


class BoardMatcher:
    def __init__(self, board):
        self.board = np.array(board)
        self.rows = len(board)
        self.cols = len(board[0])


    def get_lines(self):
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

    def get_matches(self, islands, lines):
        """
        Detects the type of match type from the bottom up
        repeats for the remaining island

        returns the match coordinates and the match type for each match in the
        island removed from bottom to top
        """

        tile_names = []
        tile_coords = []
        
        lines = sorted([sorted(i, key=lambda x: (x[0],x[1])) for i in lines], key=lambda y: (y[0][0]), reverse=True)

        while len(lines) > 0:
            line = lines.pop()
            print("Line = ", line)
            # check for cookie
            if len(line) >= 5:
                tile_names.append("cookie")
                tile_coords.append(line)
            # check for laser
            elif len(line) == 4:
                tile_names.append("laser")
                tile_coords.append(line)
            # check for bomb
            elif any([c in l for c in line for l in lines]): # REMOVE THIS AS SLOW AND IS DONE TWICE
                print("BOMB FOUND")
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
            # check for match
            # check for no match
            else:
                tile_names.append("ERR")
                tile_coords.append(line)

        # # check_coord = lambda c, t: self.board[c[0]][c[1]] == t if (c[0] >= 0 and c[0] < self.rows and c[1] >= 0 and c[1] < self.cols) else False
        # check_coord = lambda c, t: self.board[c[0]][c[1]] == t if (0 <= c[0] < self.rows and 0 <= c[1] < self.cols) else False

        # # islands = sorted(self.get_islands(lines), key=lambda x: (x[0],x[1]), reverse=True)
        # islands = sorted([sorted(i, key=lambda x: (x[0],x[1])) for i in self.get_islands(lines)], key=lambda x: (x[0][0]))

        # tile_names = []
        # tile_coords = []
        # 
        # # while there are still valid islands that have not been checked
        # while len(islands) > 0:
        #     # check the bottom island
        #     for i in range(len(islands)):
        #         # go through the neighbours of the current island
        #         # check for matches and remove them from the island
        #         # if there are no matches, move to the next island
        #         current = islands[i][0]
        #         print("current = ", current)
        #         # check horizental cookie
        #         # check vertical cookie
        #         # check horizontal laser
        #         # check vertical laser
        #         # check horizontal bomb
        #         # check vertical bomb
        #         # check horizontal norm
        #         # check vertical norm
        #     break

        # 
        # # iterate through each island
        # # for each island, check for matches and remove them from the island
        # 
        # #for island in islands:
        #     # determine the match type associated with the lowest line

        print("tile_names = ", tile_names)
        return tile_coords, tile_names
    
    def contains(self, line, coord):
        for c in line:
            if c == coord:
                return True
        return False
    
    @staticmethod
    def get_islands(lines):
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


################################################################################

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
    # utils
    sort_coords = lambda l:sorted([sorted(i, key=lambda x: (x[0], x[1])) for i in l])
    coords_match = lambda l1, l2: sort_coords(l1) == sort_coords(l2)
    format_test = lambda r, e: "result: \t"+str(r)+"\nexpected: \t"+str(e)+"\n"

    matcher = BoardMatching()
    
    boards = json.load(open("boards.json", "r"))["boards"]
    
    for board in boards:
        print("testing board: ", board['name'])
        bm = BoardMatcher(board['board'])
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
        
        
        print("PASSED")
        # print("test:", board['name'])
        # bd = np.array(board['board'])
        # lns = board['matches']
        # match_coords = matcher.get_lines(bd)
        # print("matched coords = ", match_coords)
        # print("expected coords = ", board['matches'])
        # r = [[list(coord) for coord in sublist] for sublist in match_coords]
        # # ensure the correct number of matches are found
        # assert len(match_coords) == len(board['matches']), "incorrect number of matches found"
        # # ensure the coordinates are the same
        # for l in range(len(r)):
        #     for coord in r[l]:
        #         assert coord in board['matches'][l], f"coord {coord} not in matches {board['matches'][l]}"

        # # merge matches
        # merged_matches = merge_matches(match_coords)
        # print("merged_matches = ", merged_matches)

        # """
        # Should first check if the identified line at the bottom is part of:
        #     - a cookie
        #     - a bomb
        #     - a laser
        #     - an ordinary match
        # """

        print("----")


