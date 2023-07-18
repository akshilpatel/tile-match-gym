import numpy as np
from typing import List, Tuple
import json


class BoardMatcher:
    """
    A class for matching specific tiles in a board
    """

    def __init__(self, board: List[List[int]]):
        self.board = np.array(board)
        self.rows = len(board)
        self.cols = len(board[0])
    
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

        returns the match coordinates and the match type for each match in the
        island removed from bottom to top

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
    # utils
    sort_coords = lambda l:sorted([sorted(i, key=lambda x: (x[0], x[1])) for i in l])
    coords_match = lambda l1, l2: sort_coords(l1) == sort_coords(l2)
    format_test = lambda r, e: "result: \t"+str(r)+"\nexpected: \t"+str(e)+"\n"

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
        
        print("tile_coords = ", tile_coords)
        print("tile_names = ", tile_names)
        print("PASSED")
        print("get_tiles = ", bm.get_tiles())

        print("----")


