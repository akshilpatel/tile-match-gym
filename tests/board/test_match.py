import pytest
from tile_match_gym.board import Board
from tests.utils import create_alternating_array, contains_threes
import json

################################################################################
################## This is just templating - needs to be changed ###############
################################################################################

# def inject_json(json_file):
#     """
#     read a json file to pass to the test function
#     """
#     def decorator(func):
#         def wrapper():
#             boards = json.load(open(json_file, "r"))["boards"]
#             for board in boards:
#                 func(board)
#         return wrapper
#     return decorator
# 
# def test_match():
#     # utils
#     sort_coords = lambda l:sorted([sorted(i, key=lambda x: (x[0], x[1])) for i in l])
#     coords_match = lambda l1, l2: sort_coords(l1) == sort_coords(l2)
#     format_test = lambda r, e: "result: \t"+str(r)+"\nexpected: \t"+str(e)+"\n"
# 
#     boards = json.load(open("board_test_data.json", "r"))["boards"]
#     
#     for board in boards:
#         print("testing board: ", board['name'])
#         bm = Board(0, 0, 0, board['board'])
#         matches = bm.get_lines()
#         expected_matches = [[tuple(coord) for coord in line] for line in board['matches']]
#         expected_islands = [[tuple(coord) for coord in line] for line in board['islands']]
#         expected_tile_coords = [[tuple(coord) for coord in line] for line in board['tile_locations']]
#         expected_tile_names = board['tile_names']
# 
#         assert len(matches) == len(board['matches']), "incorrect number of matches found\n"+format_test(matches, expected_matches)
#         assert coords_match(matches, expected_matches), "incorrect matches found\n"+format_test(matches, expected_matches)
#         
#         #islands = bm.get_islands(matches)
#         #assert coords_match(islands, expected_islands), "incorrect islands found\n"+format_test(sort_coords(islands), sort_coords(expected_islands))
#     
#         tile_coords, tile_names = bm.get_matches([], matches)
#         assert coords_match(tile_coords, expected_tile_coords), "incorrect tile coords found\n"+format_test(sort_coords(tile_coords), sort_coords(expected_tile_coords))
#             
#         # make sure that the tiles collected are correct and in the same order
#         print(tile_names, expected_tile_names)
#         assert all(
#             [
#                 name == expected_name
#                 for name, expected_name in zip(tile_names, expected_tile_names)
#             ]
#         ), "incorrect tile names found\n" + format_test(tile_names, expected_tile_names)
#         
#         print("tile_coords = ", tile_coords)
#         print("tile_names = ", tile_names)
#         print("PASSED")
#         print("get_tiles = ", bm.get_tiles())
# 
#         print("----")
# 
# 
