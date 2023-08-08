import numpy as np

BG_COLORS = [
    "\033[1;40m", # background red
    "\033[1;41m", # background green
    ]

color = lambda id,c: "\033[38;5;{}m{}\033[0m".format(id,c)

def print_boards(board: np.ndarray, expected: np.ndarray, gap=5) -> None:
    """Prints the differences between two boards.

    Args:
        board (np.ndarray): The first board.
        expected (np.ndarray): The second board.
    """

    height = board.shape[0]
    width = board.shape[1]

    print(' ' + '-' * (width * 2 + 1) + ' '*(gap+1) + '-' * (width * 2 + 1))
    for row_num in range(height):
        middle = False
        if row_num == height // 2:
            middle = True
        print('| ', end='')
        for tile in board[row_num]:
            print(color(tile+1,tile)+"\033[0m", end=' ')
        print('|', end = '')

        if middle:
            spaces = gap - 2
            print(' '*(spaces//2) + '->' + ' '*(spaces//2), end = '')
        else:
            print(" "*(gap//2)*2, end = '')

        print('| ', end='')
        for tile in expected[row_num]:
            print(color(tile+1,tile)+"\033[0m", end=' ')
        print('|')

    print(' ' + '-' * (width * 2 + 1) + ' '*(gap+1) + '-' * (width * 2 + 1))


def highlight_board_diff(board: np.ndarray, expected: np.ndarray, gap=5, prnt=False) -> str:
    """Prints the differences between two boards.

    Args:
        board (np.ndarray): The first board.
        expected (np.ndarray): The second board.
    """
    # 2-5 is color1, 6-9 is color2, 10-13 is color3, 14-17 is color4, 18-21 is color5, 22-25 is color6
    num_colors = 4
    get_color = lambda number, tile: (number - tile - 2) // num_colors + 1 if number != 1 else 20
    print_tile = lambda x, tile_type: "\033[1;3{}m{:2}\033[0m".format(get_color(x, tile_type), x) 

    lines = ["" for i in range(board.shape[0]+2)]
    print("lines = ", lines)

    lines[0] += (" " + "─" * board.shape[1]*3) + " "*2 + (" " + "─" * board.shape[1]*3)
    for row in range(board.shape[0]):
        lines[row+1] += "| "
        
        for i in range(board.shape[1]):
            tile1 = board[row][i]
            tile2 = expected[row][i]
            if tile1 != tile2:
                lines[row+1] += "\033[48;5;1m"+print_tile(tile1, 0)+"\033[0m"
            else:
                lines[row+1] += print_tile(tile1, 0)
            lines[row+1] += " "
        lines[row+1] += "│"
        for tile in expected[row]:
            lines[row+1] += print_tile(tile, 0) + " "
        lines[row+1] += "│"
    lines[-1] += (" " + "─" * board.shape[1]*3) + " "*2 + (" " + "─" * board.shape[1]*3)
    
    if prnt:
        print("\n".join(lines))

    return "\n".join(lines)


if __name__ == "__main__":
    board = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    expected = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    print_boards(board, expected)
    expected[1][1] = 9
    print_boards(board, expected)
    highlight_board_diff(board, expected)

    b3 = np.array([
        [0, 4, 4, 3, 1, 6],
        [2, 2, 3, 1, 5, 5],
        [6, 0, 0, 6, 7, 7],
        [8, 8, 9, 9, 2, 2],
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 4, 6, 6]
        ])
    b4 = b3.copy()
    b4[1][1] = 9
    print_boards(b3, b4)
    highlight_board_diff(b3, b4)
