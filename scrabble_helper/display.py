from scrabble_helper.engine import get_changed_locations


class Colours:
    # https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


"""
print(f"{Colours.HEADER}qwertyuiop{Colours.ENDC}")
print(f"{Colours.OKBLUE}qwertyuiop{Colours.ENDC}")
print(f"{Colours.OKCYAN}qwertyuiop{Colours.ENDC}")
print(f"{Colours.OKGREEN}qwertyuiop{Colours.ENDC}")
print(f"{Colours.WARNING}qwertyuiop{Colours.ENDC}")
print(f"{Colours.FAIL}qwertyuiop{Colours.ENDC}")
print(f"{Colours.ENDC}qwertyuiop{Colours.ENDC}")
print(f"{Colours.BOLD}qwertyuiop{Colours.ENDC}")
print(f"{Colours.UNDERLINE}qwertyuiop{Colours.ENDC}")
print("regular")
"""


def pp(board):
    num_cols = len(board[0])
    horiz_bars = "  _ " * num_cols

    print(horiz_bars)

    for row in board:
        print(f"| {' | '.join(row)} |")
        print(horiz_bars)


def pp2(board, new_board):
    changed_locations = set(get_changed_locations(board, new_board))

    max_num_digits = 2  # assume we don't have a board size over 99

    num_cols = len(board[0])

    col_nums = list(range(1, num_cols + 1))
    col_nums_1 = [x // 10 for x in col_nums]  # tens place
    col_nums_2 = [x % 10 for x in col_nums]  # ones_place
    col_nums_str_1 = " " * (max_num_digits + 1) + "".join(
        "  " if x == 0 else f" {x}" for x in col_nums_1
    )
    col_nums_str_2 = " " * (max_num_digits + 1) + "".join(f" {x}" for x in col_nums_2)

    print("")
    print(col_nums_str_1)
    print(col_nums_str_2)
    horiz_bars = " " * (max_num_digits + 1) + "_" * num_cols * 2

    print(horiz_bars)

    for r, row in enumerate(new_board):
        print(f"{str(r+1).ljust(max_num_digits)}|", end="")
        for c, tile in enumerate(row):
            if (r, c) in changed_locations:
                text = f"{Colours.OKGREEN}{tile}{Colours.ENDC}"
            else:
                text = tile
            print(f" {text}", end="")
        print(f"|{str(r+1).ljust(max_num_digits)}")

    print(horiz_bars)
    print(col_nums_str_1)
    print(col_nums_str_2)
    print("")
