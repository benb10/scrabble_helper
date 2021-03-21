"""

We use the python lib english_words https://pypi.org/project/english-words/.

from english_words import english_words_lower_alpha_set as words
At the time of writing, it has around 25,000 words
It looks like he lib does NOT have plurals.

Plan:
for each row/column:
filter words by max length (tiles + existing)
filter by containing existing letter groups
filter by issubset of tiles + existing
generate possible new rows/columns.
from this, generate new possible boards
filter to valid boards (checking words indirectly created in other rows/columns)
"""
from english_words import english_words_lower_alpha_set as words
from itertools import chain
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Set, Optional, Any
from math import ceil, floor
from random import shuffle
import json
import os


def gen_rows(board):
    yield from board


def gen_cols(board):
    for col in zip(*board):
        yield list(col)


assert list(gen_cols([[1, 2], [3, 4]])) == [[1, 3], [2, 4]]


def is_tile_subset(word, tiles):
    if not set(word).issubset(tiles):
        return False

    # this is more involved than a subset check, because if there are duplicate tiles,
    # we need to make sure we have enough

    tiles = deepcopy(tiles)  # don't mutate arg

    for char in word:
        if char not in tiles:
            return False
        tiles.remove(char)
    return True


assert is_tile_subset("abcd", ["a", "b", "c"]) is False
assert is_tile_subset("abcd", ["a", "b", "c", "d"]) is True
assert is_tile_subset("abcd", ["a", "b", "c", "d", "e", "e"]) is True
assert is_tile_subset("aadd", ["a", "b", "c", "d"]) is False
assert is_tile_subset("aaad", ["a", "b", "c", "d", "a"]) is False
assert is_tile_subset("aaad", ["a", "b", "c", "d", "a", "a"]) is True


def row_is_valid(row, new_row, availiable_tiles):
    pos_to_char_check = {i: char for i, char in enumerate(row) if char != " "}

    for pos, char in pos_to_char_check.items():
        if new_row[pos] != char:
            # print("a")
            return False

    new_char_positions = [i for i, (x, y) in enumerate(zip(row, new_row)) if x != y]

    if not new_char_positions:
        # this new row does not place any tiles, so it is not valid
        # print("b")
        return False

    existing_char_positions = set(pos_to_char_check)
    first_new_char_pos = min(new_char_positions)
    last_new_char_pos = max(new_char_positions)

    must_have_existing_range = set(range(first_new_char_pos - 1, last_new_char_pos + 2))

    if not any(i in must_have_existing_range for i in existing_char_positions):
        # print("c")
        return False  # the new word does not include an existing tile

    new_row_chars = [c for c in new_row if c != " "]
    old_row_chars = [c for c in row if c != " "]
    availiabe_chars = old_row_chars + availiable_tiles
    if not is_tile_subset(new_row_chars, availiabe_chars):
        # eg. there is only one 'w' availiable, and we are using it twice
        return False

    return True


assert row_is_valid([" ", " ", "t"], ["n", "e", "t"], ["n", "e"]) is True
assert (
    row_is_valid([" ", " ", " ", "e", " "], ["f", "e", " ", "e", " "], ["f", "e"])
    is False
)
assert (
    row_is_valid([" ", " ", " ", "e", " "], [" ", " ", "f", "e", "d"], ["f", "d"])
    is True
)
assert (
    row_is_valid(
        [" ", " ", " ", " ", " ", " ", " ", "e", " ", "a", "w", "a", "y", " ", " "],
        [" ", " ", " ", "w", "a", "i", "v", "e", " ", "a", "w", "a", "y", " ", " "],
        ["a", "a", "a", "c", "e", "i", "v"],
    )
    is False
)  # overusing 'w'


def gen_new_rows(row, word_options, tiles):
    new_rows = []

    for word in word_options:
        for start_pos in range(0, len(row) - len(word) + 1):
            new_row = deepcopy(row)

            end_pos = start_pos + len(word)

            new_row[start_pos:end_pos] = list(word)

            pos_to_char = {i: c for i, c in enumerate(new_row)}

            if pos_to_char.get(start_pos - 1, " ") != " ":
                # print("a")
                continue  # bad start

            if pos_to_char.get(end_pos, " ") != " ":
                # print("b")
                continue  # bad end

            if row_is_valid(row, new_row, tiles):
                new_rows.append(new_row)
    return new_rows


x = gen_new_rows([" ", " ", "t"], ["net", "tea", "met"], tiles=["e", "n", "m", "a"])
assert sorted(x) == [["m", "e", "t"], ["n", "e", "t"]]


def get_row_options(row, tiles):

    existing_chars = [c for c in row if c != " "]
    max_length = len(tiles) + len(existing_chars)

    if max_length > len(row):
        max_length = len(row)

    word_options = words
    # print(len(word_options))
    word_options = {w for w in words if len(w) <= max_length}
    # print(len(word_options))
    word_options = {
        w
        for w in word_options
        if any(existing_char in w for existing_char in existing_chars)
    }
    # print(len(word_options))
    all_availiable_chars = tiles + existing_chars
    word_options = {w for w in word_options if is_tile_subset(w, all_availiable_chars)}
    # print(len(word_options))

    new_row_options = gen_new_rows(row, word_options, tiles)

    # print(len(new_row_options))

    return new_row_options


r = [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'd', ' ', ' ', 'd', 'a', 'c', 'c', 'a']

tiles = ['a', 'a', 'b', 'b', 'c', 'c', 'd']
x = get_row_options(r, tiles)
assert [' ', ' ', ' ', ' ', ' ', 'b', 'a', 'd', ' ', ' ', 'd', 'a', 'c', 'c', 'a'] in x

r = [" ", " ", " ", " ", " ", " ", " ", "e", " ", "a", "w", "a", "y", " ", " "]

x = get_row_options(r, tiles=["a", "a", "a", "c", "e", "i", "v"])

# 'w' used in wrong place
bad_row = [" ", " ", " ", "w", "a", "i", "v", "e", " ", "a", "w", "a", "y", " ", " "]
assert bad_row not in x
assert len(x) == 10, len(x)

x = get_row_options(
    [" ", " ", "l", " ", " ", " ", " ", " "], tiles=["a", "i", "o", "p", "r", "t", "y"]
)
bad_row = ["p", "o", "l", "y", "t", "y", "p", "y"]
# demonstrate that this is in words:
assert "polytypy" in words
# bad row uses 'y' multiple times.
# In a previous bug, this was possible.
# make sure it doesn't happen
assert bad_row not in x
assert len(x) == 36


x = get_row_options(row=[" ", " ", " ", "e", " "], tiles=["b", "c", "d", "e"])
assert sorted(x) == [
    [" ", " ", " ", "e", "d"],
    [" ", " ", "b", "e", " "],
    [" ", " ", "b", "e", "d"],
    [" ", " ", "b", "e", "e"],
    [" ", " ", "d", "e", " "],
    [" ", " ", "d", "e", "c"],
    [" ", " ", "d", "e", "e"],
    [" ", "b", "e", "e", " "],
    [" ", "d", "e", "e", " "],
    ["c", "e", "d", "e", " "],
]


x = get_row_options(row=[" ", " ", "t"], tiles=["e", "m", "n"])
assert sorted(x) == [[" ", "e", "t"], [" ", "m", "t"], ["m", "e", "t"], ["n", "e", "t"]]


@dataclass
class BoardOption:
    baord: List[List[str]]
    new_tile_positions: Optional[Any]
    word: str


def gen_char_groups(sequence, separator=" "):
    """
    https://stackoverflow.com/questions/54372218/how-to-split-a-list-into-sublists-based-on-a-separator-similar-to-str-split
    """
    chunk = []
    for val in sequence:
        if val == separator and chunk:
            yield chunk
            chunk = []
            continue
        if val != separator:
            chunk.append(val)

    yield chunk


assert list(gen_char_groups([" ", " ", "a", " ", "b", "c"])) == [["a"], ["b", "c"]]


def gen_words(board):
    lines = chain(gen_rows(board), gen_cols(board))
    all_char_groups = chain.from_iterable(gen_char_groups(line) for line in lines)

    yield from (
        "".join(char_group) for char_group in all_char_groups if len(char_group) > 1
    )


b = [["a", "b", "c"], [" ", " ", "c"]]
assert list(gen_words(b)) == ["abc", "cc"]


def board_is_valid(board, new_board, unrecognised_words=None, allow_2l_words=True):
    current_words = set(gen_words(board))

    current_invalid_words = current_words.difference(words)

    if current_invalid_words:
        # There isn't really anything we can do about this now.  Just log it
        if unrecognised_words is not None:
            unrecognised_words.update(current_invalid_words)

    starting_words = words

    if not allow_2l_words:
        # It is important that we do this filtering before adding current words
        # We can't do anything about a 2 letter word already on the board, so we allow it
        starting_words = {w for w in starting_words if len(w) > 2}

    all_valid_words = starting_words.union(current_words)

    new_words = set(gen_words(new_board))

    new_invalid_words = new_words.difference(all_valid_words)

    if new_invalid_words:
        return False

    return True


# fmt: off
orig = [
    [" ", " ", "n"],
    [" ", " ", "o"],
    [" ", " ", "d"],
]
b1 = [
    ["a", "a", "n"],
    [" ", " ", "o"],
    [" ", " ", "d"],
]

b2 = [
    ["r", "u", "n"],
    [" ", " ", "o"],
    [" ", " ", "d"],
]
assert board_is_valid(orig, b1) is False
assert board_is_valid(orig, b2) is True
# fmt: on


def board_row_options(r, row_options, board, unrecognised_words=None, allow_2l_words=True):
    board_options = []

    for row in row_options:
        # insert into board
        new_board = deepcopy(board)
        new_board[r] = row

        if board_is_valid(board, new_board, unrecognised_words=unrecognised_words, allow_2l_words=allow_2l_words):
            board_options.append(new_board)

    return board_options


b = [[" ", " ", "n"], [" ", " ", "o"], [" ", " ", "d"]]
x = board_row_options(2, row_options=[["o", "d", "d"], ["f", "e", "d"]], board=b)
assert len(x) == 2


b = [[" ", " ", "n"], ["d", " ", "o"], [" ", " ", "d"]]
x = board_row_options(2, row_options=[["o", "d", "d"], ["f", "e", "d"]], board=b)
assert len(x) == 1


def board_col_options(c, col_options, board, unrecognised_words=None, allow_2l_words=True):
    board_options = []

    for col in col_options:
        # insert into board
        new_board = deepcopy(board)
        cols = list(gen_cols(new_board))
        cols[c] = col
        # switch it back to a list of rows:
        new_board = list(gen_cols(cols))

        if board_is_valid(board, new_board, unrecognised_words=unrecognised_words, allow_2l_words=allow_2l_words):
            board_options.append(new_board)

    return board_options


b = [[" ", " ", " "], [" ", " ", " "], ["d", "o", "g"]]
x = board_col_options(0, col_options=[["n", "o", "d"], ["f", "e", "d"]], board=b)
assert len(x) == 2


def start_of_game_words(tiles, max_word_length):
    tiles = set(tiles)

    word_options = {
        word
        for word in words
        if len(word) <= max_word_length and is_tile_subset(word, tiles)
    }
    return word_options


x = start_of_game_words(["a", "b", "c", "d", "e"], 10)
assert len(x) > 20
assert "bead" in x


def start_of_game_options(board, tiles):
    middlest_row_index = floor(len(board) / 2)
    num_cols = len(board[middlest_row_index])
    max_word_length = min(len(tiles), num_cols)

    word_options = start_of_game_words(tiles, max_word_length)

    board_options = []

    for word in word_options:
        new_board = deepcopy(board)
        # just put the word roughly in the middle of the most middle row:
        start_col_index = floor((num_cols - len(word)) / 2)
        end_col_index = start_col_index + len(word)
        new_board[middlest_row_index][start_col_index:end_col_index] = list(word)
        board_options.append(new_board)

    return board_options


def get_options(board, tiles, allow_2l_words=True):
    all_squares = list(chain(*board))
    is_start_of_game = all(square == " " for square in all_squares)

    unrecognised_words = set()

    if is_start_of_game:
        return start_of_game_options(board, tiles)

    all_board_options = []

    print(f"Looking for words along rows...")

    for r, row in enumerate(gen_rows(board)):
        row_options = get_row_options(row, tiles)

        # validate here:
        board_options = board_row_options(r, row_options, board, unrecognised_words=unrecognised_words, allow_2l_words=allow_2l_words)
        all_board_options.extend(board_options)

    print(f"Looking for words along columns...")

    for c, col in enumerate(gen_cols(board)):
        col_options = get_row_options(col, tiles)

        # validate here
        board_options = board_col_options(c, col_options, board, unrecognised_words=unrecognised_words, allow_2l_words=allow_2l_words)
        all_board_options.extend(board_options)

    print(f"Unrecognised words in existing_board: {sorted(unrecognised_words)}")

    return all_board_options


b = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'z', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', 'b', 'l', 'a', 'k', 'e', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', 'l', 'o', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', 'a', 't', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', 'n', 'y', 'c', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', 'k', ' ', 'h', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'o', 'o', 'z', 'e', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'k', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'z', 'e', 'r', 'o', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
 [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]

tiles = ['e', 'h', 'k', 'l', 'o', 't', 'z']
options = get_options(b, tiles, allow_2l_words=False)
# In a previous bug, this would give no options because of the 2 letter words
# already on the board
assert len(options) == 63, len(options)


b = [[" ", " ", " "], [" ", " ", " "], ["c", "a", "t"]]
x = get_options(b, ["a", "e", "h", "l", "n"])
assert len(x) == 16, len(x)


def get_changed_locations(board, new_board):
    """Yield pairs of (row_index, col_index) for tiles that have changed."""
    for row_index, row in enumerate(board):
        for col_index, tile in enumerate(row):
            if tile != new_board[row_index][col_index]:
                yield (row_index, col_index)


def get_score(chars):
    # https://en.wikipedia.org/wiki/Scrabble_letter_distributions
    tile_to_score = {
        "a": 1,
        "b": 3,
        "c": 3,
        "d": 2,
        "e": 1,
        "f": 4,
        "g": 2,
        "h": 4,
        "i": 1,
        "j": 8,
        "k": 5,
        "l": 1,
        "m": 3,
        "n": 1,
        "o": 1,
        "p": 3,
        "q": 10,
        "r": 1,
        "s": 1,
        "t": 1,
        "u": 1,
        "v": 4,
        "w": 4,
        "x": 8,
        "y": 4,
        "z": 10,
    }
    score = sum(tile_to_score[tile] for tile in chars)
    return score


assert get_score("qpyy") == 21


def get_new_word_score(board, new_board):
    changed_locations = get_changed_locations(board, new_board)

    added_tiles = [new_board[r][c] for r, c in changed_locations]

    # This isn't a perfect method
    # We are not counting the score on existing tiles
    # We are not counting any double/triple letter/word bonuses
    score = get_score(added_tiles)
    return score


# fmt: off
b = [
    [" ", " ", "n"],
    [" ", " ", "o"],
    [" ", " ", "d"],
]
nb = [
    ["p", "e", "n"],
    [" ", " ", "o"],
    [" ", " ", "d"],
]
# fmt: on

assert get_new_word_score(b, nb) == 4


def best_options(board, tiles, n, allow_2l_words=True):
    """Return the n best board options."""
    all_board_options = get_options(board, tiles, allow_2l_words=allow_2l_words)

    board_score_pairs = [
        (new_board, get_new_word_score(board, new_board))
        for new_board in all_board_options
    ]

    board_score_pairs.sort(key=lambda t: t[1], reverse=True)

    return board_score_pairs[:n]


b = [[" ", " ", " "], [" ", " ", " "], ["c", "a", "t"]]

x = best_options(b, tiles=["e", "x", "j", "k", "z", "a", "f", "r", "n"], n=1)
assert x[0][0] == [[" ", " ", "j"], [" ", " ", "e"], ["c", "a", "t"]]
assert x[0][1] == 9

b = [[" " for _ in range(7)] for _ in range(7)]
x = best_options(
    b, tiles=["e", "x", "j", "z", "a", "r", "n", "l", "w", "e", "a", "e"], n=5
)
assert len(x) == 5
assert x[0][0][3] == [" ", "w", "a", "x", "e", "n", " "]
assert x[0][1] == 15


def num_tiles(board):
    all_squares = list(chain(*board))
    return sum(1 for s in all_squares if s != " ")


def get_tiles_played(board, new_board):
    return [new_board[r][c] for r, c in get_changed_locations(board, new_board)]


class bcolors:
    # https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(f"{bcolors.HEADER}qwertyuiop{bcolors.ENDC}")
print(f"{bcolors.OKBLUE}qwertyuiop{bcolors.ENDC}")
print(f"{bcolors.OKCYAN}qwertyuiop{bcolors.ENDC}")
print(f"{bcolors.OKGREEN}qwertyuiop{bcolors.ENDC}")
print(f"{bcolors.WARNING}qwertyuiop{bcolors.ENDC}")
print(f"{bcolors.FAIL}qwertyuiop{bcolors.ENDC}")
print(f"{bcolors.ENDC}qwertyuiop{bcolors.ENDC}")
print(f"{bcolors.BOLD}qwertyuiop{bcolors.ENDC}")
print(f"{bcolors.UNDERLINE}qwertyuiop{bcolors.ENDC}")
print("regular")

def pp(board):
    num_cols = len(board[0])
    horiz_bars = "  _ " * num_cols

    print(horiz_bars)

    for row in board:
        print(f"| {' | '.join(row)} |")
        print(horiz_bars)


def pp2(board, new_board):
    changed_locations = set(get_changed_locations(board, new_board))

    max_num_digits = 2  #assume we don't have a board size over 99

    num_cols = len(board[0])

    col_nums = list(range(1, num_cols+1))
    col_nums_1 = [x // 10 for x in col_nums]  # tens place
    col_nums_2 = [x % 10 for x in col_nums]  # ones_place
    col_nums_str_1 = " "*(max_num_digits + 1)  + "".join("  " if x == 0 else f" {x}" for x in col_nums_1)
    col_nums_str_2 = " "*(max_num_digits + 1)  + "".join(f" {x}" for x in col_nums_2)

    print("")
    print(col_nums_str_1)
    print(col_nums_str_2)
    horiz_bars = " "*(max_num_digits + 1) + "_" * num_cols * 2

    print(horiz_bars)

    for r, row in enumerate(new_board):
        print(f"{str(r+1).ljust(max_num_digits)}|", end="")
        for c, tile in enumerate(row):
            if (r, c) in changed_locations:
                text = f"{bcolors.OKGREEN}{tile}{bcolors.ENDC}"
            else:
                text = tile
            print(f" {text}", end="")
        print(f"|{str(r+1).ljust(max_num_digits)}")

    print(horiz_bars)
    print(col_nums_str_1)
    print(col_nums_str_2)
    print("")


def simulate_player_turn(board, player_tiles, tile_bag, name, allow_2l_words=True):
    player_tiles = deepcopy(player_tiles)
    options =  best_options(board, tiles=player_tiles, n=1, allow_2l_words=allow_2l_words)
    try:
        best_option = options[0]
    except:
        import ipdb; ipdb.set_trace()
    new_board, score = best_option

    tiles_played = get_tiles_played(board, new_board)
    print(f"{name} is playing tiles {sorted(tiles_played)}")

    # update tile rack:
    for tile_played in tiles_played:
        player_tiles.remove(tile_played)

        if tile_bag:
            player_tiles.append(tile_bag.pop(0))

    return new_board, score, player_tiles, tile_bag


def simulate(allow_2l_words=True):
    # https://en.wikipedia.org/wiki/Scrabble_letter_distributions
    max_num_player_tiles = 7
    board_size = 15
    tile_frequency = {
        "a": 9,
        "b": 2,
        "c": 2,
        "d": 4,
        "e": 12,
        "f": 2,
        "g": 3,
        "h": 2,
        "i": 9,
        "j": 1,
        "k": 1,
        "l": 4,
        "m": 2,
        "n": 6,
        "o": 8,
        "p": 2,
        "q": 1,
        "r": 6,
        "s": 4,
        "t": 6,
        "u": 4,
        "v": 2,
        "w": 2,
        "x": 1,
        "y": 2,
        "z": 1,
    }
    assert (
        sum(tile_frequency.values()) == 98
    )  # mean to be 100 with 2 blanks.  Blank tiles not immplemented yet.

    tile_bag = list(
        chain.from_iterable(
            [char for _ in range(freq)] for char, freq in tile_frequency.items()
        )
    )
    shuffle(tile_bag)
    # TODO investigate issue where we can't find any options
    # but there are words we can play
    # to replicate, remove shuffle here

    a_tiles = [tile_bag.pop(0) for _ in range(max_num_player_tiles)]
    b_tiles = [tile_bag.pop(0) for _ in range(max_num_player_tiles)]

    a_score = 0
    b_score = 0

    board = [[" " for _ in range(board_size)] for _ in range(board_size)]

    while True:
        # player a goes:
        new_board, score, a_tiles, tile_bag = simulate_player_turn(
            board, a_tiles, tile_bag, name="a"
        )
        a_score += score

        pp2(board, new_board)
        board = new_board
        print(f"Player A. tiles: {sorted(a_tiles)},  Score: {a_score}.")
        print(f"Player B. tiles: {sorted(b_tiles)},  Score: {b_score}.")
        print(f"{len(tile_bag)} tiles left in bag.")

        # player b goes:
        new_board, score, b_tiles, tile_bag = simulate_player_turn(
            board, b_tiles, tile_bag, name="b", allow_2l_words=allow_2l_words
        )
        b_score += score

        pp2(board, new_board)
        board = new_board
        print(f"Player A. tiles: {sorted(a_tiles)},  Score: {a_score}.")
        print(f"Player B. tiles: {sorted(b_tiles)},  Score: {b_score}.")
        print(f"{len(tile_bag)} tiles left in bag.")


simulate(allow_2l_words=True)


def read_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def write_json(path, data):
    data_string = json.dumps(data, indent=2)
    # yes, this is pretty dodgy and fragile.  It is to get
    # the json file formatted in a square shape.
    data_string = data_string.replace('",\n      "', '", "')
    data_string = data_string.replace('"\n    ],', '"],')
    data_string = data_string.replace('[\n      "', '["')
    with open(path, "w") as f:
        f.write(data_string)


def make_blank_game(board_size=15, tile_rack_size=7):
    board = [[" " for _ in range(board_size)] for _ in range(board_size)]
    tile_rack = [" " for _ in range(tile_rack_size)]

    data = {"board": board, "tile_rack": tile_rack}
    path = os.path.join(os.path.dirname(__file__), "blank.json")
    write_json(path, data)

make_blank_game()

def advise_from_json(json_path="game.json", allow_2l_words=True, num_options_to_provide=10):
    """
    Give options for the next move to make.

    read current game data from json_path file.

    If the user chooses an option, update the json_path file.

    allow_2l_words can be toggled.  It is useful to filter out some bad
    options given by the tool.  Eg. a long word parallel to another long word
    makes many 2 letter words.  The english_words package seems to have
    a number of 2 letter words that wouldn't be valid eg. "ii", "ed", "wa"
    """
    game_json = read_json(json_path)

    board = game_json["board"]
    tile_rack = game_json["tile_rack"]

    options = best_options(board, tile_rack, n=num_options_to_provide, allow_2l_words=allow_2l_words)

    options.reverse()  # display best at the bottom

    rank_to_option = {
        len(options) - i: option
        for i, option in enumerate(options)
    }

    for rank in sorted(rank_to_option, reverse=True):
        option = rank_to_option[rank]
        new_board, score = option
        print("")
        print("==========================================================")
        print(f"Option number {rank}.  {score} points.")
        pp2(board, new_board)

    print(f"Tile rack: {sorted(tile_rack)}")
    print("")

    choice = input(f"Enter number of option, or press enter: ")

    if choice:
        choice = int(choice)
        option = rank_to_option[choice]
        # clobber the existing json with the new output
        new_board, _ = option
        tiles_played = get_tiles_played(board, new_board)
        new_tile_rack = deepcopy(tile_rack)
        for tile in tiles_played:
            new_tile_rack.remove(tile)
        new_tile_rack.sort()
        new_tiles = input(f"Please enter new tiles added to rack: ")
        new_tile_rack.extend(list(new_tiles))
        new_data = {"board": new_board, "tile_rack": new_tile_rack}
        write_json(json_path, new_data)
        print(f"Finished updating {json_path}")
    else:
        print(f"Not making any changes to {json_path}")


# advise_from_json("t1.json", allow_2l_words=False)
