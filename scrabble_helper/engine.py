"""
Plan:
for each row/column:
filter words by max length (tiles + existing)
filter by containing existing letter groups
filter by issubset of tiles + existing
generate possible new rows/columns.
from this, generate new possible boards
filter to valid boards (checking words indirectly created in other rows/columns)
"""
from scrabble_helper.words import get_english_words, get_scrabble_words
from itertools import chain
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Set, Optional, Any
from math import ceil, floor
from tqdm import tqdm

# We want to be able to  toggle which word source we use.
# But also have a nice default.
# Maybe we need to rethink this.  It might be a little hard to follow
# with DEFAULT_GET_WORD_FN being used so many times in this file.
DEFAULT_GET_WORD_FN = get_english_words


def gen_rows(board):
    yield from board


def gen_cols(board):
    for col in zip(*board):
        yield list(col)


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


def get_row_options(row, tiles, get_words_fn=DEFAULT_GET_WORD_FN):

    existing_chars = [c for c in row if c != " "]
    max_length = len(tiles) + len(existing_chars)

    if max_length > len(row):
        max_length = len(row)

    word_options = get_words_fn(max_len=max_length)
    # word_options = {w for w in word_options if len(w) <= max_length}
    word_options = {
        w
        for w in word_options
        if any(existing_char in w for existing_char in existing_chars)
    }
    all_availiable_chars = tiles + existing_chars
    word_options = {w for w in word_options if is_tile_subset(w, all_availiable_chars)}
    new_row_options = gen_new_rows(row, word_options, tiles)

    return new_row_options


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
    for i, val in enumerate(sequence):
        if val == separator and chunk:
            yield chunk
            chunk = []
            continue
        if val != separator:
            chunk.append((i, val))

    yield chunk


def gen_word_groups(board, include_indices=False):
    for row_num, row in enumerate(gen_rows(board)):
        char_groups = gen_char_groups(row)
        for char_group in char_groups:
            if include_indices:
                yield [((row_num, i), char) for (i, char) in char_group]
            else:
                yield [char for (i, char) in char_group]

    for col_num, col in enumerate(gen_cols(board)):
        char_groups = gen_char_groups(col)
        for char_group in char_groups:
            if include_indices:
                yield [((i, col_num), char) for (i, char) in char_group]
            else:
                yield [char for (i, char) in char_group]



def gen_words(board, include_indices=False):
    for word_group in gen_word_groups(board, include_indices=False):
        yield "".join(word_group)



def board_is_valid(
    board, new_board, unrecognised_words=None, get_words_fn=DEFAULT_GET_WORD_FN
):
    current_words = set(gen_words(board))

    words = get_words_fn()

    current_invalid_words = current_words.difference(words)

    if current_invalid_words:
        # There isn't really anything we can do about this now.  Just log it
        if unrecognised_words is not None:
            unrecognised_words.update(current_invalid_words)

    starting_words = words

    all_valid_words = starting_words.union(current_words)

    new_words = set(gen_words(new_board))

    new_invalid_words = new_words.difference(all_valid_words)

    if new_invalid_words:
        return False

    return True


def board_row_options(
    r, row_options, board, unrecognised_words=None, get_words_fn=DEFAULT_GET_WORD_FN
):
    board_options = []

    for row in row_options:
        # insert into board
        new_board = deepcopy(board)
        new_board[r] = row

        if board_is_valid(
            board,
            new_board,
            unrecognised_words=unrecognised_words,
            get_words_fn=get_words_fn,
        ):
            board_options.append(new_board)

    return board_options


def board_col_options(
    c, col_options, board, unrecognised_words=None, get_words_fn=DEFAULT_GET_WORD_FN
):
    board_options = []

    for col in col_options:
        # insert into board
        new_board = deepcopy(board)
        cols = list(gen_cols(new_board))
        cols[c] = col
        # switch it back to a list of rows:
        new_board = list(gen_cols(cols))

        if board_is_valid(
            board,
            new_board,
            unrecognised_words=unrecognised_words,
            get_words_fn=get_words_fn,
        ):
            board_options.append(new_board)

    return board_options


def start_of_game_words(tiles, max_word_length, get_words_fn=DEFAULT_GET_WORD_FN):
    tiles = set(tiles)

    word_options = {
        word
        for word in get_words_fn()
        if len(word) <= max_word_length and is_tile_subset(word, tiles)
    }
    return word_options


def start_of_game_options(board, tiles, get_words_fn=DEFAULT_GET_WORD_FN):
    middlest_row_index = floor(len(board) / 2)
    num_cols = len(board[middlest_row_index])
    max_word_length = min(len(tiles), num_cols)

    word_options = start_of_game_words(
        tiles, max_word_length, get_words_fn=get_words_fn
    )

    board_options = []

    for word in word_options:
        new_board = deepcopy(board)
        # just put the word roughly in the middle of the most middle row:
        start_col_index = floor((num_cols - len(word)) / 2)
        end_col_index = start_col_index + len(word)
        new_board[middlest_row_index][start_col_index:end_col_index] = list(word)
        board_options.append(new_board)

    return board_options


def get_options(board, tiles, get_words_fn=DEFAULT_GET_WORD_FN):
    all_squares = list(chain(*board))
    is_start_of_game = all(square == " " for square in all_squares)

    unrecognised_words = set()

    if is_start_of_game:
        return start_of_game_options(board, tiles, get_words_fn=get_words_fn)

    all_board_options = []

    print("Looking for words along rows...")

    for r, row in enumerate(gen_rows(board)):
        row_options = get_row_options(row, tiles, get_words_fn=get_words_fn)

        # validate here:
        board_options = board_row_options(
            r,
            row_options,
            board,
            unrecognised_words=unrecognised_words,
            get_words_fn=get_words_fn,
        )
        all_board_options.extend(board_options)

    print("Looking for words along columns...")

    for c, col in enumerate(gen_cols(board)):
        col_options = get_row_options(col, tiles, get_words_fn=get_words_fn)

        # validate here
        board_options = board_col_options(
            c,
            col_options,
            board,
            unrecognised_words=unrecognised_words,
            get_words_fn=get_words_fn,
        )
        all_board_options.extend(board_options)

    print(f"Unrecognised words in existing_board: {sorted(unrecognised_words)}")

    return all_board_options


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



def get_word_score(word_group, pos_to_bonus, changed_locations):
    """Return the score and a string explaining how the score was calculated.

    word_group is a list of (position, tile) tuples

    eg.
    [((0, 0), "d"), ((0, 1), "o"), ((0, 2), "g")]

    the bonus values mean:
    "d": double letter score
    "D": double word score
    "t": triple letter socre
    "T": triple word score
    " ": no bonus
    """
    justification = []
    total_score = 0
    # First count the word score, including any letter bonuses:
    for pos, tile in word_group:
        if pos not in changed_locations:
            continue

        score = get_score(tile)
        bonus = pos_to_bonus.get(pos)
        if bonus == "d":
            justification.append(f"{tile} {score}*2")
            score *= 2
        elif bonus == "t":
            score *= 3
            justification.append(f"{tile} {score}*3")
        else:
            justification.append(f"{tile} {score}")
        total_score += score

    # Now apply any double/triple word scores
    all_bonuses = {pos_to_bonus[pos] for pos in changed_locations if pos in pos_to_bonus}
    for bonus in all_bonuses:
        if bonus == "D":
            justification.append("double word!!")
            total_score *= 2
        elif bonus == "T":
            total_score *= 3
            justification.append("triple word!!!")

    justification_str = " + ".join(justification) + f" = {total_score}"
    return total_score, justification_str


def get_new_word_score(board, new_board, bonus_config=None):
    if bonus_config:
        pos_to_bonus = {
            (row_num, col_num): bonus
            for row_num, row in enumerate(bonus_config)
            for col_num, bonus in enumerate(row)
            if bonus != " "
        }
    else:
        pos_to_bonus = {}
    changed_locations = set(get_changed_locations(board, new_board))

    full_score = 0

    for word_group in gen_word_groups(new_board, include_indices=True):
        tile_positions = {pos for pos, tile in word_group}
        if not changed_locations.intersection(tile_positions):
            # This word does not invole any new tiles
            continue

        word = "".join(char for pos, char in word_group)
        if len(word) == 1:
            # not a word
            continue

        score, justification_string = get_word_score(word_group, pos_to_bonus, changed_locations)

        print(f"{word}: {justification_string}")
        full_score += score

    BONUS_TILE_THRESHOLD = 7
    BONUS_AMOUNT = 50
    if len(changed_locations) >= BONUS_TILE_THRESHOLD:
        print("!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Used over {BONUS_TILE_THRESHOLD} tiles.  {BONUS_AMOUNT} point bonus!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!")
        full_score += BONUS_AMOUNT

    return full_score



def best_options(board, tiles, n=None, get_words_fn=DEFAULT_GET_WORD_FN):
    """Return the n best board options."""
    all_board_options = get_options(board, tiles, get_words_fn=get_words_fn)

    board_score_pairs = [
        (new_board, get_new_word_score(board, new_board))
        for new_board in all_board_options
    ]

    board_score_pairs.sort(key=lambda t: t[1], reverse=True)

    if n is None:
        return board_score_pairs

    return board_score_pairs[:n]


def num_tiles(board):
    all_squares = list(chain(*board))
    return sum(1 for s in all_squares if s != " ")


def get_tiles_played(board, new_board):
    return [new_board[r][c] for r, c in get_changed_locations(board, new_board)]
