from scrabble_helper.engine import (
    best_options,
    board_col_options,
    board_is_valid,
    board_row_options,
    gen_char_groups,
    gen_cols,
    gen_new_rows,
    gen_words,
    get_char_permutations,
    get_new_word_score,
    get_options,
    get_row_options,
    get_score,
    is_tile_subset,
    row_is_valid,
    start_of_game_words,
)
from scrabble_helper.words import get_scrabble_words


def test_gen_cols():
    assert list(gen_cols([[1, 2], [3, 4]])) == [[1, 3], [2, 4]]


def test_is_tile_subset():
    assert is_tile_subset("abcd", ["a", "b", "c"]) is False
    assert is_tile_subset("abcd", ["a", "b", "c", "d"]) is True
    assert is_tile_subset("abcd", ["a", "b", "c", "d", "e", "e"]) is True
    assert is_tile_subset("aadd", ["a", "b", "c", "d"]) is False
    assert is_tile_subset("aaad", ["a", "b", "c", "d", "a"]) is False
    assert is_tile_subset("aaad", ["a", "b", "c", "d", "a", "a"]) is True


def test_row_is_valid():
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


def test_gen_new_rows():
    x = gen_new_rows([" ", " ", "t"], ["net", "tea", "met"], tiles=["e", "n", "m", "a"])

    assert sorted(x) == [["m", "e", "t"], ["n", "e", "t"]]


def test_get_row_options():
    r = [" ", " ", " ", " ", " ", " ", " ", "d", " ", " ", "d", "a", "c", "c", "a"]
    tiles = ["a", "a", "b", "b", "c", "c", "d"]

    x = get_row_options(r, tiles)

    assert [
        " ",
        " ",
        " ",
        " ",
        " ",
        "b",
        "a",
        "d",
        " ",
        " ",
        "d",
        "a",
        "c",
        "c",
        "a",
    ] in x
    r = [" ", " ", " ", " ", " ", " ", " ", "e", " ", "a", "w", "a", "y", " ", " "]

    x = get_row_options(r, tiles=["a", "a", "a", "c", "e", "i", "v"])

    # 'w' used in wrong place
    bad_row = [
        " ",
        " ",
        " ",
        "w",
        "a",
        "i",
        "v",
        "e",
        " ",
        "a",
        "w",
        "a",
        "y",
        " ",
        " ",
    ]
    assert bad_row not in x
    assert len(x) == 16, len(x)

    x = get_row_options(
        [" ", " ", "l", " ", " ", " ", " ", " "],
        tiles=["a", "i", "o", "p", "r", "t", "y"],
    )

    bad_word = "lotto"
    # demonstrate that this is in words:
    assert bad_word in get_scrabble_words()
    bad_row = list(bad_word)
    # bad row uses 'o' and 't' multiple times.
    # In a previous bug, this was possible.
    # make sure it doesn't happen
    assert bad_row not in x
    assert len(x) == 64, len(x)

    x = get_row_options(row=[" ", " ", " ", "e", " "], tiles=["b", "c", "d", "e"])

    assert sorted(x) == [
        [" ", " ", " ", "e", "e"],
        [" ", " ", "b", "e", " "],
        [" ", " ", "b", "e", "d"],
        [" ", " ", "b", "e", "e"],
        [" ", " ", "c", "e", "e"],
        [" ", " ", "d", "e", "b"],
        [" ", " ", "d", "e", "e"],
        [" ", " ", "e", "e", " "],
        [" ", "b", "e", "e", " "],
        [" ", "c", "e", "e", " "],
        [" ", "d", "e", "e", " "],
        ["b", "e", "d", "e", " "],
        ["c", "e", "d", "e", " "],
        ["d", "e", "b", "e", " "],
    ]

    x = get_row_options(row=[" ", " ", "t"], tiles=["e", "m", "n"])

    assert sorted(x) == [["m", "e", "t"], ["n", "e", "t"]]


def test_gen_char_groups():
    assert list(gen_char_groups([" ", " ", "a", " ", "b", "c"])) == [
        [(2, "a")],
        [(4, "b"), (5, "c")],
    ]


def test_gen_words():
    b = [["a", "b", "c"], [" ", " ", "c"]]

    assert list(gen_words(b)) == ["abc", "cc"]


def test_board_is_valid():
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
    # fmt: on

    assert board_is_valid(orig, b1) is False
    assert board_is_valid(orig, b2) is True


def test_board_row_options():
    b = [[" ", " ", "n"], [" ", " ", "o"], [" ", " ", "d"]]
    x = board_row_options(2, row_options=[["o", "d", "d"], ["f", "e", "d"]], board=b)

    assert x == [
        [[" ", " ", "n"], [" ", " ", "o"], ["o", "d", "d"]],
        [[" ", " ", "n"], [" ", " ", "o"], ["f", "e", "d"]],
    ]


def test_board_col_options():
    b = [[" ", " ", " "], [" ", " ", " "], ["d", "o", "g"]]
    x = board_col_options(0, col_options=[["n", "o", "d"], ["f", "e", "d"]], board=b)

    assert x == [
        [["n", " ", " "], ["o", " ", " "], ["d", "o", "g"]],
        [["f", " ", " "], ["e", " ", " "], ["d", "o", "g"]],
    ]


def test_start_of_game_words():
    x = start_of_game_words(["a", "b", "c", "d", "e"], 10)
    assert len(x) == 23, len(x)
    assert "bead" in x


def test_get_options():
    b = [
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", "z", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "b", "l", "a", "k", "e", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "l", "o", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "a", "t", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "n", "y", "c", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "k", " ", "h", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", "o", "o", "z", "e", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", "k", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", "z", "e", "r", "o", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
    ]
    tiles = ["e", "h", "k", "l", "o", "t", "z"]

    options = get_options(b, tiles)

    # In a previous bug, this would give no options because of the 2-letter words
    # already on the board
    assert len(options) == 207, len(options)

    b = [[" ", " ", " "], [" ", " ", " "], ["c", "a", "t"]]

    x = get_options(b, ["a", "e", "h", "l", "n"])

    assert len(x) == 20, len(x)


def test_get_score():
    assert get_score("qpyy") == 21


def test_get_new_word_score():
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

    assert get_new_word_score(b, nb) == 5


def test_get_new_word_score_bonus_config():
    # fmt: off
    b = [
        [" ", " ", " ", " ", "n"],
        [" ", " ", "a", "d", "o"],
        [" ", " ", "b", " ", "d"],
    ]
    nb = [
        ["x", "q", "p", "e", "n"],
        [" ", " ", "a", "d", "o"],
        [" ", " ", "b", " ", "d"],
    ]
    bonus_config = [
        ["D", "d", "T", "t", "t"],
        [" ", " ", " ", " ", " "],
        [" ", "D", " ", " ", " "],
    ]
    # fmt: on

    # 2*e = 2 + 3 for pen = 5 * triple word = 15
    score, jst_strings = get_new_word_score(
        b, nb, bonus_config=bonus_config, return_jst_strings=True
    )

    assert score == 236
    assert jst_strings == [
        "xqpen: x 8 + q 10*2 + p 3 + e 1*3 + n 1 + double word!! + triple word!!! = 210",
        "pab: p 3 + a 1 + b 3 + triple word!!! = 21",
        "ed: e 1*3 + d 2 = 5",
    ]


def test_best_options():
    b = [[" ", " ", " "], [" ", " ", " "], ["c", "a", "t"]]

    x = best_options(
        b, tiles=["e", "x", "j", "k", "z", "a", "f", "r", "n"], n=1, bonus_config=None
    )

    assert x[0].new_board == [[" ", "z", " "], [" ", "e", " "], ["c", "a", "t"]]
    assert x[0].score == 12


def test_best_options_bonus_config():
    b = [[" ", " ", " "], [" ", " ", " "], ["c", "a", "t"]]
    bonus_config = [[" ", " ", "D"], [" ", " ", " "], [" ", " ", " "]]

    x = best_options(
        b,
        tiles=["e", "x", "j", "k", "z", "a", "f", "r", "n"],
        n=1,
        bonus_config=bonus_config,
    )

    assert x[0].new_board == [[" ", " ", "j"], [" ", " ", "e"], ["c", "a", "t"]]
    assert x[0].score == 20


def test_best_options_start_of_game():
    b = [[" " for _ in range(7)] for _ in range(7)]

    x = best_options(
        b, tiles=["e", "x", "j", "z", "a", "r", "n", "l", "w", "e", "a", "e"], n=5
    )

    assert len(x) == 5
    assert x[0].new_board[3] == ["w", "r", "a", "x", "l", "e", " "]
    assert x[0].score == 40


def test_get_char_permutations():
    assert get_char_permutations("ab") == {"ab", "ba"}
    assert get_char_permutations("abc") == {
        "ab",
        "abc",
        "ba",
        "cb",
        "bac",
        "ca",
        "cba",
        "acb",
        "bc",
        "ac",
        "cab",
        "bca",
    }
    assert len(get_char_permutations("abczdesw")) == 109_592
