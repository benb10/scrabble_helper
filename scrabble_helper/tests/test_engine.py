import pytest
from time import time
from dataclasses import dataclass

from scrabble_helper.words import get_english_words, get_scrabble_words

from scrabble_helper.engine import (
    gen_cols,
    is_tile_subset,
    row_is_valid,
    gen_new_rows,
    get_row_options,
    gen_char_groups,
    gen_words,
    board_is_valid,
    board_row_options,
    board_col_options,
    start_of_game_words,
    get_options,
    get_score,
    get_new_word_score,
    best_options,
    get_char_permutations,
)


@pytest.mark.skip(reason="Skip this until changes settle down")
def test_performance_benchmarks():
    """Check some KPIs for different methods of finding board options.

    This test could be a little flakey (eg. the runtime varying each time),
    but I think it is valuable to be aware of any performance regressions.

    The KPIs are:
    - highest score
    - num options
    - runtime

    The different methods are:
    - word fn get_english_words
    - word fn get_scrabble_words
    """

    b = [
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "b", "n", "t", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "d", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "w", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
    ]

    @dataclass
    class Range:
        min: float
        max: float

        def within_range(self, x):
            return self.min <= x <= self.max

    @dataclass
    class Benchmarks:
        highest_score_range: Range
        num_options_range: Range
        runtime_range: Range  # seconds

    method_to_fn_and_benchmarks = {
        "get_english_words": (
            get_english_words,
            Benchmarks(
                highest_score_range=Range(min=12, max=14),
                num_options_range=Range(min=200, max=400),
                runtime_range=Range(min=0, max=5),
            ),
        ),
        "get_scrabble_words": (
            get_scrabble_words,
            Benchmarks(
                highest_score_range=Range(min=61, max=70),
                num_options_range=Range(min=500, max=1000),
                runtime_range=Range(min=60, max=60),
            ),
        ),
    }

    benchmark_regressions = []

    for method_name, (fn, benchmarks) in method_to_fn_and_benchmarks.items():
        start_time = time()
        options = best_options(
            b, tiles=["a", "e", "i", "n", "t", "e", "b"], get_words_fn=fn
        )

        runtime = round(time() - start_time, 2)
        highest_score = options[0].score
        num_options = len(options)

        if not benchmarks.highest_score_range.within_range(highest_score):
            message = f"{method_name} highest_score {highest_score}"
            print(message)
            benchmark_regressions.append(
                f"{message} not in range {benchmarks.highest_score_range}."
            )

        if not benchmarks.num_options_range.within_range(num_options):
            message = f"{method_name} num_options {num_options}"
            print(message)
            benchmark_regressions.append(
                f"{message} not in range {benchmarks.num_options_range}."
            )

        if not benchmarks.runtime_range.within_range(runtime):
            message = f"{method_name} runtime {runtime}"
            print(message)
            benchmark_regressions.append(
                f"{message} not in range {benchmarks.runtime_range}."
            )

    assert len(benchmark_regressions) == 0, "\n" + "\n".join(benchmark_regressions)


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
    x = get_row_options(r, tiles, get_words_fn=get_english_words)
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

    x = get_row_options(
        r, tiles=["a", "a", "a", "c", "e", "i", "v"], get_words_fn=get_english_words
    )

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
        get_words_fn=get_english_words,
    )
    bad_row = ["p", "o", "l", "y", "t", "y", "p", "y"]
    # demonstrate that this is in words:
    assert "polytypy" in get_english_words()
    # bad row uses 'y' multiple times.
    # In a previous bug, this was possible.
    # make sure it doesn't happen
    assert bad_row not in x
    assert len(x) == 64, len(x)

    x = get_row_options(
        row=[" ", " ", " ", "e", " "],
        tiles=["b", "c", "d", "e"],
        get_words_fn=get_english_words,
    )

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

    x = get_row_options(
        row=[" ", " ", "t"], tiles=["e", "m", "n"], get_words_fn=get_english_words
    )

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
    assert board_is_valid(orig, b1, get_words_fn=get_english_words,) is False
    assert board_is_valid(orig, b2, get_words_fn=get_english_words,) is True
    # fmt: on


def test_board_row_options():
    b = [[" ", " ", "n"], [" ", " ", "o"], [" ", " ", "d"]]
    x = board_row_options(
        2,
        row_options=[["o", "d", "d"], ["f", "e", "d"]],
        board=b,
        get_words_fn=get_english_words,
    )
    assert len(x) == 2

    b = [[" ", " ", "n"], ["d", " ", "o"], [" ", " ", "d"]]
    x = board_row_options(
        2,
        row_options=[["o", "d", "d"], ["f", "e", "d"]],
        board=b,
        get_words_fn=get_english_words,
    )
    assert len(x) == 1


def test_board_col_options():
    b = [[" ", " ", " "], [" ", " ", " "], ["d", "o", "g"]]
    x = board_col_options(
        0,
        col_options=[["n", "o", "d"], ["f", "e", "d"]],
        board=b,
        get_words_fn=get_english_words,
    )
    assert len(x) == 2


def test_start_of_game_words():
    x = start_of_game_words(
        ["a", "b", "c", "d", "e"], 10, get_words_fn=get_english_words
    )
    assert len(x) == 13, len(x)
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
    options = get_options(b, tiles, get_words_fn=get_english_words)
    # In a previous bug, this would give no options because of the 2 letter words
    # already on the board
    assert len(options) == 64, len(options)

    b = [[" ", " ", " "], [" ", " ", " "], ["c", "a", "t"]]
    x = get_options(b, ["a", "e", "h", "l", "n"], get_words_fn=get_english_words)
    assert len(x) == 12, len(x)


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
        b,
        tiles=["e", "x", "j", "k", "z", "a", "f", "r", "n"],
        get_words_fn=get_english_words,
        n=1,
        bonus_config=None,
    )
    assert x[0].new_board == [[" ", " ", "j"], [" ", " ", "e"], ["c", "a", "t"]]
    assert x[0].score == 10


def test_best_options_bonus_config():
    b = [[" ", " ", " "], [" ", " ", " "], ["c", "a", "t"]]

    bonus_config = [[" ", " ", "D"], [" ", " ", " "], [" ", " ", " "]]

    x = best_options(
        b,
        tiles=["e", "x", "j", "k", "z", "a", "f", "r", "n"],
        get_words_fn=get_english_words,
        n=1,
        bonus_config=bonus_config,
    )
    assert x[0].new_board == [[" ", " ", "j"], [" ", " ", "e"], ["c", "a", "t"]]
    assert x[0].score == 20


def test_best_options_start_of_game():
    b = [[" " for _ in range(7)] for _ in range(7)]
    x = best_options(
        b,
        tiles=["e", "x", "j", "z", "a", "r", "n", "l", "w", "e", "a", "e"],
        n=5,
        get_words_fn=get_english_words,
    )
    assert len(x) == 5
    assert x[0].new_board[3] == [" ", "w", "a", "x", "e", "n", " "]
    assert x[0].score == 30


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
