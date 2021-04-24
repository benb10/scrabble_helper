"""This file has functions for getting word lists, which are
used to determine possible moves.
"""
from english_words import english_words_lower_alpha_set
from typing import List, Set, Optional, Any
from collections import defaultdict
from functools import lru_cache
import os

# There seems to be many short words in the library
# that aren't legitimate scrabble words.
# This is a blacklist we can build up manually to
# exclude these words.
blacklist = {
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "ac",
    "ad",
    "ak",
    "al",
    "ar",
    "ax",
    "az",
    "bp",
    "bs",
    "ca",
    "cf",
    "cs",
    "ct",
    "cz",
    "dc",
    "de",
    "dr",
    "ds",
    "du",
    "ed",
    "el",
    "em",
    "en",
    "es",
    "et",
    "fe",
    "fl",
    "fm",
    "fs",
    "ft",
    "ga",
    "ge",
    "gm",
    "gs",
    "gu",
    "hs",
    "ia",
    "ii",
    "il",
    "im",
    "io",
    "iq",
    "ir",
    "iv",
    "ix",
    "jo",
    "jr",
    "js",
    "ks",
    "ku",
    "ky",
    "la",
    "lo",
    "ls",
    "md",
    "mi",
    "mn",
    "mo",
    "mr",
    "ms",
    "mt",
    "mu",
    "nc",
    "nd",
    "ne",
    "nh",
    "nj",
    "nm",
    "ns",
    "nu",
    "nv",
    "nw",
    "ny",
    "os",
    "pa",
    "ph",
    "pi",
    "pl",
    "pm",
    "po",
    "pr",
    "ps",
    "qs",
    "rd",
    "re",
    "ri",
    "rs",
    "sa",
    "sc",
    "sd",
    "se",
    "ss",
    "st",
    "sw",
    "ti",
    "tn",
    "ts",
    "tv",
    "tx",
    "uk",
    "un",
    "ut",
    "va",
    "vi",
    "vs",
    "vt",
    "wa",
    "wi",
    "ws",
    "wu",
    "wv",
    "wy",
    "xi",
    "xs",
    "ye",
    "ys",
    "zs",
}


def filter_to_az(words):
    # remove anything with non letter chars eg. &
    lower_case_chars = {chr(i) for i in range(ord("a"), ord("z") + 1)}
    return {word for word in words if set(word).issubset(lower_case_chars)}


@lru_cache(maxsize=1024)
def get_english_words(max_len=None):
    """We use the python lib english_words https://pypi.org/project/english-words/.

    from english_words import english_words_lower_alpha_set as words
    At the time of writing, it has around 25,000 words
    """
    words = english_words_lower_alpha_set
    words = words.difference(blacklist)
    words = filter_to_az(words)

    if max_len is not None:
        words = {w for w in words if len(w) <= max_len}

    print(f"Produced {len(words)} words")
    return words


def word_sources_dir():
    return os.path.join(os.path.dirname(__file__), "word_sources")


@lru_cache(maxsize=1024)
def get_scrabble_words(max_len=None):
    """This reads in a word list which has 279,496 scrabble words.

    https://boardgames.stackexchange.com/questions/38366/latest-collins-scrabble-words-list-in-text-file
    https://drive.google.com/file/d/1oGDf1wjWp5RF_X9C7HoedhIWMh5uJs8s/view
    """
    # So we know when the function actually runs
    # because it can't find anything in the lru_cache
    print(f"Running get_scrabble_words with max_len {max_len}")

    word_file = os.path.join(word_sources_dir(), "collins_scrabble_words_2019.txt")
    with open(word_file, "r") as f:
        text = f.read()

    words = set(text.strip().split("\n"))
    # user lower case
    words = {word.lower() for word in words}
    words = filter_to_az(words)
    words = words.difference(blacklist)

    if max_len is not None:
        words = {w for w in words if len(w) <= max_len}

    print(f"Produced {len(words)} words")
    return words

# helper functions:

def divide_by_len(words):
    """Helper function to check word lists."""
    len_to_words = defaultdict(list)
    for word in sorted(words):
        len_to_words[len(word)].append(word)
    return len_to_words


def get_missing_letter_stats(words, missing_chars=["e", "t", "a", "o", "i", "n"]):
    """Print some info about how the length can be cut down.
    """
    print(f"There are {len(words)} words")

    for missing_char in missing_chars:
        sub_len = len([w for w in words if missing_char not in w])
        print(f"Words without {missing_char}: {sub_len}")

# get_missing_letter_stats(get_scrabble_words(max_len=8))
# import ipdb; ipdb.set_trace()
