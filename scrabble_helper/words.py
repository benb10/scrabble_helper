"""This file has functions for getting word lists, which are
used to determine possible moves.
"""
from english_words import english_words_lower_alpha_set
from typing import List, Set, Optional, Any
from time import time
from tqdm import tqdm
from collections import defaultdict
from functools import lru_cache
import os

import json


from scrabble_helper.blacklist import blacklist


def get_az():
    return {chr(i) for i in range(ord("a"), ord("z") + 1)}


def filter_to_az(words):
    # remove anything with non letter chars eg. &
    lower_case_chars = get_az()
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


def divide_by_contained_chars(words):
    """Return a dict mapping a char to all the words containing the char

    eg.
    {
        "a": {"cat", "ant", "grass", "able",...},
        "b": {"able", "bow",...},
        ...
    }
    """
    char_to_words = defaultdict(set)
    for word in words:
        for char in word:
            char_to_words[char].add(word)
    return char_to_words


def divide_by_missing_char(words):
    """
    """
    missing_char_to_words = defaultdict(set)
    for word in words:
        az = get_az()

        for char in az.difference(set(word)):
            missing_char_to_words[char].add(word)
    return missing_char_to_words


def read_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def write_json(path, data):
    with open(path, "w") as f:
        f.write(json.dumps(data))


WW_CACHE_PREFIX = "words_without_cache_"
CACHES_DIR = os.path.join(os.path.dirname(__file__), "caches")
# Use a few of the most common chars
CHARS_USED_IN_CACHE = ["e", "s", "i", "a", "r", "n", "t", "o", "l", "c"]
MAX_NUM_CHARS_PER_CACHE = 5


def get_words_without_caches():
    files = os.listdir("caches")
    cache_names = [file[len(WW_CACHE_PREFIX) :] for file in files]
    return sorted(cache_names)


def get_dir_size(dir):
    return sum(os.path.getsize(os.path.join(dir, f)) for f in os.listdir(dir))


def get_cache(missing_chars):
    missing_char_str = "".join(sorted(set(missing_chars)))
    words = read_json(os.path.join(CACHES_DIR, WW_CACHE_PREFIX + missing_char_str))
    return set(words)


def create_cache_files():
    """
    This will create many cache files in the directory CACHES_DIR.

    These will be used later on so we can get much faster
    word suggestions.
    """
    start_time = time()
    for file in os.listdir(CACHES_DIR):
        os.remove(os.path.join(CACHES_DIR, file))
    from pathlib import Path

    Path(CACHES_DIR).mkdir(parents=True, exist_ok=True)
    words = get_scrabble_words()
    write_json(f"{CACHES_DIR}/{WW_CACHE_PREFIX}", list(words))

    for _ in range(MAX_NUM_CHARS_PER_CACHE):
        ww_caches = set(get_words_without_caches())
        for existing_cache in tqdm(sorted(ww_caches)):
            # cache_chars = {char for char in cache}
            existing_words = read_json(
                f"{CACHES_DIR}/{WW_CACHE_PREFIX}{existing_cache}"
            )
            for new_char in sorted(CHARS_USED_IN_CACHE):
                if new_char in existing_cache:
                    # This entry is already excluding new char.
                    # Nothing to add here
                    continue

                new_key = existing_cache + new_char
                if "".join(sorted(new_key)) in ww_caches:
                    # we already have a cache for this char set
                    continue

                new_words = [w for w in existing_words if new_char not in w]
                write_json(f"{CACHES_DIR}/{WW_CACHE_PREFIX}{new_key}", new_words)

    cache_size = get_dir_size(CACHES_DIR)
    cache_size_gb = round(cache_size / 10 ** 9, 3)
    print(f"finished in {time() - start_time} seconds")
    print(f"Created {len(os.listdir(CACHES_DIR))} cache files ({cache_size_gb} GB).")


def get_letter_frequencies(words):
    """
    Last output:

    char j, freq 1.5 %
    char q, freq 1.52 %
    char x, freq 2.57 %
    char z, freq 4.08 %
    char w, freq 6.71 %
    char k, freq 8.06 %
    char v, freq 8.09 %
    char f, freq 9.68 %
    char y, freq 14.07 %
    char b, freq 15.57 %
    char h, freq 20.88 %
    char g, freq 22.84 %
    char m, freq 23.59 %
    char p, freq 24.14 %
    char d, freq 27.03 %
    char u, freq 27.19 %
    char c, freq 31.47 %
    char l, freq 39.32 %
    char o, freq 46.43 %
    char t, freq 47.61 %
    char n, freq 48.53 %
    char r, freq 52.26 %
    char a, freq 54.96 %
    char i, freq 60.0 %
    char s, freq 61.72 %
    char e, freq 69.43 %
    """
    char_to_count = defaultdict(int)

    for word in words:
        for char in set(word):
            char_to_count[char] += 1

    num_words = len(words)

    char_to_percent = {
        char: round(100 * count / num_words, 2) for char, count in char_to_count.items()
    }
    for char, percent in sorted(char_to_percent.items(), key=lambda t: t[1]):
        print(f"char {char}, freq {percent} %")

    return char_to_count
