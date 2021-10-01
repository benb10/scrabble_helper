from scrabble_helper.words import get_scrabble_words


def test_get_scrabble_words():
    x = get_scrabble_words()
    # Should be a set of around 25,000 strings
    assert isinstance(x, set)
    assert all(isinstance(word, str) for word in x)

    # check all lower case chars:
    lower_case_chars = {chr(i) for i in range(97, 123)}
    all_chars = set()
    for word in x:
        all_chars.update(word)
    assert all_chars == lower_case_chars

    # May need to update this sometime.  Just check we are in the expected
    # ballpark:
    assert 270_000 < len(x) < 290_000


def test_get_scrabble_words_max_len():
    words_8 = get_scrabble_words(max_len=8)
    assert max(len(w) for w in words_8) == 8
    assert 110_000 < len(words_8) < 130_000

    words_4 = get_scrabble_words(max_len=4)
    assert max(len(w) for w in words_4) == 4
    assert 6_000 < len(words_4) < 8_000
