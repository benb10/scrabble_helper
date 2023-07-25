from copy import deepcopy
from itertools import chain
from random import shuffle
from time import time

from scrabble_helper.bonus_configs import default_bonus_config
from scrabble_helper.display import pp2
from scrabble_helper.engine import best_options, get_tiles_played


def simulate_player_turn(board, player_tiles, tile_bag, name):
    player_tiles = deepcopy(player_tiles)
    start_time = time()
    options = best_options(
        board, tiles=player_tiles, n=1, bonus_config=default_bonus_config
    )
    best_option = options[0]

    tiles_played = get_tiles_played(board, best_option.new_board)
    print(f"{name} is playing tiles {sorted(tiles_played)}")
    print(f"Scored {best_option.score} points.")
    print(f"Took {round(time() - start_time, 2)} seconds to determine options.")

    # update tile rack:
    for tile_played in tiles_played:
        player_tiles.remove(tile_played)

        if tile_bag:
            player_tiles.append(tile_bag.pop(0))

    return best_option.new_board, best_option.score, player_tiles, tile_bag


def simulate():
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
    )  # mean to be 100 with 2 blanks.  Blank tiles not implemented yet.

    tile_bag = list(
        chain.from_iterable(
            [char for _ in range(freq)] for char, freq in tile_frequency.items()
        )
    )
    shuffle(tile_bag)

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
            board, b_tiles, tile_bag, name="b"
        )
        b_score += score

        pp2(board, new_board)
        board = new_board
        print(f"Player A. tiles: {sorted(a_tiles)},  Score: {a_score}.")
        print(f"Player B. tiles: {sorted(b_tiles)},  Score: {b_score}.")
        print(f"{len(tile_bag)} tiles left in bag.")


simulate()
