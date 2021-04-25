import json
import os
from copy import deepcopy

from scrabble_helper.engine import best_options, get_tiles_played
from scrabble_helper.display import pp2


def board_files_dir():
    return os.path.join(os.path.dirname(__file__), "..", "board_files")


def make_blank_game(board_size=15, tile_rack_size=7):
    board = [[" " for _ in range(board_size)] for _ in range(board_size)]
    tile_rack = [" " for _ in range(tile_rack_size)]

    data = {"board": board, "tile_rack": tile_rack}
    path = os.path.join(board_files_dir(), "blank.json")
    write_json(path, data)


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


def advise_from_json(board_name, num_options_to_provide=30):
    """
    Give options for the next move to make.

    read current game data from json_path file.

    If the user chooses an option, update the json_path file.

    board_name must be a file within scrabble_helper/board_files.
    eg. for board_name = "abc", there should be a file called "abc.json"
    """
    if board_name == "blank":
        raise ValueError("board_name can not be blank")

    json_path = os.path.join(board_files_dir(), board_name + ".json")

    game_json = read_json(json_path)

    board = game_json["board"]
    tile_rack = game_json["tile_rack"]

    if all(tile == " " for tile in tile_rack):
        raise ValueError(f"tile_rack is empty: {tile_rack}")

    options = best_options(board, tile_rack, n=num_options_to_provide)
    options.reverse()  # display best at the bottom

    rank_to_option = {len(options) - i: option for i, option in enumerate(options)}

    for rank in sorted(rank_to_option, reverse=True):
        option = rank_to_option[rank]
        print("")
        print("==========================================================")
        print(f"Option number {rank}.  {option.score} points.")
        pp2(board, option.new_board)

    print(f"Tile rack: {sorted(tile_rack)}")
    print("")

    choice = input("Enter number of option, or press enter: ")

    if choice:
        choice = int(choice)
        option = rank_to_option[choice]
        # clobber the existing json with the new output
        tiles_played = get_tiles_played(board, option.new_board)
        new_tile_rack = deepcopy(tile_rack)
        for tile in tiles_played:
            new_tile_rack.remove(tile)
        new_tile_rack.sort()
        new_tiles = input("Please enter new tiles added to rack: ")
        new_tile_rack.extend(list(new_tiles))
        new_data = {"board": option.new_board, "tile_rack": new_tile_rack}
        write_json(json_path, new_data)
        print(f"Finished updating {json_path}")
    else:
        print(f"Not making any changes to {json_path}")


advise_from_json("test")
