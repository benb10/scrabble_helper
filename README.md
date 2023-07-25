# scrabble_helper

This is a tool which will suggest words to play in a game of scrabble.

As you play, you update a json file which records
tiles on the board and in your rack, and running
the tool will print out word suggestions.


# Installation
1. clone this repo
1. pip install -e .
1. Populate the caches directory (we use this so that the tool
   runs in a reasonable amount of time).  On the command line, enter:
   "create_caches".
1. Make a copy of `blank.json `in the board_files directory called `game.json`.
   Rename any old games to `game_{number}.json`, so that scrabble_helper knows
   they are not the current game.
1. Update "game.json" with the tiles you have in your tile rack.
1. On the command line, enter "scrabble".

# Running tests
```
pip install -e .
pip install -r reqs-test.txt
pytest
```
