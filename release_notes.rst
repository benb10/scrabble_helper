

- add functionality to create word cache files.  These files
  will be used a shortcuts when we know some popular letters
  that the new word will NOT contain.  This allows the tool to
  run much faster eg. if the word does not contain "e", there
  are 70% fewer words to consider.
- add a second word source collins_scrabble_words_2019.txt
  with around 270,000 words (10 times as many as english_words).
  english_words is still the default because the new
  one is too slow
- refactor tests into separate files
- refactor to have separate files for engine, display, simulator and words
- add blacklist to exclude some words from english_words lib
- MVP
