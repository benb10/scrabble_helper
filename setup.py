from setuptools import setup


setup(
    name="scrabble_helper",
    version="0.1",
    install_requires=[
        "english_words>=1.0.3,<2",
    ],
    entry_points={}  # TODO.  Something like "console_scripts": ["scrabble = main:run"] ?
)
