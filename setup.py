from setuptools import setup


setup(
    name="scrabble_helper",
    version="0.1",
    install_requires=[
        "english_words>=1.0.3,<2",
        # "tqdm>=4.60.0,<5"
    ],
    entry_points={"console_scripts": [
        "scrabble = scrabble_helper.main:main",
        "simulate = scrabble_helper.simulator:simulate",
    ]},
)
