from setuptools import setup


setup(
    name="scrabble_helper",
    version="0.2",
    install_requires=[
        "tqdm~=4.60",
    ],
    entry_points={
        "console_scripts": [
            "scrabble = scrabble_helper.main:main",
            "simulate = scrabble_helper.simulator:simulate",
            "create_caches = scrabble_helper.words:create_cache_files",
        ]
    },#
)
