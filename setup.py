from setuptools import setup, find_packages

setup(
    name='tile_match_gym',
    version='0.0.1',
    install_requires=[
        'numpy',
        'importlib-metadata; python_version == "3.8.10"',
    ],
    packages=find_packages(
        # All keyword arguments below are optional:
        where='tile_match_gym',
        exclude=['tests'],  
    ),
    description="A set of reinforcement learning environments for tile matching games, consistent with the OpenAI Gym API.",
    url="https://www.github.com/akshilpatel/tile_match_gym",
    author="Akshil Patel, James Elson"
)