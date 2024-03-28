from setuptools import setup

setup(
    name="tile-match-gym",
    version="0.0.4",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    data_files=[
        ("images",
            [
            "images/bomb.png",
            "images/cookie.png",
            "images/horizontal.png",
            "images/vertical.png",
            "images/ordinary.png"
            ],
        )
    ],
)