import pytest
import numpy as np

from tile_match_gym.board import Board
from tile_match_gym.utils import print_board_diffs
from typing import Optional, List, Tuple

def test_activate_special():
    # Bomb

    # V Laser

    # H Laser

    # Cookie

    # Adding to queue.

    # Test that a special can't be readded to queue after activation due to another tile activation


    # Test case that cookie is activated but surrounded already by colourless specials - current code does not work.
    assert False

def test_get_special_creation_pos():
    # Match of length 3 where middle is not special

    # Match of even length.

    # Match where middle is special.

    # Match where non-middle is special.

    # Match where all are already special... is this possible?


    assert False
