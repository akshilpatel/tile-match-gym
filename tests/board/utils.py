
import pytest
from src.board import Board
import numpy as np
from typing import Optional, List, Tuple

# create an array of alternating 2's and 3's
def create_alternating_array(height: int, width: int) -> np.ndarray:
    arr = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            arr[i, j] = (i + j) % 2 + 2
    return arr

def contains_threes(arr: np.ndarray) -> bool:
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if j < cols - 2 and arr[i, j] == arr[i, j + 1] == arr[i, j + 2]:
                return True
    if i < rows - 2 and arr[i, j] == arr[i + 1, j] == arr[i + 2, j]:
        return True
    return False

