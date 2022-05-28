from .craters import Craters
import numpy as np

def test_simple_bottom_right_boundaries():
  right_x, bottom_y = Craters.bottom_right_boundaries(np.array([10, 20]), 4)
  assert right_x == 7
  assert bottom_y == 12


def test_simple_top_left_boundaries():
  left_x, top_y = Craters.top_left_boundaries(np.array([10, 20]), 4)
  assert left_x == 3
  assert top_y == 8
