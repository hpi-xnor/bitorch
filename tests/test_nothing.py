from numpy.testing import assert_raises, assert_almost_equal
import numpy as np


def test_true_assertion():
    assert True


def test_that_something_fails():
    assert_raises(AssertionError, assert_almost_equal, np.array([1]), np.array([2]))
