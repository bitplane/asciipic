import numpy as np
import pytest

from asciipic.sampling import NUM_SAMPLES, enhance_contrast


def test_enhance_contrast_uniform_grid_unchanged():
    grid = np.full((3, 4, NUM_SAMPLES), 0.5)
    result = enhance_contrast(grid, exponent=2.0)
    np.testing.assert_allclose(result, grid)


def test_enhance_contrast_exponent_one_is_identity():
    rng = np.random.default_rng(42)
    grid = rng.random((5, 8, NUM_SAMPLES))
    result = enhance_contrast(grid, exponent=1.0)
    np.testing.assert_allclose(result, grid)


def test_enhance_contrast_dark_next_to_bright_gets_darker():
    grid = np.zeros((1, 2, NUM_SAMPLES))
    grid[0, 0, :] = 0.3
    grid[0, 1, :] = 0.9
    result = enhance_contrast(grid, exponent=2.0)

    # Left cell's right-edge positions should be darker than 0.3
    right_edge = [2, 5, 8, 11, 14]
    for idx in right_edge:
        assert result[0, 0, idx] < 0.3

    # Left cell's interior positions should be unchanged
    interior = [4, 7, 10]
    for idx in interior:
        assert result[0, 0, idx] == pytest.approx(0.3)

    # Right cell should be unchanged (it's the brighter one)
    np.testing.assert_allclose(result[0, 1, :], 0.9)


def test_enhance_contrast_single_cell_unchanged():
    grid = np.full((1, 1, NUM_SAMPLES), 0.7)
    result = enhance_contrast(grid, exponent=3.0)
    np.testing.assert_allclose(result, grid)


def test_enhance_contrast_all_zeros_stays_zero():
    grid = np.zeros((3, 3, NUM_SAMPLES))
    result = enhance_contrast(grid, exponent=2.0)
    np.testing.assert_array_equal(result, 0.0)


def test_enhance_contrast_higher_exponent_more_contrast():
    grid = np.zeros((1, 2, NUM_SAMPLES))
    grid[0, 0, :] = 0.3
    grid[0, 1, :] = 0.9

    result_low = enhance_contrast(grid, exponent=2.0)
    result_high = enhance_contrast(grid, exponent=4.0)

    # Higher exponent should push border values even darker
    right_edge = [2, 5, 8, 11, 14]
    for idx in right_edge:
        assert result_high[0, 0, idx] < result_low[0, 0, idx]


def test_enhance_contrast_vertical_boundary():
    grid = np.zeros((2, 1, NUM_SAMPLES))
    grid[0, 0, :] = 0.8  # bright top
    grid[1, 0, :] = 0.2  # dim bottom
    result = enhance_contrast(grid, exponent=2.0)

    # Bottom cell's top-row positions should get darker
    top_row = [0, 1, 2]
    for idx in top_row:
        assert result[1, 0, idx] < 0.2

    # Bottom cell's interior positions should be unchanged
    interior = [4, 7, 10]
    for idx in interior:
        assert result[1, 0, idx] == pytest.approx(0.2)
