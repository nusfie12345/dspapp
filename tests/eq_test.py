import numpy as np
import pytest

from dspeffects import EQ


def test_eq_bypass_returns_copy_and_same_values():
    eq = EQ()
    x = np.random.default_rng(0).normal(size=1024)

    y = eq.process(x)

    np.testing.assert_allclose(y, x)
    assert y is not x


def test_eq_enabled_runs_parametric():
    eq = EQ(use_parametric=True, use_graphic=False)
    eq.enabled = True

    x = np.random.default_rng(1).normal(scale=0.1, size=4096)
    y = eq.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_eq_enabled_runs_graphic():
    eq = EQ(use_parametric=False, use_graphic=True)
    eq.enabled = True

    x = np.random.default_rng(2).normal(scale=0.1, size=4096)
    y = eq.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_eq_enabled_runs_both_modes():
    eq = EQ(use_parametric=True, use_graphic=True)
    eq.enabled = True

    x = np.random.default_rng(3).normal(scale=0.1, size=4096)
    y = eq.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_parametric_band_update_changes_output():
    x = np.random.default_rng(4).normal(scale=0.1, size=4096)

    eq1 = EQ(use_parametric=True, use_graphic=False)
    eq2 = EQ(use_parametric=True, use_graphic=False)

    eq1.enabled = True
    eq2.enabled = True

    eq2.set_parametric_band(2, type="peak", fc=1000, gain=12, Q=1.0)

    y1 = eq1.process(x)
    y2 = eq2.process(x)

    assert not np.allclose(y1, y2)


def test_graphic_band_update_changes_output():
    x = np.random.default_rng(5).normal(scale=0.1, size=4096)

    eq1 = EQ(use_parametric=False, use_graphic=True)
    eq2 = EQ(use_parametric=False, use_graphic=True)

    eq1.enabled = True
    eq2.enabled = True

    eq2.set_graphic_band(3, 12)

    y1 = eq1.process(x)
    y2 = eq2.process(x)

    assert not np.allclose(y1, y2)


def test_invalid_parametric_band_type_raises():
    eq = EQ()

    with pytest.raises(ValueError):
        eq.set_parametric_band(0, type="bad")


def test_invalid_graphic_gain_raises():
    eq = EQ()

    with pytest.raises(ValueError):
        eq.set_graphic_band(0, 99)


def test_toggle_parametric_and_graphic_modes():
    eq = EQ(use_parametric=True, use_graphic=False)

    assert eq.params["use_parametric"] is True
    assert eq.params["use_graphic"] is False

    eq.enable_parametric(False)
    eq.enable_graphic(True)

    assert eq.params["use_parametric"] is False
    assert eq.params["use_graphic"] is True