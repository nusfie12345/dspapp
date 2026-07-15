import numpy as np
import pytest

from audio.effects.dspeffects import Reverb, CabSim
from audio.effects.reverbs import (
    SchroederReverb,
    FreeVerb,
    ConvolutionReverb,
)


def test_schroeder_bypass_returns_copy():
    rvb = SchroederReverb()
    x = np.random.default_rng(0).normal(size=2048)

    y = rvb.process(x)

    np.testing.assert_allclose(y, x)
    assert y is not x


def test_schroeder_runs_enabled():
    rvb = SchroederReverb(room_size=5, damping=5, mix=5, level=5)
    rvb.enabled = True

    x = np.zeros(44100)
    x[0] = 1.0

    y = rvb.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) > 0


def test_freeverb_mono_input_returns_stereo():
    rvb = FreeVerb(room_size=5, damping=5, width=10, mix=5, level=5)
    rvb.enabled = True

    x = np.zeros(44100)
    x[0] = 1.0

    y = rvb.process(x)

    assert y.shape == (len(x), 2)
    assert np.all(np.isfinite(y))


def test_freeverb_stereo_input_returns_stereo():
    rvb = FreeVerb(room_size=5, damping=5, width=10, mix=5, level=5)
    rvb.enabled = True

    x = np.zeros((44100, 2))
    x[0, 0] = 1.0
    x[0, 1] = 1.0

    y = rvb.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_convolution_mono_ir_mono_input():
    ir = np.zeros(1000)
    ir[0] = 1.0
    ir[100] = 0.5

    rvb = ConvolutionReverb(ir=ir, mix=10, level=10)
    rvb.enabled = True

    x = np.zeros(1000)
    x[0] = 1.0

    y = rvb.process(x)

    assert y.shape == x.shape
    assert y[0] == pytest.approx(1.0)
    assert y[100] == pytest.approx(0.5)


def test_convolution_stereo_ir_mono_input_returns_stereo():
    ir = np.zeros((1000, 2))
    ir[0, 0] = 1.0
    ir[0, 1] = 1.0
    ir[100, 0] = 0.5
    ir[200, 1] = 0.5

    rvb = ConvolutionReverb(ir=ir, mix=10, level=10)
    rvb.enabled = True

    x = np.zeros(1000)
    x[0] = 1.0

    y = rvb.process(x)

    assert y.shape == (len(x), 2)
    assert y[100, 0] == pytest.approx(0.5)
    assert y[200, 1] == pytest.approx(0.5)


def test_reverb_wrapper_schroeder_runs():
    rvb = Reverb(mode="schroeder", room_size=5, damping=5, mix=5, level=5)
    rvb.enabled = True

    x = np.random.default_rng(1).normal(scale=0.05, size=2048)
    y = rvb.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_reverb_wrapper_convolution_requires_ir():
    with pytest.raises(ValueError):
        Reverb(mode="convolution")


def test_invalid_reverb_mode_raises():
    with pytest.raises(ValueError):
        Reverb(mode="bad")


def test_cabsim_bypass_returns_copy_and_same_values():
    cab = CabSim(mode="filter", cab="closed_4x12")
    x = np.random.default_rng(0).normal(size=1024)

    y = cab.process(x)

    np.testing.assert_allclose(y, x)
    assert y is not x


@pytest.mark.parametrize("preset", [
    "open_1x12",
    "closed_2x12",
    "closed_4x12",
    "bright_2x12",
    "dark_1x12",
])
def test_filter_cab_presets_run(preset):
    cab = CabSim(mode="filter", cab=preset)
    cab.enabled = True

    x = np.random.default_rng(1).normal(scale=0.1, size=4096)
    y = cab.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_filter_cab_stereo_runs():
    cab = CabSim(mode="filter", cab="closed_4x12")
    cab.enabled = True

    x = np.random.default_rng(2).normal(scale=0.1, size=(4096, 2))
    y = cab.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_cab_preset_change_changes_output():
    x = np.random.default_rng(3).normal(scale=0.1, size=4096)

    cab1 = CabSim(mode="filter", cab="closed_4x12")
    cab2 = CabSim(mode="filter", cab="open_1x12")

    cab1.enabled = True
    cab2.enabled = True

    y1 = cab1.process(x)
    y2 = cab2.process(x)

    assert not np.allclose(y1, y2)


def test_ir_cab_mono_ir_mono_input():
    ir = np.zeros(512)
    ir[0] = 1.0
    ir[100] = 0.5

    cab = CabSim(mode="ir", ir=ir, mix=10, level=10)
    cab.enabled = True

    x = np.zeros(1024)
    x[0] = 1.0

    y = cab.process(x)

    assert y.shape == x.shape
    assert y[0] == pytest.approx(1.0)
    assert y[100] == pytest.approx(0.5)


def test_ir_cab_stereo_ir_mono_input_returns_stereo():
    ir = np.zeros((512, 2))
    ir[0, 0] = 1.0
    ir[0, 1] = 1.0
    ir[100, 0] = 0.5
    ir[200, 1] = 0.5

    cab = CabSim(mode="ir", ir=ir, mix=10, level=10)
    cab.enabled = True

    x = np.zeros(1024)
    x[0] = 1.0

    y = cab.process(x)

    assert y.shape == (len(x), 2)
    assert y[100, 0] == pytest.approx(0.5)
    assert y[200, 1] == pytest.approx(0.5)


def test_ir_mode_requires_ir():
    with pytest.raises(ValueError):
        CabSim(mode="ir")


def test_invalid_cab_preset_raises():
    with pytest.raises(ValueError):
        CabSim(mode="filter", cab="bad_cab")


def test_cannot_switch_to_ir_without_ir():
    cab = CabSim(mode="filter", cab="closed_4x12")

    with pytest.raises(ValueError):
        cab.set_param("mode", "ir")