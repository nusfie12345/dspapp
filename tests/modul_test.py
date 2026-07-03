import numpy as np
import pytest

from dspeffects import Delay, Flanger, Chorus, Phaser
from utils import bpm_to_delay_seconds, feedback_from_echoes
from filters import FirstOrderAPF, DelayLine


def test_bpm_to_delay_seconds_quarter_note():
    assert bpm_to_delay_seconds(120, "1/4") == pytest.approx(0.5)


def test_bpm_to_delay_seconds_dotted_eighth():
    assert bpm_to_delay_seconds(120, "1/8d") == pytest.approx(0.375)


def test_bpm_to_delay_seconds_quarter_triplet():
    assert bpm_to_delay_seconds(120, "1/4t") == pytest.approx(1 / 3)


def test_feedback_from_echoes():
    fb = feedback_from_echoes(4, threshold=0.01)
    assert 0 < fb < 1
    assert fb ** 4 == pytest.approx(0.01)


def test_delay_ms_mode_runs():
    d = Delay(mix=5, feedback=5, delay_ms=250, mode="ms")
    d.enabled = True

    x = np.zeros(44100)
    x[0] = 1.0

    y = d.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_delay_sync_mode_runs():
    d = Delay(mix=5, feedback=5, mode="sync", bpm=120, division="1/8")
    d.enabled = True

    x = np.zeros(44100)
    x[0] = 1.0

    y = d.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_delay_echoes_override_feedback():
    d = Delay(mix=5, feedback=10, delay_ms=100, mode="ms", echoes=3)

    fb = d._feedback_gain()

    assert fb == pytest.approx(feedback_from_echoes(3))


def test_invalid_note_division_raises():
    with pytest.raises(ValueError):
        bpm_to_delay_seconds(120, "bad")


def test_invalid_delay_mode_raises():
    with pytest.raises(ValueError):
        Delay(mix=5, mode="bad")


def test_setting_echoes_requires_integer():
    d = Delay(mix=5)

    with pytest.raises(ValueError):
        d.set_param("echo", 2.5)

def test_delayline_impulse_integer_delay():
    fs = 1000
    dl = DelayLine(max_delay_seconds=1.0, fs=fs)

    out = []

    for n in range(20):
        x = 1.0 if n == 0 else 0.0
        y = dl.read(5)
        dl.write(x)
        out.append(y)

    out = np.array(out)

    assert out[5] == pytest.approx(1.0)


def test_delayline_fractional_delay_is_finite():
    fs = 1000
    dl = DelayLine(max_delay_seconds=1.0, fs=fs)

    for n in range(20):
        dl.write(float(n))

    y = dl.read(3.5)

    assert np.isfinite(y)


@pytest.mark.parametrize("cls,args", [
    (Flanger, dict(rate=5, depth=5, feedback=5, mix=5)),
    (Chorus, dict(rate=5, depth=5, feedback=5, mix=5)),
    (Delay, dict(delay=5, feedback=5, mix=5)),
])
def test_mod_delay_effects_bypass_returns_copy(cls, args):
    fx = cls(**args)
    x = np.random.default_rng(0).normal(size=1024)

    y = fx.process(x)

    np.testing.assert_allclose(y, x)
    assert y is not x


@pytest.mark.parametrize("cls,args", [
    (Flanger, dict(rate=5, depth=5, feedback=5, mix=5)),
    (Chorus, dict(rate=5, depth=5, feedback=5, mix=5)),
    (Delay, dict(delay=5, feedback=5, mix=5)),
])
def test_mod_delay_effects_run_enabled(cls, args):
    fx = cls(**args)
    fx.enabled = True

    x = np.random.default_rng(1).normal(scale=0.1, size=5000)
    y = fx.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_flanger_phase_is_persistent_across_blocks():
    fx1 = Flanger(rate=5, depth=5, feedback=2, mix=5)
    fx2 = Flanger(rate=5, depth=5, feedback=2, mix=5)

    fx1.enabled = True
    fx2.enabled = True

    x = np.random.default_rng(2).normal(scale=0.1, size=4000)

    y_full = fx1.process(x)

    y_split = np.concatenate([
        fx2.process(x[:2000]),
        fx2.process(x[2000:])
    ])

    np.testing.assert_allclose(y_full, y_split, atol=1e-12)


def test_delay_feedback_decay_does_not_explode():
    fx = Delay(delay=1, feedback=9, mix=10)
    fx.enabled = True

    x = np.zeros(44100)
    x[0] = 1.0

    y = fx.process(x)

    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 10.0

def test_first_order_apf_runs_and_is_finite():
    apf = FirstOrderAPF(0.5)

    x = np.random.default_rng(0).normal(size=1000)
    y = np.array([apf.process(s) for s in x])

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_first_order_apf_rejects_invalid_coefficients():
    with pytest.raises(ValueError):
        FirstOrderAPF(1.0)

    with pytest.raises(ValueError):
        FirstOrderAPF(-1.0)


def test_phaser_bypass_returns_copy_and_same_values():
    ph = Phaser(rate=5, depth=5, feedback=5, mix=5)

    x = np.random.default_rng(1).normal(size=1024)
    y = ph.process(x)

    np.testing.assert_allclose(y, x)
    assert y is not x


def test_phaser_toggle_changes_enabled_state():
    ph = Phaser(rate=5, depth=5, feedback=5, mix=5)

    assert ph.enabled is False

    state = ph.toggle()

    assert state is True
    assert ph.enabled is True


def test_phaser_runs_enabled():
    ph = Phaser(rate=5, depth=5, feedback=5, mix=5)
    ph.enabled = True

    x = np.random.default_rng(2).normal(scale=0.1, size=5000)
    y = ph.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("stages", [1, 2, 4, 6, 8, 12])
def test_phaser_stage_counts_run(stages):
    ph = Phaser(rate=5, depth=5, feedback=5, mix=5, stages=stages)
    ph.enabled = True

    x = np.random.default_rng(3).normal(scale=0.1, size=2048)
    y = ph.process(x)

    assert len(ph.stages) == stages
    assert np.all(np.isfinite(y))


def test_phaser_stage_count_can_be_changed():
    ph = Phaser(rate=5, depth=5, feedback=5, mix=5, stages=4)

    ph.set_param("stages", 8)

    assert ph.params["stages"] == 8
    assert len(ph.stages) == 8

    ph.set_param("stages", 2)

    assert ph.params["stages"] == 2
    assert len(ph.stages) == 2


def test_phaser_phase_persists_across_blocks():
    ph1 = Phaser(rate=5, depth=5, feedback=2, mix=5)
    ph2 = Phaser(rate=5, depth=5, feedback=2, mix=5)

    ph1.enabled = True
    ph2.enabled = True

    x = np.random.default_rng(4).normal(scale=0.1, size=4000)

    y_full = ph1.process(x)

    y_split = np.concatenate([
        ph2.process(x[:2000]),
        ph2.process(x[2000:])
    ])

    np.testing.assert_allclose(y_full, y_split, atol=1e-12)


def test_phaser_invalid_stage_count_raises():
    with pytest.raises(ValueError):
        Phaser(rate=5, depth=5, feedback=5, mix=5, stages=0)


def test_phaser_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        Phaser(rate=5, depth=5, feedback=5, mix=5, min_freq=1000, max_freq=500)


def test_phaser_mix_zero_is_close_to_input():
    ph = Phaser(rate=5, depth=5, feedback=0, mix=0)
    ph.enabled = True

    x = np.random.default_rng(5).normal(scale=0.1, size=2048)
    y = ph.process(x)

    np.testing.assert_allclose(y, x, atol=1e-12)