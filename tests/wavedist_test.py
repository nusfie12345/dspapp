import numpy as np
import pytest

from audio.effects.dspeffects import Overdrive, Distortion, Fuzz, BlockAmp


@pytest.mark.parametrize("cls,args", [
    (Overdrive, dict(drive=5, tone=5, level=5)),
    (Distortion, dict(dist=5, tone=5, level=5)),
    (Fuzz, dict(sustain=5, tone=5, level=5)),
])
def test_bypass_returns_copy_and_same_values(cls, args):
    fx = cls(**args)
    x = np.random.default_rng(0).normal(size=1024)

    y = fx.process(x)

    np.testing.assert_allclose(y, x)
    assert y is not x


def test_overdrive_modes_and_clip_setters():
    od = Overdrive(5, 5, 5)

    od.set_param("mode", "quad")
    assert od.params["mode"] == "quad"

    od.set_param("clip", "atan")
    assert od.params["clip"] == "atan"

    with pytest.raises(ValueError):
        od.set_param("mode", "bad")

    with pytest.raises(ValueError):
        od.set_param("clip", "bad")


def test_distortion_filter_and_mode_setters():
    d = Distortion(5, 5, 5)

    d.set_param("filter", "seesaw")
    assert d.params["filter"] == "seesaw"

    d.set_param("mode", "ord5")
    assert d.params["mode"] == "ord5"

    with pytest.raises(ValueError):
        d.set_param("filter", "bad")

    with pytest.raises(ValueError):
        d.set_param("mode", "bad")


def test_fuzz_asym_setter():
    fz = Fuzz(5, 5, 5)

    fz.set_param("asym", False)
    assert fz.params["asym"] is False

    with pytest.raises(ValueError):
        fz.set_param("asym", 1)


@pytest.mark.parametrize("cls,args", [
    (Overdrive, dict(drive=5, tone=5, level=5)),
    (Distortion, dict(dist=5, tone=5, level=5)),
    (Fuzz, dict(sustain=5, tone=5, level=5)),
])
def test_enabled_effect_runs_without_error(cls, args):
    fx = cls(**args)
    fx.enabled = True

    x = np.random.default_rng(1).normal(scale=0.05, size=5000)
    y = fx.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("mode", ["clear", "quad", "qcubic"])
@pytest.mark.parametrize("clip", ["atan", "tanh"])
def test_overdrive_all_modes_run(mode, clip):
    od = Overdrive(7, 5, 5, mode=mode, clip=clip)
    od.enabled = True

    x = np.random.default_rng(2).normal(scale=0.05, size=2048)
    y = od.process(x)

    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("mode", ["standard", "cubic", "ord5"])
@pytest.mark.parametrize("filter_type", ["classic", "seesaw"])
def test_distortion_all_modes_run(mode, filter_type):
    d = Distortion(7, 5, 5, mode=mode, filter=filter_type)
    d.enabled = True

    x = np.random.default_rng(3).normal(scale=0.05, size=2048)
    y = d.process(x)

    assert np.all(np.isfinite(y))


def test_distortion_polynomial_modes_are_bounded():
    d = Distortion(10, 5, 10)

    for mode in ("cubic", "ord5"):
        d.set_param("mode", mode)
        values = np.array([d._hardclip(x) for x in np.linspace(-1, 1, 1000)])
        assert np.max(np.abs(values)) <= 1.0 + 1e-9


def test_fuzz_tone_filters_initialized():
    fz = Fuzz(5, 5, 5)
    fz.enabled = True

    x = np.random.default_rng(4).normal(scale=0.05, size=1024)
    y = fz.process(x)

    assert np.all(np.isfinite(y))

def test_blockamp_bypass_returns_copy_and_same_values():
    amp = BlockAmp(5, 5, 5, 5, 5, 5, 5)
    x = np.random.default_rng(0).normal(size=1024)

    y = amp.process(x)

    np.testing.assert_allclose(y, x)
    assert y is not x


def test_blockamp_toggle_changes_enabled_state():
    amp = BlockAmp(5, 5, 5, 5, 5, 5, 5)

    assert amp.enabled is False

    state = amp.toggle()

    assert state is True
    assert amp.enabled is True


@pytest.mark.parametrize("clip_type", ["atan", "tanh", "hard", "null"])
def test_blockamp_clip_types_run(clip_type):
    amp = BlockAmp(5, 5, 5, 5, 5, 5, 5, clipType=clip_type)
    amp.enabled = True

    x = np.random.default_rng(1).normal(scale=0.05, size=4096)
    y = amp.process(x)

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_blockamp_set_clip_type():
    amp = BlockAmp(5, 5, 5, 5, 5, 5, 5)

    amp.set_param("clipType", "hard")
    assert amp.params["clipType"] == "hard"

    with pytest.raises(ValueError):
        amp.set_param("clipType", "bad")


@pytest.mark.parametrize("param", ["drive", "tone", "bass", "mid", "treb", "pres", "vol"])
def test_blockamp_numeric_params_scale(param):
    amp = BlockAmp(5, 5, 5, 5, 5, 5, 5)

    amp.set_param(param, 10)
    assert amp.params[param] == 1.0

    amp.set_param(param, 0)
    assert amp.params[param] == 0.0


def test_invalid_param_name_raises():
    amp = BlockAmp(5, 5, 5, 5, 5, 5, 5)

    with pytest.raises(ValueError):
        amp.set_param("bad_param", 5)


def test_blockamp_neutral_eq_does_not_explode():
    amp = BlockAmp(3, 5, 5, 5, 5, 5, 5, clipType="null")
    amp.enabled = True

    x = np.random.default_rng(2).normal(scale=0.05, size=4096)
    y = amp.process(x)

    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 10 * np.max(np.abs(x))


def test_drive_changes_output_when_clipping_enabled():
    x = np.random.default_rng(3).normal(scale=0.05, size=4096)

    low = BlockAmp(1, 5, 5, 5, 5, 5, 10, clipType="tanh")
    high = BlockAmp(10, 5, 5, 5, 5, 5, 10, clipType="tanh")

    low.enabled = True
    high.enabled = True

    y_low = low.process(x)
    y_high = high.process(x)

    assert not np.allclose(y_low, y_high)