import numpy as np
import pytest
from scipy.signal import freqz, lfilter

from audio.misc.filters import Biquad, DelayRevAPF


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-30))


def biquad_arrays(f):
    b = np.array([f.b0, f.b1, f.b2])
    a = np.array([1.0, f.a1, f.a2])
    return b, a


def test_biquad_process_matches_lfilter():
    fs = 48_000
    f = Biquad(fs)
    b = np.array([0.2, 0.1, 0.05])
    a = np.array([1.0, -0.3, 0.12])
    f.set_coefs(b, a)

    x = np.random.default_rng(0).normal(size=1000)
    y_custom = np.array([f.process(sample) for sample in x])
    y_ref = lfilter(b, a, x)

    np.testing.assert_allclose(y_custom, y_ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method", ["LPF", "HPF"])
def test_pass_filters_have_finite_stable_coefficients(method):
    fs = 48_000
    f = Biquad(fs)
    getattr(f, method)(1_000)

    b, a = biquad_arrays(f)

    assert np.all(np.isfinite(b))
    assert np.all(np.isfinite(a))

    poles = np.roots(a)
    assert np.all(np.abs(poles) < 1)


def test_lpf_dc_gain_is_one_and_nyquist_is_small():
    fs = 48_000
    f = Biquad(fs)
    f.LPF(1_000)
    b, a = biquad_arrays(f)

    w, h = freqz(b, a, worN=[0, np.pi])

    assert np.isclose(abs(h[0]), 1.0, atol=1e-6)
    assert abs(h[1]) < 1e-6


def test_hpf_dc_is_small_and_nyquist_gain_is_one():
    fs = 48_000
    f = Biquad(fs)
    f.HPF(1_000)
    b, a = biquad_arrays(f)

    w, h = freqz(b, a, worN=[0, np.pi])

    assert abs(h[0]) < 1e-6
    assert np.isclose(abs(h[1]), 1.0, atol=1e-6)


@pytest.mark.parametrize("shelf_type", ["low", "high"])
def test_zero_gain_shelf_is_identity(shelf_type):
    fs = 48_000
    f = Biquad(fs)
    f.Shelf(1_000, gain=0, type=shelf_type)

    b, a = biquad_arrays(f)

    np.testing.assert_allclose(b, [1.0, f.a1, f.a2], atol=1e-12)
    np.testing.assert_allclose(a, [1.0, f.a1, f.a2], atol=1e-12)

    x = np.random.default_rng(1).normal(size=1000)
    y = np.array([f.process(sample) for sample in x])

    np.testing.assert_allclose(y, x, atol=1e-10)


@pytest.mark.parametrize("gain", [-18, -6, 6, 18])
@pytest.mark.parametrize("shelf_type", ["low", "high"])
def test_shelf_coefficients_are_finite_and_stable(gain, shelf_type):
    fs = 48_000
    f = Biquad(fs)
    f.Shelf(1_000, gain=gain, type=shelf_type)

    b, a = biquad_arrays(f)

    assert np.all(np.isfinite(b))
    assert np.all(np.isfinite(a))

    poles = np.roots(a)
    assert np.all(np.abs(poles) < 1)


def test_low_shelf_endpoint_gains():
    fs = 48_000
    gain = 6
    f = Biquad(fs)
    f.Shelf(1_000, gain=gain, type="low")
    b, a = biquad_arrays(f)

    w, h = freqz(b, a, worN=[0, np.pi])

    assert np.isclose(db(h[0]), gain, atol=0.05)
    assert np.isclose(db(h[1]), 0.0, atol=0.05)


def test_high_shelf_endpoint_gains():
    fs = 48_000
    gain = 6
    f = Biquad(fs)
    f.Shelf(1_000, gain=gain, type="high")
    b, a = biquad_arrays(f)

    w, h = freqz(b, a, worN=[0, np.pi])

    assert np.isclose(db(h[0]), 0.0, atol=0.05)
    assert np.isclose(db(h[1]), gain, atol=0.05)


@pytest.mark.parametrize("g", [-0.7, -0.3, 0.0, 0.3, 0.7])
def test_apf_has_flat_magnitude_response(g):
    delay = 37
    apf = DelayRevAPF(delay, g)

    n = 65536
    impulse = np.zeros(n)
    impulse[0] = 1.0

    y = np.array([apf.process(x) for x in impulse])

    H = np.fft.rfft(y, n * 2)
    mag = np.abs(H)

    assert np.max(np.abs(mag - 1.0)) < 1e-3


def test_apf_delay_and_gain_validation():
    with pytest.raises(ValueError):
        DelayRevAPF(0, 0.5)

    with pytest.raises(ValueError):
        DelayRevAPF(10, 1.0)

    with pytest.raises(ValueError):
        DelayRevAPF(10, -1.0)

