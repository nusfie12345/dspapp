# imports

import numpy as np

#______________________

def normalize(param):
    """
    A function to properly normalize all the numeric inputs to the scale [0, 1].

    Args:
        param: designed parameter for the effect to normalize.
    """
    if not isinstance(param, (int, float)):
        raise ValueError(f"Parameter {param} must be numeric.")

    param = max(0, min(10, param))

    return param / 10

def volume_gain(param):
    """
    A scale function reserved specifically for the "volume" parameter. 
    
    Remaps the makeup(pre-master) volume from [0, 1] to [-20, +6] dB.

    Args:
        param: volume value
    """
    if not 0 <= param <= 1:
        raise ValueError("Volume parameter must be between 0 and 1.")
    
    db = -20 + 26 * param
    return 10**(db/20)

def drive_eff(param, min_val, max_val):
    """
    A drive effect rescale designed for OD, distortion, and fuzz.

    Args:
        param: initial drive value
        min_val: minimum value for the effect
        max_val: maximum value for the effect
    """
    if not isinstance(param, (int, float)):
        raise ValueError("Drive parameter must be numeric.")

    if not 0 <= param <= 1:
        raise ValueError("Drive parameter must be between 0 and 1.")

    if min_val <= 0:
        raise ValueError("min_val must be > 0.")

    if max_val <= min_val:
        raise ValueError("max_val must exceed min_val.")

    return min_val * ((max_val/min_val)**param)

NOTE_DIVISIONS = {
    "1/1": 4.0,
    "1/2": 2.0,
    "1/4": 1.0,
    "1/8": 0.5,
    "1/16": 0.25,
    "1/32": 0.125,

    "1/2d": 3.0,
    "1/4d": 1.5,
    "1/8d": 0.75,
    "1/16d": 0.375,

    "1/2t": 4.0 / 3.0,
    "1/4t": 2.0 / 3.0,
    "1/8t": 1.0 / 3.0,
    "1/16t": 1.0 / 6.0,
}


def bpm_to_delay_seconds(bpm, division):
    if bpm <= 0:
        raise ValueError("bpm must be positive")

    if division not in NOTE_DIVISIONS:
        raise ValueError(f"Invalid note division: {division}")

    return (60.0 / bpm) * NOTE_DIVISIONS[division]


def feedback_from_echoes(echoes, threshold=0.01, max_feedback=0.95):
    if echoes <= 0:
        return 0.0

    fb = threshold ** (1.0 / echoes)
    return min(fb, max_feedback)

CABSIM_PRESETS = {
    "open_1x12": {
        "hpf": 90,
        "lpf": 6500,
        "peaks": [
            (130, 1.5, 0.8),
            (350, -2.0, 1.0),
            (1200, 2.0, 1.0),
            (3200, -2.5, 1.2),
        ],
    },

    "closed_2x12": {
        "hpf": 80,
        "lpf": 5800,
        "peaks": [
            (110, 2.0, 0.8),
            (300, -2.5, 1.0),
            (900, 2.5, 1.0),
            (2500, -1.5, 1.3),
            (4200, -3.0, 1.0),
        ],
    },

    "closed_4x12": {
        "hpf": 75,
        "lpf": 5200,
        "peaks": [
            (100, 2.5, 0.8),
            (240, -2.0, 1.1),
            (750, 2.0, 1.0),
            (1600, 2.5, 1.0),
            (3600, -3.5, 1.2),
        ],
    },

    "bright_2x12": {
        "hpf": 75,
        "lpf": 7200,
        "peaks": [
            (120, 1.5, 0.9),
            (500, -1.5, 1.0),
            (1800, 2.5, 1.0),
            (4300, 1.5, 1.0),
        ],
    },

    "dark_1x12": {
        "hpf": 110,
        "lpf": 4300,
        "peaks": [
            (150, 2.0, 0.8),
            (450, -2.0, 1.0),
            (1000, 1.5, 1.1),
            (2800, -3.0, 1.2),
        ],
    },
}

import numpy as np


class StreamingFIR:
    def __init__(self, h):
        h = np.asarray(h, dtype=np.float32)

        if h.ndim != 1:
            raise ValueError("FIR coefficients must be 1-D")

        if len(h) < 2:
            raise ValueError("FIR filter must have at least 2 taps")

        self.h = h
        self.state = np.zeros(len(h) - 1, dtype=np.float32)

    def reset(self):
        self.state.fill(0.0)

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 1:
            raise ValueError(f"StreamingFIR expects 1-D input, got {x.shape}")

        x_ext = np.concatenate([self.state, x])

        # 'valid' gives exactly len(x) samples because state has len(h)-1 samples.
        y = np.convolve(x_ext, self.h, mode="valid").astype(np.float32)

        self.state = x_ext[-(len(self.h) - 1):].astype(np.float32)

        return y

def design_lowpass_fir(cutoff=0.225, taps=63):
    """
    Windowed-sinc low-pass FIR.

    cutoff is in cycles/sample, where Nyquist = 0.5.
    For 2x oversampling, 0.225 is a practical transition below 0.25.
    """

    if not 0.0 < cutoff < 0.5:
        raise ValueError("cutoff must be between 0 and 0.5")

    if taps % 2 == 0:
        raise ValueError("taps should be odd for linear-phase symmetry")

    n = np.arange(taps, dtype=np.float32)
    m = (taps - 1) / 2.0

    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * (n - m))

    # Hamming window.
    w = np.hamming(taps).astype(np.float32)
    h *= w

    # Unity DC gain.
    h /= np.sum(h)

    return h.astype(np.float32)

class TwoXOversamplingStage:
    def __init__(self, taps=63, cutoff=0.225):
        h = design_lowpass_fir(cutoff=cutoff, taps=taps)

        # Interpolation filter needs gain 2 because zero-stuffing halves amplitude.
        self.up_filter = StreamingFIR(2.0 * h)

        # Decimation anti-alias filter should have unity DC gain.
        self.down_filter = StreamingFIR(h)

    def reset(self):
        self.up_filter.reset()
        self.down_filter.reset()

    def upsample(self, x):
        x = np.asarray(x, dtype=np.float32)

        z = np.zeros(len(x) * 2, dtype=np.float32)
        z[::2] = x

        return self.up_filter.process(z)

    def downsample(self, x):
        x = np.asarray(x, dtype=np.float32)

        filtered = self.down_filter.process(x)

        # Input length should normally be even, so output length is len(x)//2.
        return filtered[::2].astype(np.float32)

class Oversampler:
    def __init__(self, factor=2, taps=63, cutoff=0.225):
        if factor not in (1, 2, 4, 8):
            raise ValueError("factor must be 1, 2, 4, or 8")

        self.factor = factor

        stages_count = {
            1: 0,
            2: 1,
            4: 2,
            8: 3,
        }[factor]

        self.stages = [
            TwoXOversamplingStage(taps=taps, cutoff=cutoff)
            for _ in range(stages_count)
        ]

    def reset(self):
        for stage in self.stages:
            stage.reset()

    def upsample(self, x):
        y = np.asarray(x, dtype=np.float32)

        for stage in self.stages:
            y = stage.upsample(y)

        return y

    def downsample(self, x):
        y = np.asarray(x, dtype=np.float32)

        for stage in reversed(self.stages):
            y = stage.downsample(y)

        return y

    def process(self, x, shape_func):
        """
        Oversample x, apply shape_func at high sample rate,
        then downsample back.

        shape_func must accept and return a 1-D np.ndarray.
        """

        x = np.asarray(x, dtype=np.float32)

        if self.factor == 1:
            return shape_func(x).astype(np.float32)

        up = self.upsample(x)

        shaped = shape_func(up).astype(np.float32)

        down = self.downsample(shaped).astype(np.float32)

        # This implementation should naturally return len(x), but guard anyway.
        if len(down) > len(x):
            down = down[:len(x)]

        if len(down) < len(x):
            down = np.pad(down, (0, len(x) - len(down)))

        return down.astype(np.float32)
    
