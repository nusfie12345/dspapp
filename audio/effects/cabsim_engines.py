import numpy as np
from scipy.signal import fftconvolve

import audio.misc.filters as flt
import audio.misc.utils as uts


class FilterCabEngine:
    real_time_safe = True

    def __init__(self, fs=44100, cab="closed_4x12"):
        if cab not in uts.CABSIM_PRESETS:
            raise ValueError(f"Unknown cab preset: {cab}")

        self.fs = fs
        self.cab = cab

        self.filter_l = flt.CabFiltChain(fs, uts.CABSIM_PRESETS[cab])
        self.filter_r = flt.CabFiltChain(fs, uts.CABSIM_PRESETS[cab])

    def reset(self):
        self.filter_l.reset()
        self.filter_r.reset()

    def set_cab(self, cab):
        if cab not in uts.CABSIM_PRESETS:
            raise ValueError(f"Unknown cab preset: {cab}")

        self.cab = cab
        preset = uts.CABSIM_PRESETS[cab]

        self.filter_l.configure(preset)
        self.filter_r.configure(preset)

    def process_sample(self, x):
        return self.filter_l.process(x)

    def process_block(self, x):
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            y = self.filter_l.process(x)

            return y

        if x.ndim == 2 and x.shape[1] == 2:
            y = np.zeros_like(x)

            y[:, 0] = self.filter_l.process(x[:, 0])
            y[:, 1] = self.filter_r.process(x[:, 1])

            return y

        raise ValueError("Input must be mono shape (n,) or stereo shape (n, 2)")


class IRCabEngine:
    real_time_safe = False

    def __init__(self, ir, normalize_ir=True, fs=44100):
        self.fs = fs
        self.ir = None
        self.set_ir(ir, normalize_ir=normalize_ir)

    def reset(self):
        # fftconvolve version is stateless across blocks.
        pass

    def set_ir(self, ir, normalize_ir=True):
        ir = np.asarray(ir, dtype=float)

        if ir.ndim not in (1, 2):
            raise ValueError("IR must be mono shape (n,) or stereo shape (n, 2)")

        if ir.ndim == 2 and ir.shape[1] != 2:
            raise ValueError("Stereo IR must have shape (n, 2)")

        if normalize_ir:
            peak = np.max(np.abs(ir))
            if peak > 0:
                ir = ir / peak

        self.ir = ir

    def process_sample(self, x):
        raise RuntimeError("IR cab sim is block-based, not sample-based")

    def process_block(self, x):
        x = np.asarray(x, dtype=float)
        ir = self.ir

        if ir is None:
            raise ValueError("No IR loaded")

        # Mono input + mono IR → mono output
        if x.ndim == 1 and ir.ndim == 1:
            return fftconvolve(x, ir, mode="full")[:len(x)]

        # Mono input + stereo IR → stereo output
        if x.ndim == 1 and ir.ndim == 2:
            wet_l = fftconvolve(x, ir[:, 0], mode="full")[:len(x)]
            wet_r = fftconvolve(x, ir[:, 1], mode="full")[:len(x)]

            y = np.zeros((len(x), 2), dtype=float)
            y[:, 0] = wet_l
            y[:, 1] = wet_r

            return y

        # Stereo input + mono IR → stereo output
        if x.ndim == 2 and x.shape[1] == 2 and ir.ndim == 1:
            y = np.zeros_like(x)

            y[:, 0] = fftconvolve(x[:, 0], ir, mode="full")[:len(x)]
            y[:, 1] = fftconvolve(x[:, 1], ir, mode="full")[:len(x)]

            return y

        # Stereo input + stereo IR → channel-wise stereo output
        if x.ndim == 2 and x.shape[1] == 2 and ir.ndim == 2:
            y = np.zeros_like(x)

            y[:, 0] = fftconvolve(x[:, 0], ir[:, 0], mode="full")[:len(x)]
            y[:, 1] = fftconvolve(x[:, 1], ir[:, 1], mode="full")[:len(x)]

            return y

        raise ValueError("Unsupported input/IR shape combination")