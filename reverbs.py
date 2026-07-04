import numpy as np
import utils as uts
import filters as flt

from scipy.signal import fftconvolve

# Need a separate file for reverbs. Holy sh*t, there's so many of them...


class SchroederReverb:
    r"""
    Dubbed the most 'theoretical' reverb algorithm. (We're gonna use it anyways lmao)

    Equation:

     // ah who am i kidding? i'll add it later. hope i won't forget.

    NB!!! RemindMe! remake the description
    """

    def __init__(self, room_size=5, damping=4, mix=3, level=5, fs=44100):
        real_time_safe = True
        self.params = {
            "room_size": uts.normalize(room_size),
            "damping": uts.normalize(damping),
            "mix": uts.normalize(mix),
            "level": uts.normalize(level),
        }

        self.enabled = False

        # Prime-ish / non-overlapping delay times help reduce metallic ringing.
        self.comb_times = [0.0297, 0.0371, 0.0411, 0.0437]
        self.apf_times = [0.0050, 0.0017]

        self.combs = [flt.FeedbackComb(t, fs=fs) for t in self.comb_times]
        self.apfs = [flt.DelayLine(t, g=0.5, fs=fs) for t in self.apf_times]

        self.upd_param()

    def reset(self):
        for comb in self.combs:
            comb.reset()
        for apf in self.apfs:
            apf.reset()

    def upd_param(self):
        # room_size 0..1 → feedback 0.55..0.88
        feedback = 0.55 + 0.33 * self.params["room_size"]

        # damping 0..1; higher means darker / more damped.
        damping = 0.05 + 0.75 * self.params["damping"]

        for comb in self.combs:
            comb.set_params(feedback=feedback, damping=damping)

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")

        if 0 <= val <= 10:
            val = uts.normalize(val)
        else:
            raise ValueError(f"{name} must be in range [0, 10]")

        self.params[name] = val

        if name in ("room_size", "damping"):
            self.upd_param()

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def algo(self, x):
        wet = 0.0

        for comb in self.combs:
            wet += comb.process(x)

        wet *= 1.0 / len(self.combs)

        for apf in self.apfs:
            wet = apf.process(wet)

        mix = self.params["mix"]
        level = uts.volume_gain(self.params["level"])

        return level * ((1.0 - mix) * x + mix * wet)

    def process(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        y = np.zeros_like(x)

        for n, sample in enumerate(x):
            y[n] = self.algo(sample)

        return y

class FreeVerb:
    def __init__(self, room_size, damping, width, mix, level, fs=44100):
        real_time_safe = True
        self.params = {
            "room_size": uts.normalize(room_size),
            "damping": uts.normalize(damping),
            "width": uts.normalize(width),
            "mix": uts.normalize(mix),
            "level": uts.normalize(level)
        }

        self.enabled = False

        # Freeverb-ish delay lengths at 44.1k converted to seconds.
        # Left channel tunings.
        comb_samples_l = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
        apf_samples_l = [556, 441, 341, 225]

        # Stereo spread. Original Freeverb uses +23 samples for right.
        stereo_spread = 23
        comb_samples_r = [s + stereo_spread for s in comb_samples_l]
        apf_samples_r = [s + stereo_spread for s in apf_samples_l]

        self.combs_l = [flt.FeedbackComb(s / 44100.0, fs=fs) for s in comb_samples_l]
        self.combs_r = [flt.FeedbackComb(s / 44100.0, fs=fs) for s in comb_samples_r]

        self.apfs_l = [flt.DelayRevAPF(s / 44100.0, g=0.5, fs=fs) for s in apf_samples_l]
        self.apfs_r = [flt.DelayRevAPF(s / 44100.0, g=0.5, fs=fs) for s in apf_samples_r]

        self.upd_param()

    def reset(self):
        for bank in (self.combs_l, self.combs_r, self.apfs_l, self.apfs_r):
            for f in bank:
                f.reset()

    def upd_param(self):
        # Freeverb-style ranges.
        feedback = 0.7 + 0.28 * self.params["room_size"]
        damping = 0.05 + 0.75 * self.params["damping"]

        feedback = min(feedback, 0.98)

        for comb in self.combs_l + self.combs_r:
            comb.set_params(feedback=feedback, damping=damping)

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")

        if 0 <= val <= 10:
            val = uts.normalize(val)
        else:
            raise ValueError(f"{name} must be in range [0, 10]")

        self.params[name] = val

        if name in ("room_size", "damping"):
            self.upd_param()

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def _process_mono_to_stereo_sample(self, x):
        wet_l = 0.0
        wet_r = 0.0

        for comb in self.combs_l:
            wet_l += comb.process(x)

        for comb in self.combs_r:
            wet_r += comb.process(x)

        wet_l *= 1.0 / len(self.combs_l)
        wet_r *= 1.0 / len(self.combs_r)

        for apf in self.apfs_l:
            wet_l = apf.process(wet_l)

        for apf in self.apfs_r:
            wet_r = apf.process(wet_r)

        width = self.params["width"]

        wet_mid = 0.5 * (wet_l + wet_r)
        wet_l = wet_mid * (1.0 - width) + wet_l * width
        wet_r = wet_mid * (1.0 - width) + wet_r * width

        return wet_l, wet_r

    def process(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        mix = self.params["mix"]
        level = uts.volume_gain(self.params["level"])

        # Mono input: return stereo output.
        if x.ndim == 1:
            y = np.zeros((len(x), 2), dtype=float)

            for n, sample in enumerate(x):
                wet_l, wet_r = self._process_mono_to_stereo_sample(sample)

                y[n, 0] = level * ((1.0 - mix) * sample + mix * wet_l)
                y[n, 1] = level * ((1.0 - mix) * sample + mix * wet_r)

            return y

        # Stereo input.
        if x.ndim == 2 and x.shape[1] == 2:
            y = np.zeros_like(x)

            for n in range(len(x)):
                mono_in = 0.5 * (x[n, 0] + x[n, 1])
                wet_l, wet_r = self._process_mono_to_stereo_sample(mono_in)

                y[n, 0] = level * ((1.0 - mix) * x[n, 0] + mix * wet_l)
                y[n, 1] = level * ((1.0 - mix) * x[n, 1] + mix * wet_r)

            return y

        raise ValueError("Input must be mono shape (n,) or stereo shape (n, 2)")
    
class ConvolutionReverb:
    def __init__(self, ir, mix=5, level=5, normalize_ir=True, fs=44100):
        real_time_safe = False
        self.params = {
            "mix": uts.normalize(mix),
            "level": uts.normalize(level),
        }

        self.enabled = False
        self.ir = self._prepare_ir(ir, normalize_ir)

    def _prepare_ir(self, ir, normalize):
        ir = np.asarray(ir, dtype=float)

        if ir.ndim not in (1, 2):
            raise ValueError("IR must be mono shape (n,) or stereo shape (n, 2)")

        if ir.ndim == 2 and ir.shape[1] != 2:
            raise ValueError("Stereo IR must have shape (n, 2)")

        if normalize:
            peak = np.max(np.abs(ir))
            if peak > 0:
                ir = ir / peak

        return ir

    def set_ir(self, ir, normalize_ir=True):
        self.ir = self._prepare_ir(ir, normalize_ir)

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")

        if 0 <= val <= 10:
            val = uts.normalize(val)
        else:
            raise ValueError(f"{name} must be in range [0, 10]")

        self.params[name] = val

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def process(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        mix = self.params["mix"]
        level = uts.volume_gain(self.params["level"])

        # Mono input, mono IR → mono output.
        if x.ndim == 1 and self.ir.ndim == 1:
            wet = fftconvolve(x, self.ir, mode="full")[:len(x)]
            return level * ((1.0 - mix) * x + mix * wet)

        # Mono input, stereo IR → stereo output.
        if x.ndim == 1 and self.ir.ndim == 2:
            wet_l = fftconvolve(x, self.ir[:, 0], mode="full")[:len(x)]
            wet_r = fftconvolve(x, self.ir[:, 1], mode="full")[:len(x)]

            y = np.zeros((len(x), 2), dtype=float)
            y[:, 0] = level * ((1.0 - mix) * x + mix * wet_l)
            y[:, 1] = level * ((1.0 - mix) * x + mix * wet_r)

            return y

        # Stereo input, mono IR → stereo output.
        if x.ndim == 2 and x.shape[1] == 2 and self.ir.ndim == 1:
            y = np.zeros_like(x)

            wet_l = fftconvolve(x[:, 0], self.ir, mode="full")[:len(x)]
            wet_r = fftconvolve(x[:, 1], self.ir, mode="full")[:len(x)]

            y[:, 0] = level * ((1.0 - mix) * x[:, 0] + mix * wet_l)
            y[:, 1] = level * ((1.0 - mix) * x[:, 1] + mix * wet_r)

            return y

        # Stereo input, stereo IR → channel-wise stereo output.
        if x.ndim == 2 and x.shape[1] == 2 and self.ir.ndim == 2:
            y = np.zeros_like(x)

            wet_l = fftconvolve(x[:, 0], self.ir[:, 0], mode="full")[:len(x)]
            wet_r = fftconvolve(x[:, 1], self.ir[:, 1], mode="full")[:len(x)]

            y[:, 0] = level * ((1.0 - mix) * x[:, 0] + mix * wet_l)
            y[:, 1] = level * ((1.0 - mix) * x[:, 1] + mix * wet_r)

            return y

        raise ValueError("Unsupported input/IR shape combination")