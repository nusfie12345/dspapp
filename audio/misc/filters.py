import numpy as np
from scipy.signal import lfilter

class Biquad:
    """
    A class of IIR(infinite impulse response) filters. The general formula is 
    y[n] = b_0x[n] + b_1x[n-1] + b_2x[n-2] - a_1y[n-1] - a_2y[n-2]. 
    
    One of the most used frequency filter class in DSP. Includes low- and high-pass filters, as well as low- and high-shelf filters. 
    """
    def __init__(self, fs=44100):
        self.fs = fs

        # Identity filter by default.
        self.b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.a = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # lfilter / transposed DF-II state.
        self.zi = np.zeros(2, dtype=np.float32)

        # Backward-compatible coefficient attributes.
        self.b0 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0

    def reset(self):
        self.zi.fill(0.0)

    def set_coefs(self, b, a, reset_state=False):
        b = np.asarray(b, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)

        if b.shape != (3,):
            raise ValueError("b must have shape (3,)")

        if a.shape != (3,):
            raise ValueError("a must have shape (3,)")

        if abs(a[0]) < 1e-12:
            raise ValueError("a[0] cannot be zero")

        # Normalize so a[0] == 1.
        if a[0] != 1.0:
            b = b / a[0]
            a = a / a[0]

        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)):
            raise ValueError("Biquad coefficients produced NaN/Inf")

        self.b = b.astype(np.float32)
        self.a = a.astype(np.float32)

        self.b0 = float(self.b[0])
        self.b1 = float(self.b[1])
        self.b2 = float(self.b[2])
        self.a1 = float(self.a[1])
        self.a2 = float(self.a[2])

        if reset_state:
            self.reset()

    def process(self, x):
        """
        Scalar sample processing.
        Uses the same transposed DF-II state convention as scipy.signal.lfilter.
        """
        x = float(x)

        y = self.b0 * x + self.zi[0]

        z0 = self.b1 * x - self.a1 * y + self.zi[1]
        z1 = self.b2 * x - self.a2 * y

        self.zi[0] = z0
        self.zi[1] = z1

        return float(y)

    def process_block(self, x):
        """
        Fast block processing using scipy.signal.lfilter.
        """
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 1:
            raise ValueError(f"Biquad expects 1-D input, got {x.shape}")

        y, self.zi = lfilter(
            self.b,
            self.a,
            x,
            zi=self.zi,
        )

        return y.astype(np.float32)

    def LPF(self, fc, Q=0.707):
        if not 0 < fc < self.fs / 2:
            raise ValueError("'fc' must be between 0 and Nyquist")
        if Q <= 0:
            raise ValueError("'Q' must be positive")

        w0 = 2.0 * np.pi * fc / self.fs
        cosw = np.cos(w0)
        alpha = np.sin(w0) / (2.0 * Q)

        b0 = (1.0 - cosw) / 2.0
        b1 = 1.0 - cosw
        b2 = (1.0 - cosw) / 2.0

        a0 = 1.0 + alpha
        a1 = -2.0 * cosw
        a2 = 1.0 - alpha

        b = np.array([b0, b1, b2], dtype=np.float32)
        a = np.array([a0, a1, a2], dtype=np.float32)

        self.set_coefs(b, a)

    def HPF(self, fc, Q=0.707):
        if not 0 < fc < self.fs / 2:
            raise ValueError("'fc' must be between 0 and Nyquist")
        if Q <= 0:
            raise ValueError("'Q' must be positive")

        w0 = 2.0 * np.pi * fc / self.fs
        cosw = np.cos(w0)
        alpha = np.sin(w0) / (2.0 * Q)

        b0 = (1.0 + cosw) / 2.0
        b1 = -(1.0 + cosw)
        b2 = (1.0 + cosw) / 2.0

        a0 = 1.0 + alpha
        a1 = -2.0 * cosw
        a2 = 1.0 - alpha

        b = np.array([b0, b1, b2], dtype=np.float32)
        a = np.array([a0, a1, a2], dtype=np.float32)

        self.set_coefs(b, a)

    def Peak(self, fc, gain, Q=1.0):
        if not 0 < fc < self.fs / 2:
            raise ValueError("'fc' must be between 0 and Nyquist")
        if Q <= 0:
            raise ValueError("'Q' must be positive")

        A = 10.0 ** (gain / 40.0)
        w0 = 2.0 * np.pi * fc / self.fs
        cosw = np.cos(w0)
        alpha = np.sin(w0) / (2.0 * Q)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * cosw
        b2 = 1.0 - alpha * A

        a0 = 1.0 + alpha / A
        a1 = -2.0 * cosw
        a2 = 1.0 - alpha / A

        b = np.array([b0, b1, b2], dtype=np.float32)
        a = np.array([a0, a1, a2], dtype=np.float32)

        self.set_coefs(b, a)

    def Shelf(self, fc, gain, flt_type="low", s=1.0, **kwargs):
        """
        Low/high shelf filter.
        """
        if "type" in kwargs:
            flt_type = kwargs.pop("type")

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        if not 0 < fc < self.fs / 2:
            raise ValueError("'fc' must be between 0 and Nyquist")

        if s <= 0:
            raise ValueError("'s' must be positive")

        if flt_type not in ("low", "high"):
            raise ValueError("'flt_type' must be either 'low' or 'high'")

        A = 10.0 ** (gain / 40.0)
        w0 = 2.0 * np.pi * fc / self.fs
        cosw = np.cos(w0)

        alpha = np.sin(w0) / 2.0 * np.sqrt(
            (A + 1.0 / A) * (1.0 / s - 1.0) + 2.0
        )

        two_sqrt_A_alpha = 2.0 * np.sqrt(A) * alpha

        if flt_type == "low":
            b0 = A * ((A + 1.0) - (A - 1.0) * cosw + two_sqrt_A_alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosw)
            b2 = A * ((A + 1.0) - (A - 1.0) * cosw - two_sqrt_A_alpha)

            a0 = (A + 1.0) + (A - 1.0) * cosw + two_sqrt_A_alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosw)
            a2 = (A + 1.0) + (A - 1.0) * cosw - two_sqrt_A_alpha

        else:
            b0 = A * ((A + 1.0) + (A - 1.0) * cosw + two_sqrt_A_alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw)
            b2 = A * ((A + 1.0) + (A - 1.0) * cosw - two_sqrt_A_alpha)

            a0 = (A + 1.0) - (A - 1.0) * cosw + two_sqrt_A_alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosw)
            a2 = (A + 1.0) - (A - 1.0) * cosw - two_sqrt_A_alpha

        b = np.array([b0, b1, b2], dtype=np.float32)
        a = np.array([a0, a1, a2], dtype=np.float32)

        self.set_coefs(b, a)
    
class DelayLine:
    def __init__(self, max_delay_seconds, fs):
        if max_delay_seconds <= 0:
            raise ValueError("max_delay_seconds must be positive")

        self.fs = fs
        self.size = int(np.ceil(max_delay_seconds * fs)) + 4
        self.buffer = np.zeros(self.size, dtype=np.float32)
        self.write_idx = 0

    def reset(self):
        self.buffer.fill(0.0)
        self.write_idx = 0

    def write(self, sample):
        self.buffer[self.write_idx] = np.float32(sample)
        self.write_idx = (self.write_idx + 1) % self.size

    def read(self, delay_samples):
        delay_samples = float(delay_samples)

        if delay_samples < 0:
            delay_samples = 0.0

        if delay_samples > self.size - 2:
            delay_samples = self.size - 2

        read_idx = self.write_idx - delay_samples

        # modulo wrap
        read_idx = read_idx % self.size

        i0 = int(np.floor(read_idx))
        i1 = (i0 + 1) % self.size
        frac = read_idx - i0

        return (1.0 - frac) * self.buffer[i0] + frac * self.buffer[i1]
    
class DelayRevAPF:
    def __init__(self, delay_seconds, g=0.5, fs=44100):
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")

        if abs(g) >= 1.0:
            raise ValueError("all-pass coefficient requires |g| < 1")

        self.fs = fs
        self.delay_samples = delay_seconds * fs
        self.g = g

        self.delayline = DelayLine(
            max_delay_seconds=delay_seconds + 0.01,
            fs=fs,
        )

    def reset(self):
        self.delayline.reset()

    def process(self, x):
        delayed = self.delayline.read(self.delay_samples)

        y = -self.g * x + delayed

        self.delayline.write(x + self.g * y)

        return y
    
class FirstOrderAPF:
    """
    First-order all-pass filter:

        y[n] = -a*x[n] + x[n-1] + a*y[n-1]

    Used for classic phaser stages.
    """

    def __init__(self, coef=0.0):
        self.a = float(coef)
        self.z = 0.0

    def reset(self):
        self.z = 0.0

    def set_coef(self, coef):
        # Stability guard.
        self.a = float(np.clip(coef, -0.999, 0.999))

    def process(self, x):
        # Stable first-order all-pass:
        # y[n] = -a*x[n] + z[n-1]
        # z[n] = x[n] + a*y[n]
        y = -self.a * x + self.z
        self.z = x + self.a * y
        return y
    
class FeedbackComb:
    def __init__(self, delay_seconds, feedback=0.6, damping=0.4, fs=44100):
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")

        self.fs = fs
        self.delay_samples = delay_seconds * fs

        self.delayline = DelayLine(
            max_delay_seconds=delay_seconds + 0.01,
            fs=fs,
        )

        self.feedback = feedback
        self.damping = damping
        self.filter_state = 0.0

    def reset(self):
        self.delayline.reset()
        self.filter_state = 0.0

    def set_params(self, feedback=None, damping=None):
        if feedback is not None:
            if not 0.0 <= feedback < 0.95:
                raise ValueError("feedback must be in [0, 0.95)")
            self.feedback = feedback

        if damping is not None:
            if not 0.0 <= damping < 1.0:
                raise ValueError("damping must be in [0, 1)")
            self.damping = damping

    def process(self, x):
        delayed = self.delayline.read(self.delay_samples)

        self.filter_state = (
            (1.0 - self.damping) * delayed
            + self.damping * self.filter_state
        )

        self.delayline.write(x + self.feedback * self.filter_state)

        return delayed
    
class CabFiltChain:
    # helper chain class
    def __init__(self, fs, preset):
        self.fs = fs
        self.lpf = Biquad(fs)
        self.hpf = Biquad(fs)
        self.peaks = []

        self.configure(preset)

    def configure(self, preset):
        self.hpf.HPF(preset["hpf"])
        self.lpf.LPF(preset["lpf"])

        self.peaks = []

        for fc, gain, Q in preset["peaks"]:
            bq = Biquad(self.fs)
            bq.Peak(fc, gain, Q)
            self.peaks.append(bq)
    
    def reset(self):
        self.hpf.reset()
        self.lpf.reset()

        for p in self.peaks:
            p.reset()

    def process(self, x):
        y = self.hpf.process_block(x)

        for p in self.peaks:
            y = p.process_block(y)

        y = self.lpf.process_block(y)

        return y.astype(np.float32)