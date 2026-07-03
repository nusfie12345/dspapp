import numpy as np
from scipy.signal import butter

class Biquad:
    """
    A class of IIR(infinite impulse response) filters. The general formula is 
    y[n] = b_0x[n] + b_1x[n-1] + b_2x[n-2] - a_1y[n-1] - a_2y[n-2]. 
    
    One of the most used frequency filter class in DSP. Includes low- and high-pass filters, as well as low- and high-shelf filters. 
    """
    def __init__(self, fs):
        self.fs = fs
        self.reset()

    def reset(self):
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0

    def set_coefs(self, b, a):
        self.b0, self.b1, self.b2 = b
        self.a1, self.a2 = a[1], a[2]

    def LPF(self, fc):
        b, a = butter(2, fc / (0.5 * self.fs), btype='low')
        self.set_coefs(b, a)

    def HPF(self, fc):
        b, a = butter(2, fc / (0.5 * self.fs), btype='high')
        self.set_coefs(b, a)
    
    def Shelf(self, fc, gain, type="low", s=1):
        if s <= 0:
            raise ValueError("'s' must be greater than 0")
        if not 0 < fc < self.fs / 2:
            raise ValueError("'fc' must be between 0 and Nyquist frequency")
        if type not in ["low", "high"]:
            raise ValueError("'type' must be either 'low' or 'high'")

        A = 10**(gain/40)
        o = 2 * np.pi * fc / self.fs
        alpha = np.sin(o)/2 * np.sqrt((A + 1/A)*(1/s - 1) + 2)
        cos = np.cos(o)
        root = np.sqrt(A) # just shortening stuff. yes i am lazy

        if type == "low":

            b0 = A*((A+1)-(A-1)*cos+2*alpha*root)
            b1 = 2*A*((A-1)-(A+1)*cos)
            b2 = A*((A+1)-(A-1)*cos-2*alpha*root)

            a0 = (A+1)+(A-1)*cos+2*alpha*root
            a1 = -2*((A-1)+(A+1)*cos)
            a2 = (A+1)+(A-1)*cos-2*alpha*root

            b = np.array([b0, b1, b2]) / a0
            a = np.array([1, a1/a0, a2/a0])
            self.set_coefs(b, a)
    
        if type == "high": 

            b0 = A*((A+1)+(A-1)*cos+2*alpha*root)
            b1 = -2*A*((A+1)+(A-1)*cos)
            b2 = A*((A+1)+(A-1)*cos-2*alpha*root)

            a0 = (A+1)-(A-1)*cos+2*alpha*root
            a1 = 2*((A+1)-(A-1)*cos)
            a2 = (A+1)-(A-1)*cos-2*alpha*root

            b = np.array([b0, b1, b2]) / a0
            a = np.array([1, a1/a0, a2/a0])
            self.set_coefs(b, a)
        else:
            raise ValueError("'type' has to be 'low' or 'high'")

    def Peak(self, fc, gain, Q=1.0):
        if not 0 < fc < self.fs / 2:
            raise ValueError("'fc' must be between 0 and Nyquist freq")
        if Q <= 0:
            raise ValueError("'Q' must be strictly positive")
        
        A = 10 ** (gain / 40)
        w0 = 2 * np.pi * fc / self.fs
        alpha = np.sin(w0) / (2 * Q)
        cos = np.cos(w0)

        b0 = 1 + alpha * A
        b1 = -2 * cos
        b2 = 1 - alpha * A

        a0 = 1 + alpha / A
        a1 = -2 * cos
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        self.set_coefs(b, a)

    def process(self, x):
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y
    
class DelayLine:
    def __init__(self, max_delay_seconds, fs):
        if max_delay_seconds <= 0:
            raise ValueError("max_delay_seconds must be positive")

        self.fs = fs
        self.size = int(np.ceil(max_delay_seconds * fs)) + 2
        self.buffer = np.zeros(self.size, dtype=float)
        self.write_idx = 0

    def reset(self):
        self.buffer.fill(0.0)
        self.write_idx = 0

    def write(self, sample):
        self.buffer[self.write_idx] = sample
        self.write_idx = (self.write_idx + 1) % self.size

    def read(self, delay_samples):
        if delay_samples < 0:
            raise ValueError("delay_samples must be non-negative")

        # Clamp instead of crashing if modulation barely exceeds max.
        delay_samples = min(delay_samples, self.size - 2)

        read_idx = self.write_idx - delay_samples

        while read_idx < 0:
            read_idx += self.size
        while read_idx >= self.size:
            read_idx -= self.size

        i0 = int(np.floor(read_idx))
        i1 = (i0 + 1) % self.size

        frac = read_idx - i0

        return (1.0 - frac) * self.buffer[i0] + frac * self.buffer[i1]
    
class DelayRevAPF:
    def __init__(self, delay_seconds, g=0.5, fs=44100):
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")
        if abs(g) >= 1:
            raise ValueError("all-pass coefficient requires |g| < 1")

        self.fs = fs
        self.delay_samples = delay_seconds * fs
        self.delay = DelayLine(delay_seconds + 0.01, fs)
        self.g = g

    def reset(self):
        self.delay.reset()

    def set_params(self, g=None):
        if g is not None:
            if abs(g) >= 1:
                raise ValueError("all-pass coefficient requires |g| < 1")
            self.g = g

    def process(self, x):
        delayed = self.delay.read(self.delay_samples)

        y = -self.g * x + delayed
        self.delay.write(x + self.g * y)

        return y
    
class FirstOrderAPF:
    """
    First-order all-pass filter:

        y[n] = -a*x[n] + x[n-1] + a*y[n-1]

    Used for classic phaser stages.
    """

    def __init__(self, a=0.0):
        if not -1.0 < a < 1.0:
            raise ValueError("All-pass coefficient a must be in (-1, 1)")

        self.a = a
        self.x1 = 0.0
        self.y1 = 0.0

    def set_coef(self, a):
        if not -1.0 < a < 1.0:
            raise ValueError("All-pass coefficient a must be in (-1, 1)")
        self.a = a

    def reset(self):
        self.x1 = 0.0
        self.y1 = 0.0

    def process(self, x):
        y = -self.a * x + self.x1 + self.a * self.y1

        self.x1 = x
        self.y1 = y

        return y
    
class FeedbackComb:
    def __init__(self, delay_seconds, feedback=0.7, damping=0.2, fs=44100):
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")
        if not 0 <= feedback < 1:
            raise ValueError("feedback must be in [0, 1)")
        if not 0 <= damping <= 1:
            raise ValueError("damping must be in [0, 1]")

        self.fs = fs
        self.delay_samples = delay_seconds * fs
        self.delay = DelayLine(delay_seconds + 0.01, fs)

        self.feedback = feedback
        self.damping = damping
        self.filter_state = 0.0

    def reset(self):
        self.delay.reset()
        self.filter_state = 0.0

    def set_params(self, feedback=None, damping=None):
        if feedback is not None:
            if not 0 <= feedback < 1:
                raise ValueError("feedback must be in [0, 1)")
            self.feedback = feedback

        if damping is not None:
            if not 0 <= damping <= 1:
                raise ValueError("damping must be in [0, 1]")
            self.damping = damping

    def process(self, x):
        delayed = self.delay.read(self.delay_samples)

        # One-pole low-pass in feedback path.
        self.filter_state = (
            (1.0 - self.damping) * delayed
            + self.damping * self.filter_state
        )

        self.delay.write(x + self.feedback * self.filter_state)

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
        y = self.hpf.process(x)

        for p in self.peaks:
            y = p.process(y)

        y = self.lpf.process(y)

        return y