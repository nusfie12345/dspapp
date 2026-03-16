# This is a DSP quasi-app project, designed as a LinAl final project.
# Structure and config by Taras Duma, Vladyslav Cherniuk, Bohdan Dhzus. 

import numpy as np
from scipy.signal import butter, lfilter

def scale_down(param):
    """
    A function to properly normalize all the inputs to the scale [0, 1].

    Args:
        param: designed parameter for the effect to normalize.
    """
    if param < 0:
        param = 0
    if param > 10:
        param = 10
    return param / 10

def volume(param):
    """
    A scale function reserved specifically for the "volume" parameter.

    Args:
        param: volume value
    """
    assert param >= 0
    assert param <= 1
    db = -20 + 26 * param
    return 10**(db/20)

def lowpass(x, cutoff, fs=44100):
    """
    A simple LPF.

    Args:
        x: input signal
        cutoff: cutoff frequency
    """
    b, a = butter(1, cutoff/(fs/2), "low")
    return lfilter(b, a, x)

def highpass(x, cutoff, fs=44100):
    """
    A simple HPF.

    Args:
        x: input signal
        cutoff: cutoff frequency
    """
    b, a = butter(1, cutoff/(fs/2), "high")
    return lfilter(b, a, x)

def drive_eff(param, min_val, max_val):
    """
    A drive effect rescale designed for OD, distortion, and fuzz.

    Args:
        param: initial drive value
        min_val: minimum value for the effect
        max_val: maximum value for the effect
    """
    assert param >= 0
    assert param <= 1
    return min_val * (max_val/min_val)**param

def shelf(x, gain, s, type, fc, fs=44100):
    """
    A setup for the shelf filter; used in the block amp and the EQ.

    Args:
        x: imput signal
        gain: volume gain for the frequency selected
        s: 'cutoff slope'(refer to Audio EQ Cookbook)
        type: high/low shelf filter; string format
        fc: target frequency
        fs: sample frequency, default at 44100 Hz
    """
    A = 10**(gain/40)
    o = 2 * np.pi * fc / fs
    alpha = np.sin(o)/2 * np.sqrt((A + 1/A)*(1/s - 1) + 2)
    cosine = np.cos(o)
    root = np.sqrt(A) # just shortening stuff

    if type == "low":

        b0 = A*((A+1)-(A-1)*cosine+2*alpha*root)
        b1 = 2*A*((A+1)-(A-1)*cosine)
        b2 = A*((A+1)-(A-1)*cosine-2*alpha*root)

        a0 = (A+1)+(A-1)*cosine+2*alpha*root
        a1 = -2*((A+1)+(A-1)*cosine)
        a2 = (A+1)+(A-1)*cosine-2*alpha*root

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1/a0, a2/a0])
        return lfilter(b, a, x)
    
    else: 

        b0 = A*((A+1)+(A-1)*cosine+2*alpha*root)
        b1 = -2*A*((A+1)+(A-1)*cosine)
        b2 = A*((A+1)+(A-1)*cosine-2*alpha*root)

        a0 = (A+1)-(A-1)*cosine+2*alpha*root
        a1 = 2*((A+1)-(A-1)*cosine)
        a2 = (A+1)-(A-1)*cosine-2*alpha*root

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1/a0, a2/a0])
        return lfilter(b, a, x)
    

class Effect:
    def __init__(self, fs):
        self.fs = fs
        self.enabled = False
    
    def process(self, x):
        return x

class Compressor(Effect):

    # 

    def __init__(self, sens, level, atk=5e-04, rel=0.01, r=4, fs=44100):
        super().__init__(fs)
        self.params = {
            "sens": scale_down(sens),
            "level": scale_down(level),
            "atk": atk,
            "rel": rel,
            "r": r
        }
        self.enabled = False

    def upd_param(self):
        pass

    def set_param(self, name, val):
        if val < 0 or val > 10:
            val = scale_down(val)
        self.params[name] = val
        self.upd_param()

    def toggle(self):
        if self.enabled == True:
            return False
        else:
            return True
        
    def process(self, x):
        if not self.enabled:
            return x
        
        samp = self.params["fs"]
        atk = self.params["atk"]
        rel = self.params["rel"]
        level = self.params["level"]
        sens = self.params["sens"]
        r = self.params["r"]
        
        y = np.zeros_like(x)
        env = 0.0
        a_atk = np.exp(-1/(samp*atk))
        a_rel = np.exp(-1/(samp*rel))
        for i, smp in enumerate(x):
            rectif = abs(smp)
            if rectif > env:
                env = a_atk*env + (1 - a_atk)*rectif
            else:
                env = a_rel*env + (1 - a_rel)*rectif
            if env > sens:
                gain = (sens + (env - sens)/r)/env
            else:
                gain = level

            end_level = 0.99*level + 0.01*gain 
            vol = volume(end_level)

        y[i] = vol * smp
        return y

class Overdrive(Effect):

    # y = atan2(gx + b) or y = tanh(gx + b)
    # TO IMPLEMENT: b (asymmetry coefficient)

    def __init__(self, drive, tone, level, fs=44100):
        super().__init__(fs)
        self.params = {
            "drive": scale_down(drive),
            "tone": scale_down(tone),
            "level": scale_down(level)
        }
        self.enabled = False

    def upd_param(self):
        pass

    def set_param(self, name, val):
        if val < 0 or val > 10:
            val = scale_down(val)
        self.params[name] = val
        self.upd_param()

    def toggle(self):
        if self.enabled == True:
            return False
        else:
            return True
        
    def process(self, x):
        if not self.enabled:
            return x
        
        drive = self.params["drive"]
        tone = self.params["tone"]
        level = self.params["level"]
        fs = self.params["fs"]
        
        drive = drive_eff(drive, 1, 20)
        vol = volume(level)

        x = highpass(x, 720, fs)
        x = np.tanh(drive*x)
        x = lowpass(x, 1000+(tone*4000), fs)

        return vol * x

class BlockAmp(Effect):

    # a bunch of shelf filters + some basic overdrive

    def __init__(self, gain, tone, bass, mid, treb, pres, vol, fs=44100):
        super().__init__(fs)
        self.params = {
            "drive": scale_down(gain),
            "tone": scale_down(tone),
            "bass": scale_down(bass),
            "mid": scale_down(mid),
            "treb": scale_down(treb),
            "pres": scale_down(pres),
            "vol": scale_down(vol)
        }
        self.enabled = False

    def upd_param(self):
        pass

    def set_param(self, name, val):
        if val < 0 or val > 10:
            val = scale_down(val)
        self.params[name] = val
        self.upd_param()

    def toggle(self):
        if self.enabled == True:
            return False
        else:
            return True
        
    def preamp_gain(x, gain, dc=0):
        gain = scale_down(gain)
        gain = drive_eff(gain, 1, 50)
        return np.tanh(gain * x + dc)
    
    def preamp_tone(x, gain):
        """
        General setup for the "tone" knob at the block amp.
        """

        gain = scale_down(gain)
        G = gain * 6

        x = shelf(x, G, 1, "high", 1000)
        x = shelf(x, -G, 1, "low", 1000)

        return x

    def preamp_bass(x, gain):
        gain = scale_down(gain)
        G = gain * 10

        x = shelf(x, G, 1, "high", 80)
        x = shelf(x, G, 1, "low", 120)

        return x

    def preamp_mid(x, gain):

        gain = scale_down(gain)
        G = gain * 12

        x = shelf(x, G, 1, "high", 500)
        x = shelf(x, G, 1, "low", 900)

        return x

    def preamp_treble(x, gain):

        gain = scale_down(gain)
        G = gain * 10

        x = shelf(x, G, 1, "high", 3000)
        x = shelf(x, G, 1, "low", 5000)

        return x

    def preamp_presence(x, gain):

        gain = scale_down(gain)
        G = gain * 6

        x = shelf(x, G, 0.7, "high", 6000)
        x = shelf(x, G, 0.7, "low", 10000)

        return x

    def process(self, x):
        if not self.enabled:
            return x
        
        gain = self.params["gain"]
        tone = self.params["tone"]
        bass = self.params["bass"]
        mid = self.params["mid"]
        treb = self.params["treb"]
        pres = self.params["pres"]
        vol = self.params["vol"]

        x = self.preamp_gain(x, gain)
        x = self.preamp_tone(x, tone)
        x = self.preamp_bass(x, bass)
        x = self.preamp_mid(x, mid)
        x = self.preamp_treble(x, treb)
        x = self.preamp_presence(x, pres)
        vol = scale_down(vol)

        return vol * x
    

class Flanger(Effect):

    # y[t] = x[t] + \alpha x(t - \tau[t]) 

    def __init__(self, rate, depth, feedback, mix, fs=44100):
        super().__init__(fs)
        self.params = {
            "rate": scale_down(rate),
            "depth": scale_down(depth),
            "feedback": scale_down(feedback),
            "mix": scale_down(mix)
        }
        self.enabled = False
        self.max_delay = int(0.01 * self.fs)
        self.buffer = np.zeros(self.max_delay)
        self.write_idx = 0
    
    def upd_param(self):
        pass

    def set_param(self, name, val):
        if val < 0 or val > 10:
            val = scale_down(val)
        self.params[name] = val
        self.upd_param()

    def toggle(self):
        if self.enabled == True:
            return False
        else:
            return True
        
    def delay_write(self, samples):
        self.max_delay[self.write_idx] = samples
        self.buffer = np.zeros(self.max_delay)
        self.write_idx = (self.write_idx + 1) % self.max_delay
        return None
    
    def delay_read(self, samples):
        read_idx = (self.write_idx - samples) % self.max_delay
        i0 = int(np.floor(read_idx))
        i1 = (i0 + 1) % self.max_delay

        f = read_idx - i0
        flang_delay = (1 - f) * self.buffer[i0] + f * self.buffer[i1]
        return flang_delay

    def process(self, x):
        if not self.enabled:
            return x

        rate = 0.05 + 5*self.params["rate"]
        depth = 0.001 + 0.004*self.params["depth"]
        base_delay = 0.0003 + 0.004*self.params["delay"]
        feedback = 0.95 * self.params["feedback"]
        alpha = self.params["mix"]
        fs = self.fs

        samps = base_delay * self.fs
        depth_samps = depth * self.fs

        y = np.zeros_like(x)
        phase = 0

        for n in range(len(x)):
            lfo = np.sin(phase)
            tau = samps + depth_samps * lfo
            flang_delay = self.delay_read(tau)
            y[n] = x[n] + alpha * flang_delay
            self.delay_write(x[n] + feedback * flang_delay)
            phase += 2*np.pi*rate/fs

        return y

class Flanger(Effect):

    # same as flanger - but delay is much longer

    def __init__(self, feedback, mix, fs=44100):
        super().__init__(fs)
        self.params = {
            "feedback": scale_down(feedback),
            "mix": scale_down(mix)
        }
        self.enabled = False
        self.max_delay = int(0.01 * self.fs)
        self.buffer = np.zeros(self.max_delay)
        self.write_idx = 0
    
    def upd_param(self):
        pass

    def set_param(self, name, val):
        if val < 0 or val > 10:
            val = scale_down(val)
        self.params[name] = val
        self.upd_param()

    def toggle(self):
        if self.enabled == True:
            return False
        else:
            return True
        
    def delay_write(self, samples):
        self.max_delay[self.write_idx] = samples
        self.buffer = np.zeros(self.max_delay)
        self.write_idx = (self.write_idx + 1) % self.max_delay
        return None
    
    def delay_read(self, samples):
        read_idx = (self.write_idx - samples) % self.max_delay
        i0 = int(np.floor(read_idx))
        i1 = (i0 + 1) % self.max_delay

        f = read_idx - i0
        flang_delay = (1 - f) * self.buffer[i0] + f * self.buffer[i1]
        return flang_delay

    def process(self, x):
        if not self.enabled:
            return x

        base_delay = 0.05 + 1.5*self.params["delay"]
        feedback = 0.95 * self.params["feedback"]
        alpha = self.params["mix"]

        samps = base_delay * self.fs

        y = np.zeros_like(x)

        for n in range(len(x)):
            delay = self.delay_read(samps)
            y[n] = (1-alpha)*x[n] + alpha*delay
            self.delay_write(x[n] + feedback * delay)

        return y