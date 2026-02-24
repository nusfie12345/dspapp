# This is a DSP quasi-app project, designed as a LinAl final project.
# Structure and config by Taras Duma, Vladyslav Cherniuk, Bohdan Dhzus. 

import numpy as np
from scipy.signal import butter, lfilter
from scipy import fft

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

class Effect:
    def __init__(self, fs):
        self.fs = fs
        self.enabled = False
    
    def process(self, x):
        return x

class Compressor(Effect):
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

