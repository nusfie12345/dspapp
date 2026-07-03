# This is a DSP quasi-app project, designed as a LinAl final project.
# Structure and config by Taras Duma, Vladyslav Cherniuk, Bohdan Dhzus. 

import numpy as np

import filters as flt
import utils as uts
import reverbs as revb

from scipy.signal import fftconvolve 

# IGNORE; COPY-PASTEABLE CUSTOM LETTERS FOR FORMULAS
# α
# 𝜏
# \(\pi \)

# maybe a separate file for filters and scalings? porting the stuff to cpp will be HELL tho

class Effect:
    def __init__(self, fs):
        self.fs = fs
        self.enabled = False
    
    def process(self, x):
        return x

class Compressor(Effect):

    # anti-fizzy stuff

    def __init__(self, sens, level, atk=5e-04, rel=0.01, r=4, fs=44100):
        super().__init__(fs)
        self.params = {
            "sens": uts.normalize(sens),
            "level": uts.normalize(level),
            "atk": atk,
            "rel": rel,
            "r": r
        }
        self.fs = fs
        self.enabled = False
        self.env = 0.0
        self.gain = 1.0

    def upd_param(self):
        pass

    def set_param(self, name, val):
        if 0 <= val <= 10:
            val = uts.normalize(val)

        if name in ("atk", "rel") and val <= 0:
            raise ValueError(f"{name} must be positive")

        if name == "r" and val < 1:
            raise ValueError("ratio must be >= 1")
            
        self.params[name] = val
        self.upd_param()

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

        
    def process(self, x):
        x = np.array(x, dtype=float)

        if not self.enabled:
            return x
        
        samp = self.fs
        atk = self.params["atk"]
        rel = self.params["rel"]
        level = self.params["level"]
        sens = self.params["sens"]
        r = self.params["r"]
        
        y = np.zeros_like(x)

        a_atk = np.exp(-1/(samp*atk))
        a_rel = np.exp(-1/(samp*rel))

        mkp = uts.volume_gain(level)

        for i, smp in enumerate(x):
            rectif = abs(smp)

            if rectif > self.env:
                self.env = a_atk*self.env + (1 - a_atk)*rectif
            else:
                self.env = a_rel*self.env + (1 - a_rel)*rectif

            if self.env > sens:
                cmp_gain = (sens + (self.env - sens)/r)/self.env
            else:
                cmp_gain = 1.0

            target = mkp * cmp_gain

            if target < self.gain:
                coef = a_atk
            else:
                coef = a_rel

            self.gain = coef*self.gain + (1 - coef)*target

            y[i] = self.gain * smp
        return y

class Overdrive(Effect):

    # GHOOOOST OOOF PERDITIOOOOOON STUCK IN HER CHEEEEEST
    # TO-DO: maybe order-fibonacci clip? idk, will probably be super messy with order-8 terms
    # stacking anything higher than ord5(which is harmonics up to 2 octaves plus a major 3rd above the root btw) will probably be a sonic hell

    r"""
    Initially formatted after Ibanez Tube Screamer TS9 pedal, but with more options over the clipping type. 
    Maybe will be updated with more tone shaping options, but it's good enough for now.

    Equation: 
    $$
    y = softclip(gx + b) 
    $$
    g is the drive, b is the asymmetry control, and the softclip can be either atan or tanh depending on the "clip" parameter. Polynomial softclips allow for greater harmonics control.

    Args:
        drive: Gain parameter, the amount of incoming signal scaling. Input: [0, 10], scaled down to [0, 1] and then to [1, 20].
        tone: Tone control, the amount of high-end boost. Input: [0, 10], scaled down to [0, 1] and then to [0, 4000] - essentially an LPF.
        level: Output volume. Input: [0, 10], scaled down to [0, 1] and then to [-20, 6] dB for the volume control.
        mode: Advanced waveshaping method, selectable between "clear"(y = x), "quad"(y = x + ax^2) and "qcubic"(y = x + ax^2 + bx^3). Default is "clear".
        clip: Clipping(bounding to [-1, 1]) type, selectable between "atan" and "tanh". Default is "tanh". 
        fs: Sample frequency, default at 44100 Hz.
    """

    def __init__(self, drive, tone, level, mode="clear", clip="tanh", fs=44100):
        super().__init__(fs)
        self.params = {
            "drive": uts.normalize(drive),
            "tone": uts.normalize(tone),
            "level": uts.normalize(level),
            "clip": clip,   # didn't want to make it into a separate effect, so will be a simple switch
            "mode": mode
        }
        self.lpf = flt.Biquad(fs)
        self.lpf.LPF(1000+(self.params["tone"]*4000))
        self.hpf = flt.Biquad(fs)
        self.hpf.HPF(720)
        self.postclip_hpf = flt.Biquad(fs)
        self.postclip_hpf.HPF(20)
        self.enabled = False

    def upd_param(self):
        self.lpf.LPF(1000+(self.params["tone"]*4000))

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")
        
        if name == "mode":
            if val not in ["clear", "quad", "qcubic"]:
                raise ValueError("Invalid clip type. Use 'clear', 'quad' or 'qcubic'.")
            self.params[name] = val
            return 
        
        if name == "clip":
            if val not in ["atan", "tanh"]: 
                raise ValueError("Invalid bounding method. Use 'atan' or 'tanh'." )
            self.params[name] = val
            return
            
        if name in ("drive", "tone", "level"):
            if 0 <= val <= 10:
                val = uts.normalize(val)
            else:
                raise ValueError(f"Parameter '{name}' must be in the range [0, 10]")
            
            self.params[name] = val 
            self.upd_param()

        return

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
    
    def _bound(self, x):   # MUST-USE btw. otherwise the final equation might explode
        clip = self.params["clip"]

        if clip == "atan":
            return (2/np.pi) * np.arctan(x)
        elif clip == "tanh":
            return np.tanh(x)
        else:
            raise ValueError(f"Invalid bounding method: {clip}")
    
    def _softclip(self, x):
        drive = uts.drive_eff(self.params["drive"], 1, 20)
        mode = self.params["mode"]

        gx = drive * x  
        gxb = gx + 1e-4  # DC offset for atan and tanh

        match mode:
            case "clear":
                y = gxb
            case "quad":
                y = gx + (0.25 * (gx ** 2))
            case "qcubic":
                y = gx + (0.2 * (gx ** 2)) - (0.3 * (gx ** 3))
            case _:
                raise ValueError("Please use the valid waveshaping method.")
        
        return self._bound(y)
        
    def algo(self, x): 
        vol = uts.volume_gain(self.params["level"])

        x = self.hpf.process(x)
        x = self._softclip(x)
        x = self.postclip_hpf.process(x)
        x = self.lpf.process(x)

        return vol * x
    
    def process(self, x):
        x = np.asarray(x, dtype=float)
        if not self.enabled:
            return x.copy()

        y = np.zeros_like(x)
        for n, sample in enumerate(x):
            y[n] = self.algo(sample)
        return y
    
class Distortion(Effect):

    # good ol classic, harsher clip and steeper gain but more bass at LPF, also no asymmetry control

    # conforming fasteeer
    # obey your masteeer
    # your life burns fasteeeer
    # obey your MASTER, MASTER

    """
    A distortion simulation, initially formatted after Boss DS-1 pedal, might get expanded to cover other distortion algorithms of different analog pedals.
    This one offers a seesaw tone control, which boosts the highs and cuts the lows at one end of the spectrum, and does the opposite at the other end. 
    (Optional: traditional LPF)
    The clipping is a hard clip, with the threshold being a set value for now but may be made a parameter later.

    Equation: 
    
        y = hardclip(gx, T)
    g is the drive, T is the clipping threshold.

    Args:
        dist: Drive parameter, the amount of incoming signal scaling. Input: [0, 10], scaled down to [0, 1] and then to [1, 50].
        tone: Tone control, the amount of high-end boost. Input: [0, 10], scaled down to [0, 1] and then to [0, 4000].
        level: Output volume. Input: [0, 10], scaled down to [0, 1] and then to [-20, 6] dB for the volume control.
        mode: Advanced waveshaping method. Selectable between 'standard'(shown above), cubic- ('cubic': y = 1.5x - 0.5x^3), or order-5- ('ord5': y = 1.5x - 0.5x^3 + 0.0833x^5) polynomial clippings.
        filter: Type of tone shaping. Selectable between 'seesaw' for a Boss DS-1-like toneshaping, or 'classic' for a standard LPF cutoff.
        fs: Sample frequency, default at 44100 Hz.
    """

    def __init__(self, dist, tone, level, mode="standard", filter="classic", fs=44100):
        super().__init__(fs)
        self.params = {
            "dist": uts.normalize(dist),
            "tone": uts.normalize(tone),
            "level": uts.normalize(level),
            "filter": filter,
            "mode": mode
        }
        self.hpf = flt.Biquad(fs)
        self.hpf.HPF(100)

        self.tone_lpf = flt.Biquad(fs)
        self.tone_lpf.LPF(800+(self.params["tone"]*5200))
        self.tone_hpf = flt.Biquad(fs)
        self.tone_hpf.HPF(800+(self.params["tone"]*5200))
        self.enabled = False

    def upd_param(self):
        self.tone_lpf.LPF(800+(self.params["tone"]*5200))
        self.tone_hpf.HPF(800+(self.params["tone"]*5200))

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")
        
        if name == "filter":
            if val not in ["classic", "seesaw"]:
                raise ValueError("Invalid filter type. Use 'classic' or 'seesaw'.")
            self.params[name] = val
            return
        
        if name == "mode":
            if val not in ["standard", "cubic", "ord5"]:
                raise ValueError("Invalid clipping type. Use 'standard', 'cubic' or 'ord5'.")
            self.params[name] = val
            return
            
        if name in ("dist", "tone", "level"):
            if 0 <= val <= 10:
                val = uts.normalize(val)
            else:
                raise ValueError(f"Parameter '{name}' must be in the range [0, 10]")
        
            self.params[name] = val
            self.upd_param()

        return

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
    
    def _seesaw(self, x, tone):
        low = self.tone_lpf.process(x)
        high = self.tone_hpf.process(x)

        return (1 - tone)*low + tone*high
    
    def _tone_selector(self, x, filter):
        if filter == "classic":
            return self.tone_lpf.process(x)
        elif filter == "seesaw":
            return self._seesaw(x, self.params["tone"])
        else:
            raise ValueError("Invalid filter type. Use 'classic' or 'seesaw'.")
    
    def _hardclip(self, x):
        dist = uts.drive_eff(self.params["dist"], 1, 50)
        mode = self.params["mode"]

        gx = dist * x

        match mode:
            case "standard":
                y = gx
            case "cubic":
                y = 1.5 * gx - (0.5 * (gx ** 3))
            case "ord5":
                y = 1.5 * gx - (0.5 * (gx ** 3)) + (0.0833 * (gx ** 5))
            case _:
                raise ValueError("Please pick a valid clipping method.")
            
        return np.clip(y, -0.6, 0.6)
        
    def algo(self, x):
        
        level = self.params["level"]

        vol = uts.volume_gain(level)

        x = self.hpf.process(x) # for them lower grooves
        x = self._hardclip(x) 
        x = self._tone_selector(x, self.params["filter"])

        return vol * x
    
    def process(self, x):
        x = np.asarray(x, dtype=float)
        if not self.enabled:
            return x.copy()

        y = np.zeros_like(x)
        for n, sample in enumerate(x):
            y[n] = self.algo(sample)
        return y
    
class Fuzz(Effect):

    # IF, WHEN I SAY I MAY FADE LIKE A SIGH IF I STAAAAAY
    # YOU MINIMIZE MY MOVEMENT ANYYYYWAAAAAY
    # I MUST PERSUADE YOU ANOTHER WAAAAAAAAY
    # PUSHING AND SHOVING, PUSHING AND SHOVING

    """
    Modeled after Big Muff analog pedal, a simulator of fuzz distortion. Offers huge gain and tone stacking, which gives this iconic square-like waveform.
    Discovered almost accidentally, by using faulty music equipment, but this simulation's here so that y'all don't break your stuff.
    
    Equation: 
    
        y = tone_stack(hardclip(gx, T), tone)
    g is the drive, T is the clipping threshold, and tone is the parameter for the tone stack which scoops the mids and boosts the lows and highs.

    Args:
        sustain: Drive parameter, the amount of incoming signal scaling. Input: [0, 10], scaled down to [0, 1] and then to [1, 200].
        tone: Tone control, the amount of mid-range scooping and high-end boosting. Input: [0, 10], scaled down to [0, 1].
        level: Output volume. Input: [0, 10], scaled down to [0, 1] and then to [-20, 6] dB for the volume control.
        asym: Clipping asymmetry, either on or off.
        fs: Sample frequency, default at 44100 Hz.
    """

    def __init__(self, sustain, tone, level, asym=True, fs=44100):
        super().__init__(fs)
        self.params = {
            "sustain": uts.normalize(sustain), # cuz this is hella strong drive, might as well be just sustain
            "tone": uts.normalize(tone),
            "level": uts.normalize(level),
            "asym": asym
        }
        self.prehpf = flt.Biquad(fs)
        self.prehpf.HPF(30)
        self.tone_lpf = flt.Biquad(fs)
        self.tone_hpf = flt.Biquad(fs)

        self.dc_block = flt.Biquad(fs)
        self.dc_block.HPF(20)

        self.enabled = False
        self._filter_upd()

    def _freqs(self):
        tone = self.params["tone"]
        fq_lo = 400 + tone * 5600
        fq_hi = 700 + tone * 2300
        return fq_lo, fq_hi

    def _filter_upd(self):
        fq_lo, fq_hi = self._freqs()
        self.tone_lpf.LPF(fq_lo)
        self.tone_hpf.HPF(fq_hi)

    def upd_param(self):
        self._filter_upd()

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")
        
        if name == "asym":
            if not isinstance(val, bool):
                raise ValueError(f"Asymmetry param should be type bool, got: {type(name)}")
            self.params[name] = val
            return
        
        if name in ("sustain", "tone", "level"):
            if 0 <= val <= 10:
                val = uts.normalize(val)
            else:
               raise ValueError(f"Parameter '{name}' must be in the range [0, 10]")

            self.params[name] = val
            self.upd_param()

        return

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
        
    def _hardclip(self, x, T=1.0):
        e = 0.2
        asym = self.params["asym"]
        if asym == True:
            return np.clip(x+e, -T, T) - e  # asymmetric hardclip
        else:
            return np.clip(x, -T, T)
    
    def _fuzz_shape(self, x):
        gain = uts.drive_eff(self.params["sustain"], 5, 100)
        gx = gain * x

        return self._hardclip(gx, 1.0)
        
    def _tone_stack(self, x):
        tone = self.params["tone"]

        low = self.tone_lpf.process(x)
        high = self.tone_hpf.process(x)

        return (1 - tone) * low + tone * high
        
    def algo(self, x):
        vol = uts.volume_gain(self.params["level"])

        # also screw your HPF, we going hard
        x = self.prehpf.process(x)
        x = self._fuzz_shape(x)
        x = self.dc_block.process(x)
        x = self._tone_stack(x)

        return vol * x
    
    def process(self, x):
        x = np.asarray(x, dtype=float)
        if not self.enabled:
            return x.copy()

        y = np.zeros_like(x)
        for n, sample in enumerate(x):
            y[n] = self.algo(sample)
        return y

class BlockAmp(Effect):

    # a bunch of shelf filters + some basic overdrive. THE holy grail for literally every player, EVH used only the block amp for distortion
    # > "Eruption" playing in the background, RIP EVH

    """
    A block amp simulation, usually a must-have for any pedalboard setup. Offers some distortion/overdrive on its own, as well as volume and EQ control.
    (I forgot which exact analog I was modeling after, but it works pretty much as any other standard-ish amp anyways.)

    Equation: 
    
        y = presence(treble(bass(mid(tone(preamp_gain(x))))))

    (Yep, it's just filter after filter after filter, plus some gain.)
    Args:
        gain: Overall gain parameter, the amount of incoming signal scaling for both block-amp distortion and bandwith boosts. Input: [0, 10], scaled down to [0, 1] and then to [1, 50].
        tone: Tone control, the amount of mid-range scooping and high-end boosting. Input: [0, 10], scaled down to [0, 1].
        bass: Bass control, the amount of boosting frequencies at range 80-120 Hz. Input: [0, 10], scaled down to [0, 1].
        mid: Mid control, the amount of boosting frequencies at range 250-500 Hz. Input: [0, 10], scaled down to [0, 1].
        treb: Treble control, the amount of boosting frequencies at range 3000-5000 Hz. Input: [0, 10], scaled down to [0, 1].
        pres: Presence control, the amount of boosting frequencies at range 6000-10000 Hz. Input: [0, 10], scaled down to [0, 1].
        vol: Volume control, the amount of output volume scaling. Input: [0, 10], scaled down to [0, 1].
        fs: Sample frequency, default at 44100 Hz.

    """

    def __init__(self, drive, tone, bass, mid, treb, pres, vol, fs=44100, clipType="tanh"):
        super().__init__(fs)
        self.params = {
            "drive": uts.normalize(drive),
            "tone": uts.normalize(tone),
            "bass": uts.normalize(bass),
            "mid": uts.normalize(mid),
            "treb": uts.normalize(treb),
            "pres": uts.normalize(pres),
            "vol": uts.normalize(vol),
            "clipType": clipType
        }
        self.toneFilterHI = flt.Biquad(fs)  # need both high and low filters for each frequency range
        self.toneFilterLO = flt.Biquad(fs)
        self.bassFilterHI = flt.Biquad(fs)
        self.bassFilterLO = flt.Biquad(fs)
        self.midFilterHI = flt.Biquad(fs)
        self.midFilterLO = flt.Biquad(fs)
        self.trebFilterHI = flt.Biquad(fs)
        self.trebFilterLO = flt.Biquad(fs)
        self.presFilterHI = flt.Biquad(fs)
        self.presFilterLO = flt.Biquad(fs)
        self.enabled = False

    def _knob_gain_db(self, knob, max_boost_db):
        """
        Maps knob 0..1 to -max_boost_db..+max_boost_db.
        0.5 becomes 0 dB.
        """
        return (knob * 2.0 - 1.0) * max_boost_db

    def _update_filters(self):
        tone_G = self._knob_gain_db(self.params["tone"], 6)
        bass_G = self._knob_gain_db(self.params["bass"], 10)
        mid_G = self._knob_gain_db(self.params["mid"], 12)
        treb_G = self._knob_gain_db(self.params["treb"], 10)
        pres_G = self._knob_gain_db(self.params["pres"], 6)

        self.toneFilterHI.Shelf(1000, tone_G, type="high")
        self.toneFilterLO.Shelf(1000, tone_G, type="low")

        self.bassFilterHI.Shelf(80, bass_G, type="high")
        self.bassFilterLO.Shelf(120, bass_G, type="low")

        self.midFilterHI.Shelf(500, mid_G, type="high")
        self.midFilterLO.Shelf(900, mid_G, type="low")

        self.trebFilterHI.Shelf(3000, treb_G, type="high")
        self.trebFilterLO.Shelf(5000, treb_G, type="low")

        self.presFilterHI.Shelf(6000, pres_G, type="high")
        self.presFilterLO.Shelf(10000, pres_G, type="low")

    def upd_param(self):
        self._update_filters()

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")
        
        if name in ("drive", "tone", "bass", "mid", "treb", "pres", "vol"):
            if 0 <= val <= 10:
                val = uts.normalize(val)
            else:
                raise ValueError(f"Parameter {name} should be in range [0, 10]")
            self.params[name] = val
            return
        
        if name == "clipType":
            if val not in ["atan", "tanh", "hard", "null"]:
                raise ValueError(f"Invalid clipping type. Use 'atan', 'tanh', 'hard', or 'null'.")
            self.params[name] = val
            self.upd_param()
        
        return

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
        
    # tons of amp-specific peak-shelf filters ahead, nothing much. just stack em up and you have the block amp
    # too lazy to code the peak filter separately. sorry, so just a stack of hi-/lo-shelf filters

    def preamp_gain(self, x):
        """
        Additional/optional gain at the block amp stage. Can be used to further drive distortions or smoothen them out.
        """

        drive = uts.drive_eff(self.params["drive"], 1, 50)
        clipType = self.params["clipType"]

        gx = drive * x

        match clipType:
            case "atan":
                return (2/np.pi) * np.arctan(gx)
            case "tanh":
                return np.tanh(gx)
            case "hard":
                return np.clip(gx, -1, 1)
            case "null":
                return x    # skip the drive stage completely; useful if you have enough drive with your OD/distortion/fuzz already
            case _:
                raise ValueError("Invalid clipping type")
    
    def preamp_tone(self, x, drive):
        """
        General setup for the "tone" knob at the block amp.
        """

        x = self.toneFilterHI.process(x)
        x = self.toneFilterLO.process(x)

        return x

    def preamp_bass(self, x, gain):

        """
        Peak-filter/boost for the low-end/bass frequencies, typically ranging from 80 to 120 Hz. 
        
        Reference:
        Comparable to a normal or a slightly lower male voice. 
        """

        x = self.bassFilterHI.process(x)
        x = self.bassFilterLO.process(x)

        return x

    def preamp_mid(self, x, gain):

        """
        Peak-filter/boost for the mid-range frequencies, often determined to peak at 500-900 Hz.

        Reference:
        Comparable to a slightly higher-pitched children's voice, or the mid-range of a soprano pitch. 
        """

        x = self.midFilterHI.process(x)
        x = self.midFilterLO.process(x)

        return x

    def preamp_treble(self, x, gain):

        """
        Peak-filter/boost for the high-end/treble frequencies, typically ranging from 3000 to 5000 Hz.

        Reference:
        Comparable to a bright, crisp sound, often associated with the treble range of a guitar or piano.
        Also, this is the mid-range for human consonants in our everyday speech - what makes our language make sense.
        """

        x = self.trebFilterHI.process(x)
        x = self.trebFilterLO.process(x)

        return x

    def preamp_presence(self, x, gain):

        """
        Peak-filter/boost for the presence frequencies, typically ranging from 6000 to 10000 Hz.

        Reference:
        Comparable to crash cymbals, pinch harmonics(those iconic guitar squeals), or slightly similar to tinnitus. Birds may often chirp at such frequencies.
        """

        x = self.presFilterHI.process(x)
        x = self.presFilterLO.process(x)

        return x

    def algo(self, x):
        
        tone = self.params["tone"]
        bass = self.params["bass"]
        mid = self.params["mid"]
        treb = self.params["treb"]
        pres = self.params["pres"]
        vol = self.params["vol"]

        x = self.preamp_gain(x)
        x = self.preamp_tone(x, tone)
        x = self.preamp_bass(x, bass)
        x = self.preamp_mid(x, mid)
        x = self.preamp_treble(x, treb)
        x = self.preamp_presence(x, pres)
        vol = uts.volume_gain(vol)

        return vol * x
    
    def process(self, x):
        x = np.asarray(x, dtype=float)
        if not self.enabled:
            return x.copy()

        y = np.zeros_like(x)
        for n, sample in enumerate(x):
            y[n] = self.algo(sample)
        return y

class Flanger(Effect):

    # STIR US FROM OUR
    # WANTON SLUMBER
    # MITIGATE THE RUIN, CALL US ALL TO ARMS AND ORDER

    r"""
    Modeled in its standard algorithmic form, a simulation of the flanger effect. Offers the standard parameters for the effect, such as rate, depth, feedback, and mix.
    Discovered by Les Paul in the 1940s using two tape recorders, with one tape being slightly delayed using a finger on the reel.
    Used in creating the comb-like filtering and sweeping using LFO-modulated delay and feedback.
    Perfect in combination with some OD/distortion and wah.

    Equation: 

       y[t] = x[t] + αx[t - 𝜏[t]] 
       𝜏[t] = 𝜏_0 + d * sin(2πf_{LFO}*t)
    α is the mix parameter, 𝜏_0 is the base delay, d is the depth of modulation, and LFO is a low-frequency oscillator (sine wave) that modulates the delay time.

    (Actually, the equation above is shared with the chorus effect as well, except the signal isn't sent into LFO like here.)

    Args:
        rate: LFO rate, the speed of the sweeping effect. Input: [0, 10], scaled down to [0, 1] and then to [0.05, 5] Hz.
        depth: LFO depth, the amount of delay modulation. Input: [0, 10], scaled down to [0, 1] and then to [0.001, 0.005] seconds.
        feedback: Feedback amount, the amount of output signal fed back into the input. Input: [0, 10], scaled down to [0, 1] and then to [0, 0.95].
        mix: Mix parameter, the balance between the dry and wet signals. Input: [0, 10], scaled down to [0, 1].
        fs: Sample frequency, default at 44100 Hz.
    """

    def __init__(self, rate, depth, feedback, mix, delay=5, fs=44100):
        super().__init__(fs)
        self.params = {
            "rate": uts.normalize(rate),
            "depth": uts.normalize(depth),
            "feedback": uts.normalize(feedback),
            "mix": uts.normalize(mix),
            "delay": uts.normalize(delay)
        }
        self.enabled = False
        self.delayline = flt.DelayLine(max_delay_seconds=0.03, fs=fs)
        self.phase = 0.0
    
    def upd_param(self):
        pass

    def set_param(self, name, val):
        if name not in self.params[name]:
            raise ValueError(f"Invalid parameter name: {name}")
        
        if 0 <= val <= 10:
            val = uts.normalize(val)
        else:
            raise ValueError(f"Parameter {name} must be in range [0, 10]")
        
        self.params[name] = val
        self.upd_param()

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def process(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x

        rate = 0.05 + 5.0 * self.params["rate"]
        depth = 0.0002 + 0.004 * self.params["depth"]  # 0.2-4.2 ms
        base_delay = 0.0005 + 0.006 * self.params["delay"]  # 0.5-6.5 ms
        feedback = 0.95 * self.params["feedback"]
        mix = self.params["mix"]

        y = np.zeros_like(x)

        for n, sample in enumerate(x):
            lfo = 0.5 + 0.5 * np.sin(self.phase)
            tau = base_delay + depth * lfo
            tau_samps = tau * self.fs

            delayed = self.delayline.read(tau_samps)

            y[n] = (1.0 - mix) * sample + mix * delayed

            self.delayline.write(sample + feedback * delayed)

            self.phase += 2.0 * np.pi * rate / self.fs
            if self.phase >= 2.0 * np.pi:
                self.phase -= 2.0 * np.pi

        return y

class Chorus(Effect):

    # TO ASSUME FROM IGNORANCE
    # INFLICTING WOUNDS WITH YOUR CROSS TURNED DAGGER

    # LITERALLY FLANGER BUT WITH A FEW TWEAKS

    def __init__(self, rate, depth, feedback, mix, delay=5, fs=44100):
        super().__init__(fs)
        self.params = {
            "rate": uts.normalize(rate),
            "depth": uts.normalize(depth),
            "feedback": uts.normalize(feedback),
            "mix": uts.normalize(mix),
            "delay": uts.normalize(delay)
        }
        self.enabled = False
        self.delayline1 = flt.DelayLine(max_delay_seconds=0.08, fs=fs)
        self.delayline2 = flt.DelayLine(max_delay_seconds=0.08, fs=fs)
        self.phase = 0.0
    
    def upd_param(self):
        pass

    def set_param(self, name, val):
        if name not in self.params[name]:
            raise ValueError(f"Invalid parameter name: {name}")
        
        if 0 <= val <= 10:
            val = uts.normalize(val)
        else:
            raise ValueError(f"Parameter {name} must be in range [0, 10]")
        
        self.params[name] = val
        self.upd_param()

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def process(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        # delay slightly longer than that of chorus
        rate = 0.05 + 2.5 * self.params["rate"]
        base_delay = 0.008 + 0.020 * self.params["delay"]   # 8–28 ms
        depth = 0.002 + 0.012 * self.params["depth"]        # 2–14 ms
        feedback = 0.2 * self.params["feedback"]
        mix = self.params["mix"]

        y = np.zeros_like(x)

        for n, sample in enumerate(x):
            lfo1 = 0.5 + 0.5 * np.sin(self.phase)
            lfo2 = 0.5 + 0.5 * np.sin(self.phase + np.pi / 2)

            tau1 = (base_delay + depth * lfo1) * self.fs
            tau2 = (base_delay + depth * lfo2) * self.fs

            d1 = self.delayline1.read(tau1)
            d2 = self.delayline2.read(tau2)

            wet = 0.5 * (d1 + d2)

            y[n] = (1.0 - mix) * sample + mix * wet

            # feedback should typically be near-zero
            self.delayline1.write(sample + feedback * d1)
            self.delayline2.write(sample + feedback * d2)

            self.phase += 2.0 * np.pi * rate / self.fs
            if self.phase >= 2.0 * np.pi:
                self.phase -= 2.0 * np.pi

        return y

class Delay(Effect):

    # same as flanger - but delay is much longer, also no modulation and no mix

    # black then white are all I see in my infancy
    # > proceeds to change the time signature 11235813 times
    # > F E T T U C C I N E  S E Q U E N C E

    def __init__(self, feedback, mix, delay_ms=500, mode="ms", bpm=120, div="1/4", echo=None, max_delay_seconds=3.0, fs=44100):
        super().__init__(fs)
        self.params = {
            "feedback": uts.normalize(feedback),
            "mix": uts.normalize(mix),
            "delay_ms": delay_ms,
            "mode": mode,
            "bpm": bpm,
            "div": div,
            "echo": echo
        }
        self.enabled = False
        self.delayline = flt.DelayLine(max_delay_seconds=max_delay_seconds, fs=fs)
    
    def _delay_seconds(self):
        mode = self.params["mode"]

        if mode == "ms":
            delay_ms = self.params["delay_ms"]
            if delay_ms <= 0:
                raise ValueError("delay_ms must be positive")
            return delay_ms / 1000.0

        if mode == "sync":
            return uts.bpm_to_delay_seconds(self.params["bpm"], self.params["div"])

        raise ValueError(f"Invalid delay mode: {mode}")

    def _feedback_gain(self):
        echo = self.params["echo"]

        if echo is not None:
            return uts.feedback_from_echoes(echo)

        return 0.95 * self.params["feedback"]

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")

        match name:
            case "mix":
                if 0 <= val <= 10:
                    val = uts.normalize(val)
                else:
                    raise ValueError(f"{name} must be in range [0, 10]")
            case "sync":
                if 0 <= val <= 10:
                    val = uts.normalize(val)
                else:
                    raise ValueError(f"{name} must be in range [0, 10]")

            case "delay_ms":
                if val <= 0:
                    raise ValueError("delay_ms must be positive")

            case "mode":
                if val not in ("ms", "sync"):
                    raise ValueError("mode must be 'ms' or 'sync'")

            case "bpm":
                if val <= 0:
                    raise ValueError("bpm must be positive")

            case "div":
                if val not in uts.NOTE_DIVISIONS:
                    raise ValueError(f"Invalid note division: {val}")

            case "echo":
                if val is not None and (not isinstance(val, int) or val < 0):
                    raise ValueError("echo must be None or a non-negative integer")

        self.params[name] = val

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def process(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        delay_seconds = self._delay_seconds()
        delay_samples = delay_seconds * self.fs

        feedback = self._feedback_gain()
        mix = self.params["mix"]

        y = np.zeros_like(x)

        for n, sample in enumerate(x):
            delayed = self.delayline.read(delay_samples)

            y[n] = (1.0 - mix) * sample + mix * delayed

            self.delayline.write(sample + feedback * delayed)

        return y
    
class Phaser(Effect):

    # i don't remember any songs using the phaser, sorry.
    # Basically an APF stack. Will come up with a meaningful docstring later.

    def __init__(self, rate, depth, feedback, mix, stages=4, min_freq=300, max_freq=1600, fs=44100):
        super().__init__(fs)
        self.params = {
            "rate": uts.normalize(rate), 
            "depth": uts.normalize(depth),
            "feedback": uts.normalize(feedback),
            "mix": uts.normalize(mix),
            "stages": int(stages),
            "min_freq": float(min_freq),
            "max_freq": float(max_freq)
        }
        self.enabled = False
        self.phase = 0.0
        self.fb_state = 0.0

        self.stages = [flt.FirstOrderAPF(0.0) for _ in range(self.params["stages"])]

    def _coef_from_freq(self, freq):
        freq = np.clip(freq, 20.0, 0.45 * self.fs)
        t = np.tan(np.pi * freq / self.fs)
        return (1.0 - t) / (1.0 + t)

    def _set_stage_count(self, stages):
        stages = int(stages)

        if stages < 1:
            raise ValueError("stages must be >= 1")

        current = len(self.stages)

        if stages > current:
            self.stages.extend(flt.FirstOrderAPF(0.0) for _ in range(stages - current))
        elif stages < current:
            self.stages = self.stages[:stages]

        self.params["stages"] = stages

    def reset(self):
        self.phase = 0.0
        self.fb_state = 0.0

        for stage in self.stages:
            stage.reset()

    def upd_param(self):
        self._set_stage_count(self.params["stages"])

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")

        if name in ("rate", "depth", "feedback", "mix"):
            if 0 <= val <= 10:
                val = uts.normalize(val)
            else:
                raise ValueError(f"{name} must be in range [0, 10]")

            self.params[name] = val
            return

        if name == "stages":
            self._set_stage_count(val)
            return

        if name == "min_freq":
            if val <= 0:
                raise ValueError("min_freq must be positive")
            if val >= self.params["max_freq"]:
                raise ValueError("min_freq must be less than max_freq")
            self.params[name] = float(val)
            return

        if name == "max_freq":
            if val <= self.params["min_freq"]:
                raise ValueError("max_freq must be greater than min_freq")
            if val >= 0.45 * self.fs:
                raise ValueError("max_freq must be safely below Nyquist")
            self.params[name] = float(val)
            return

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def _current_sweep_freq(self):
        depth = self.params["depth"]
        min_f = self.params["min_freq"]
        max_f = self.params["max_freq"]

        # Unipolar LFO: 0..1
        lfo = 0.5 + 0.5 * np.sin(self.phase)

        # If depth is 0, hold at midpoint.
        lfo = 0.5 + depth * (lfo - 0.5)

        # Log-frequency sweep sounds more natural than linear.
        log_min = np.log(min_f)
        log_max = np.log(max_f)

        return np.exp(log_min + lfo * (log_max - log_min))

    def algo(self, x):
        rate = 0.05 + 5.0 * self.params["rate"]
        feedback = 0.95 * self.params["feedback"]
        mix = self.params["mix"]

        sweep_freq = self._current_sweep_freq()
        coef = self._coef_from_freq(sweep_freq)

        for stage in self.stages:
            stage.set_coef(coef)

        # Feedback into the phase-shift network.
        apf_input = x + feedback * self.fb_state

        wet = apf_input

        for stage in self.stages:
            wet = stage.process(wet)

        self.fb_state = wet

        y = (1.0 - mix) * x + mix * wet

        self.phase += 2.0 * np.pi * rate / self.fs
        if self.phase >= 2.0 * np.pi:
            self.phase -= 2.0 * np.pi

        return y

    def process(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        y = np.zeros_like(x)

        for n, sample in enumerate(x):
            y[n] = self.algo(sample)

        return y
        
class Reverb(Effect):
    # the big bad chonky boy, with all his bros and sisters
    """
    Description incoming.
    """
    def __init__(self, mode="schroeder", fs=44100, ir=None, **kwargs):
        super().__init__(fs)
        self.mode = mode
        self.enabled = False

        match mode:
            case "schroeder":
                self.engine = revb.SchroederReverb(fs=fs, **kwargs)
            case "freeverb":
                self.engine = revb.FreeVerb(fs=fs, **kwargs)
            case "convolution":
                if ir is None:
                    raise ValueError("Convolution reverb requires an IR")
                self.engine = revb.ConvolutionReverb(ir=ir, fs=fs, **kwargs)
            case _:
                raise ValueError("mode must be 'schroeder', 'freeverb', or 'convolution'")
            
        self.engine.enabled = True

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def set_param(self, name, val):
        self.engine.set_param(name, val)

    def process(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        return self.engine.process(x)
    
class EQ(Effect):
    # at last we got here. idk no songs for heavy EQ use, but let it be smh
    def __init__(self, fs=44100, prm=True, grph=False, level=10):
        super().__init__(fs)
        self.enabled = False
        self.params = {
            "prm": bool(prm),
            "grph": bool(grph),
            "level": uts.normalize(level)
        }

        # let it be 5 bands only, no time to make more of them

        self.prm_bands = [
            {
                "enabled": True,
                "type": "hpf",
                "fc": 80.0,
                "gain": 0.0,
                "Q": 0.7
            },
            {
                "enabled": True,
                "type": "loshelf",
                "fc": 120.0,
                "gain": 0.0,
                "Q": 0.7
            },
            {
                "enabled": True,
                "type": "peak",
                "fc": 750.0,
                "gain": 0.0,
                "Q": 1.0
            },
            {
                "enabled": True,
                "type": "hishelf",
                "fc": 4000.0,
                "gain": 0.0,
                "Q": 0.7
            },
            {
                "enabled": True,
                "type": "lpf",
                "fc": 12000.0,
                "gain": 0.0,
                "Q": 0.7
            }
        ]

        # graphic eq

        self.grph_freqs = [80, 160, 320, 640, 1250, 2500, 5000]
        self.grph_gains = [0.0 for _ in self.grph_freqs]
        self.grph_Q = 1.0

        self.prm_filters = [flt.Biquad(fs) for _ in self.prm_bands]
        self.grph_filters = [flt.Biquad(fs) for _ in self.grph_freqs]

        # placeholder for filter updates

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
    
    def reset(self):
        for f in self.param_filters:
            f.reset()
        for f in self.graphic_filters:
            f.reset()
    
    def set_param(self, name, val):
        if name == "level":
            if 0 <= val <= 10:
                self.params[name] = uts.normalize(val)
            else:
                raise ValueError("level must be in range [0, 10]")
            return

        if name in ("prm", "grph"):
            if not isinstance(val, bool):
                raise ValueError(f"{name} must be bool")
            self.params[name] = val
            return
    
    def enable_parametric(self, enabled=True):
        self.params["prm"] = bool(enabled)

    def enable_graphic(self, enabled=True):
        self.params["grph"] = bool(enabled)

    def set_parametric_eqband(self, index, *, enabled=None, type=None, fc=None, gain=None, Q=None):
        if not 0 <= index < len(self.param_bands):
            raise IndexError("parametric band index out of range")

        band = self.prm_bands[index]

        if enabled is not None:
            band["enabled"] = bool(enabled)

        if type is not None:
            valid_types = {
                "peak",
                "loshelf",
                "higshelf",
                "lpf",
                "hpf",
            }
            if type not in valid_types:
                raise ValueError(f"type must be one of {valid_types}")
            band["type"] = type

        if fc is not None:
            if not 0 < fc < self.fs / 2:
                raise ValueError("fc must be between 0 and Nyquist")
            band["fc"] = float(fc)

        if gain is not None:
            if not -24 <= gain <= 24:
                raise ValueError("gain must be in range [-24, 24] dB")
            band["gain"] = float(gain)

        if Q is not None:
            if Q <= 0:
                raise ValueError("Q must be positive")
            band["Q"] = float(Q)

        self._update_parametric_filter(index)
    
    def set_graphic_band(self, index, gain):
        if not 0 <= index < len(self.grph_gains):
            raise IndexError("graphic band index out of range")

        if not -24 <= gain <= 24:
            raise ValueError("graphic gain must be in range [-24, 24] dB")

        self.grph_gains[index] = float(gain)
        self._update_graphic_filter(index)

    def set_graphic_gains(self, gains):
        if len(gains) != len(self.grph_gains):
            raise ValueError(
                f"Expected {len(self.grph_gains)} gains, got {len(gains)}"
            )

        for i, gain in enumerate(gains):
            if not -24 <= gain <= 24:
                raise ValueError("graphic gains must be in range [-24, 24] dB")
            self.grph_gains[i] = float(gain)

        self._update_graphic_filters()

    def _update_parametric_filters(self):
        for i in range(len(self.prm_bands)):
            self._update_parametric_filter(i)
    
    def _update_parametric_filter(self, index):
        band = self.prm_bands[index]
        filt = self.prm_filters[index]

        band_type = band["type"]
        fc = band["fc"]
        gain = band["gain"]
        Q = band["Q"]

        match band_type:
            case "peak":
                filt.Peak(fc, gain, Q)
            case "lpf":
                filt.LPF(fc)
            case "hpf":
                filt.HPF(fc)
            case "loshelf":
                filt.Shelf(fc, gain, type="low")
            case "hishelf":
                filt.Shelf(fc, gain, type="high")
            case _:
                raise ValueError(f"Invalid band type: {band_type}")
    
    def _update_graphic_filters(self):
        for i in range(len(self.grph_filters)):
            self._update_graphic_filter(i)

    def _update_graphic_filter(self, index):
        fc = self.grph_freqs[index]
        gain = self.grph_gains[index]
        self.grph_filters[index].Peak(fc, gain, self.grph_Q)

    def _process_sample(self, x):
        y = x

        if self.params["use_parametric"]:
            for band, filt in zip(self.prm_bands, self.prm_filters):
                if band["enabled"]:
                    y = filt.process(y)

        if self.params["use_graphic"]:
            for filt in self.grph_filters:
                y = filt.process(y)

        return uts.volume_gain(self.params["level"]) * y

    def process(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        if x.ndim == 1:
            y = np.zeros_like(x)

            for n, sample in enumerate(x):
                y[n] = self._process_sample(sample)

            return y

        if x.ndim == 2 and x.shape[1] == 2:
            y = np.zeros_like(x)

            # Important: this mono filter state is not ideal for stereo.
            # Better stereo version shown below.
            for n in range(len(x)):
                y[n, 0] = self._process_sample(x[n, 0])
                y[n, 1] = self._process_sample(x[n, 1])

            return y

        raise ValueError("Input must be mono shape (n,) or stereo shape (n, 2)")

class CabSim(Effect):
    # good cabsim = good tone. otherwise nothing, literally NOTHING will save you. you'll be subject to eternal damnation, lol
    def __init__(self, mode="filter", cab="closed_4x12", ir=None, mix=10, level=10, normalize_ir=True, fs=44100):
        super().__init__(fs)
        if mode not in ("filter", "ir"):
            raise ValueError("mode must be 'filter' or 'ir'")

        if cab not in uts.CABSIM_PRESETS:
            raise ValueError(f"Unknown cab preset: {cab}")

        self.params = {
            "mode": mode,
            "cab": cab,
            "mix": uts.normalize(mix),
            "level": uts.normalize(level),
        }

        self.enabled = False

        self.filter_l = flt.CabFiltChain(fs, uts.CABSIM_PRESETS[cab])
        self.filter_r = flt.CabFiltChain(fs, uts.CABSIM_PRESETS[cab])

        self.ir = None
        if ir is not None:
            self.set_ir(ir, normalize_ir=normalize_ir)

        if mode == "ir" and self.ir is None:
            raise ValueError("mode='ir' requires an impulse response")
    
    def reset(self):
        self.filter_l.reset()
        self.filter_r.reset()

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")

        if name == "mode":
            if val not in ("filter", "ir"):
                raise ValueError("mode must be 'filter' or 'ir'")
            if val == "ir" and self.ir is None:
                raise ValueError("Cannot use mode='ir' without loading an IR")
            self.params[name] = val
            return

        if name == "cab":
            if val not in uts.CABSIM_PRESETS:
                raise ValueError(f"Unknown cab preset: {val}")

            self.params[name] = val
            self.filter_l.configure(uts.CABSIM_PRESETS[val])
            self.filter_r.configure(uts.CABSIM_PRESETS[val])
            return

        if name in ("mix", "level"):
            if 0 <= val <= 10:
                self.params[name] = uts.normalize(val)
            else:
                raise ValueError(f"{name} must be in range [0, 10]")
            return
    
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
    
    def _process_filter_mono(self, x):
        y = np.zeros_like(x)

        for n, sample in enumerate(x):
            y[n] = self.filter_l.process(sample)

        return y

    def _process_filter_stereo(self, x):
        y = np.zeros_like(x)

        for n in range(len(x)):
            y[n, 0] = self.filter_l.process(x[n, 0])
            y[n, 1] = self.filter_r.process(x[n, 1])

        return y

    def _process_ir(self, x):
        if self.ir is None:
            raise ValueError("No IR loaded")

        ir = self.ir

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

    def process(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        mode = self.params["mode"]
        mix = self.params["mix"]
        level = uts.volume_gain(self.params["level"])

        if mode == "filter":
            if x.ndim == 1:
                wet = self._process_filter_mono(x)
                return level * ((1.0 - mix) * x + mix * wet)

            if x.ndim == 2 and x.shape[1] == 2:
                wet = self._process_filter_stereo(x)
                return level * ((1.0 - mix) * x + mix * wet)

            raise ValueError("Input must be mono shape (n,) or stereo shape (n, 2)")

        if mode == "ir":
            wet = self._process_ir(x)

            # Mono input + stereo IR gives stereo wet output.
            # In that case, duplicate dry signal for mixing.
            if x.ndim == 1 and wet.ndim == 2:
                dry = np.column_stack([x, x])
            else:
                dry = x

            return level * ((1.0 - mix) * dry + mix * wet)

        raise ValueError(f"Invalid cab sim mode: {mode}")