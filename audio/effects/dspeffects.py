# This is a DSP quasi-app project, designed as a LinAl final project.
# Structure and config by Taras Duma, Vladyslav Cherniuk, Bohdan Dhzus. 

import numpy as np

import audio.misc.filters as flt
import audio.misc.utils as uts
import audio.effects.reverbs as revb
import audio.misc.jit_optimizer as speed

from audio.effects.cabsim_engines import FilterCabEngine, IRCabEngine
from audio.misc.basics import Effect

# IGNORE; COPY-PASTEABLE CUSTOM LETTERS FOR FORMULAS
# α
# 𝜏
# \(\pi \)

# maybe consider JUCE implementation? porting the stuff to cpp will be HELL tho

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

        self.oversample = 1
        self.oversampler = uts.Oversampler(
            self.oversample,
            taps=63,
            cutoff=0.225
        )
    
    def set_oversampling(self, factor, taps=63):
        if factor not in (1, 2, 4, 8):
            raise ValueError("factor must be 1, 2, 4, or 8")

        self.oversample = factor
        self.oversampler = uts.Oversampler(
            factor=factor,
            taps=taps,
            cutoff=0.225,
        )
    
    def reset(self):
        self.lpf.reset()
        self.hpf.reset()
        self.postclip_hpf.reset()
        self.oversampler.reset()

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
            if val not in ["atan", "tanh", "clear"]: 
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
    
    def _bound(self, x):   # MUST-USE btw. otherwise the final equation might explode
        clip = self.params["clip"]

        match clip:
            case "atan":
                return (2/np.pi) * np.arctan(x)
            case "tanh":
                return np.tanh(x)
            case "clear":
                return x    # do at your own peril
            case _:
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
        
    def process_sample(self, x):
        vol = uts.volume_gain(self.params["level"])

        x = self.hpf.process_block(x)
        x = self._softclip(x)
        x = self.postclip_hpf.process_block(x)
        x = self.lpf.process_block(x)

        return vol * x
    
    def process_block(self, x):
        x = np.asarray(x, dtype=np.float32)

        if not self.enabled:
            return x.copy()

        if x.ndim != 1:
            raise ValueError(f"Overdrive expects mono input shape (n,), got {x.shape}")

        pre = self.hpf.process_block(x)

        shaped = self.oversampler.process(pre, self._softclip)

        shaped = np.asarray(shaped, dtype=np.float32)

        if len(shaped) > len(x):
            shaped = shaped[:len(x)]
        elif len(shaped) < len(x):
            shaped = np.pad(shaped, (0, len(x) - len(shaped)))

        y = self.postclip_hpf.process_block(shaped)
        y = self.lpf.process_block(y)

        vol = uts.volume_gain(self.params["level"])

        return vol * y
    
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
        self.lpf = flt.Biquad(fs)
        self.lpf.LPF(15000)

        self.tone_lpf = flt.Biquad(fs)
        self.tone_lpf.LPF(800+(self.params["tone"]*5200))
        self.tone_hpf = flt.Biquad(fs)
        self.tone_hpf.HPF(800+(self.params["tone"]*5200))

        self.oversample = 1
        self.oversampler = uts.Oversampler(\
            self.oversample,
            taps=63,
            cutoff=0.225
        )
    
    def set_oversampling(self, factor, taps=63):
        if factor not in (1, 2, 4, 8):
            raise ValueError("factor must be 1, 2, 4, or 8")

        self.oversample = factor
        self.oversampler = uts.Oversampler(
            factor=factor,
            taps=taps,
            cutoff=0.225,
        )

    def reset(self):
        self.hpf.reset()
        self.tone_lpf.reset()
        self.tone_hpf.reset()
        self.oversampler.reset()

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
    
    def _seesaw(self, x, tone):
        low = self.tone_lpf.process(x)
        high = self.tone_hpf.process(x)

        return (1 - tone)*low + tone*high
    
    def _tone_selector(self, x):
        filter = self.params["filter"]

        if filter == "classic":
            return self.tone_lpf.process_block(x)
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
        
    def process_sample(self, x):
        
        level = self.params["level"]

        vol = uts.volume_gain(level)

        x = self.hpf.process_block(x) # for them lower grooves
        x = self._hardclip(x) 
        x = self._tone_selector(x)

        return vol * x
    
    def process_block(self, x):
        x = np.asarray(x, dtype=np.float32)

        if not self.enabled:
            return x.copy()

        if x.ndim != 1:
            raise ValueError(f"Distortion expects mono input shape (n,), got {x.shape}")

        pre = self.hpf.process_block(x)

        shaped = self.oversampler.process(pre, self._hardclip)
        shaped = np.asarray(shaped, dtype=np.float32)

        y = self._tone_selector(shaped)
        y = self.lpf.process_block(y)

        vol = uts.volume_gain(self.params["level"])

        return vol * y
    
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

        self.oversample = 1
        self.oversampler = uts.Oversampler(
            self.oversample,
            taps=63,
            cutoff=0.225
        )

        self._filter_upd()
    
    def set_oversampling(self, factor, taps=63):
        if factor not in (1, 2, 4, 8):
            raise ValueError("factor must be 1, 2, 4, or 8")

        self.oversample = factor
        self.oversampler = uts.Oversampler(
            factor=factor,
            taps=taps,
            cutoff=0.225,
        )
    
    def reset(self):
        self.prehpf.reset()
        self.tone_lpf.reset()
        self.tone_hpf.reset()
        self.dc_block.reset()
        self.oversampler.reset()

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
        
    def _hardclip(self, x, T=1.0):
        e = 0.2
        asym = self.params["asym"]
        if asym == True:
            y = np.clip(x+e, -T, T) - e  # asymmetric hardclip
        else:
            y = np.clip(x, -T, T)
        
        return y
    
    def _fuzz_shape(self, x):
        gain = uts.drive_eff(self.params["sustain"], 5, 100)
        gx = gain * x

        return self._hardclip(gx, 1.0)
        
    def _tone_stack(self, x):
        tone = self.params["tone"]

        low = self.tone_lpf.process_block(x)
        high = self.tone_hpf.process_block(x)

        return (1 - tone) * low + tone * high
        
    def process_sample(self, x):
        vol = uts.volume_gain(self.params["level"])

        # also screw your HPF, we going hard
        x = self.prehpf.process_block(x)
        x = self._fuzz_shape(x)
        x = self.dc_block.process(x)
        x = self._tone_stack(x)

        return vol * x

    def process_block(self, x):
        x = np.asarray(x, dtype=np.float32)

        if not self.enabled:
            return x.copy()

        if x.ndim != 1:
            raise ValueError(f"Fuzz expects mono input shape (n,), got {x.shape}")

        pre = self.prehpf.process_block(x)

        shaped = self.oversampler.process(pre, self._fuzz_shape)
        shaped = np.asarray(shaped, dtype=np.float32)

        y = self.dc_block.process_block(shaped)
        y = self._tone_stack(y)

        vol = uts.volume_gain(self.params["level"])

        return vol * y

class BlockAmp(Effect):

    # a bunch of shelf filters + some basic overdrive. THE holy grail for literally every player, EVH used only the block amp for distortion
    # > "Eruption" playing in the background, RIP EVH

    # i should probably allow more customizations to it. after all, it's the most crucial gear piece, like, ever.
    # i wonder how this will tie in with PositiveGrid Spark 2 - i also have one atm. Spark 2 is also supposed to be a block amp. 
    # but it's supposed to go in the middle? how tf am i going to enforce it? i'm gonna play around more to find ts out.
    # new drivers maybe? testing for Spark MINI only now. hit me up if sum shi freaks out.
    # if i manage to crack Diezel VH9 beyond-the-hood mechanics, i'm gonna officially cover Tool istg. no cap ong
    # in retrospection: i implemented the peak biquad filter, but screw it. all hail high+low shelf combo
    # also all hail block amp, it's as crucial as the guitar itself

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

    def __init__(self, drive, tone, bass, mid, treb, pres, vol, fs=44100, oversample=4, clipType="tanh"):
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

        self.prehpf = flt.Biquad(fs)
        self.prehpf.HPF(20)

        self.postdriveLPF = flt.Biquad(fs)
        self.postdriveLPF.LPF(15000)

        self.oversample = 2
        self.oversampler = uts.Oversampler(
            self.oversample, 
            taps=63,
            cutoff=0.225
        )

        self._update_filters()
    
    def set_oversampling(self, factor, taps=63):
        if factor not in (1, 2, 4, 8):
            raise ValueError("factor must be 1, 2, 4, or 8")

        self.oversample = factor
        self.oversampler = uts.Oversampler(
            factor=factor,
            taps=taps,
            cutoff=0.225,
        )

    def reset(self):
        self.toneFilterHI.reset()
        self.toneFilterLO.reset()
        self.bassFilterHI.reset()
        self.bassFilterLO.reset()
        self.midFilterHI.reset()
        self.midFilterLO.reset()
        self.trebFilterHI.reset()
        self.trebFilterLO.reset()
        self.presFilterHI.reset()
        self.presFilterLO.reset()
        
        self.postdriveLPF.reset()
        self.prehpf.reset()
        self.oversampler.reset()

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

        self.toneFilterHI.Shelf(1000, tone_G, flt_type="high")
        self.toneFilterLO.Shelf(1000, tone_G, flt_type="low")

        self.bassFilterHI.Shelf(80, bass_G, flt_type="high")
        self.bassFilterLO.Shelf(120, bass_G, flt_type="low")

        self.midFilterHI.Shelf(500, mid_G, flt_type="high")
        self.midFilterLO.Shelf(900, mid_G, flt_type="low")

        self.trebFilterHI.Shelf(3000, treb_G, flt_type="high")
        self.trebFilterLO.Shelf(5000, treb_G, flt_type="low")

        self.presFilterHI.Shelf(6000, pres_G, flt_type="high")
        self.presFilterLO.Shelf(10000, pres_G, flt_type="low")

    def upd_param(self):
        self._update_filters()

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")

        if name in ("drive", "tone", "bass", "mid", "treb", "pres", "vol"):
            if 0 <= val <= 10:
                self.params[name] = uts.normalize(val)
            else:
                raise ValueError(f"Parameter {name} should be in range [0, 10]")

            if name in ("tone", "bass", "mid", "treb", "pres"):
                self._update_filters()

            return

        if name == "clipType":
            if val not in ["atan", "tanh", "hard", "null"]:
                raise ValueError(
                    "Invalid clipping type. Use 'atan', 'tanh', 'hard', or 'null'."
                )

            self.params[name] = val
            return
        
    # tons of amp-specific peak-shelf filters ahead, nothing much. just stack em up and you have the block amp
    # too lazy to code the peak filter separately. sorry, so just a stack of hi-/lo-shelf filters

    # NOT ACTUAL ANYMORE! I HAVE PEAK BIQUADS. BUT SCREW THEM. STACKED HI-LO SHELVES RULE

    def preamp_gain(self, x):
        """
        Additional/optional gain at the block amp stage. Can be used to further drive distortions or smoothen them out.
        """
        drive = uts.drive_eff(self.params["drive"], 1, 12)
        clipType = self.params["clipType"]

        gx = drive * x

        match clipType:
            case "atan":
                return (2.0 / np.pi) * np.arctan(gx)

            case "tanh":
                return np.tanh(gx)

            case "hard":
                return np.clip(gx, -1.0, 1.0)

            case "null":
                return x     # useful if you don't need more drive

            case _:
                raise ValueError("Invalid clipping type")

    def preamp_tone(self, x):
        """
        General setup for the "tone" knob at the block amp.
        """

        x = self.toneFilterHI.process_block(x)
        x = self.toneFilterLO.process_block(x)

        return x

    def preamp_bass(self, x):

        """
        Peak-filter/boost for the low-end/bass frequencies, typically ranging from 80 to 120 Hz. 
        
        Reference:
        Comparable to a normal or a slightly lower male voice. 

        (Not the actual bass guitar, though; divide the lowest frequencies by 2, then you get into the true bass guitar range. It's one octave lower than your typical 6-string guitar - as long as it makes any musical sense though.)
        """

        x = self.bassFilterHI.process_block(x)
        x = self.bassFilterLO.process_block(x)

        return x

    def preamp_mid(self, x):

        """
        Peak-filter/boost for the mid-range frequencies, often determined to peak at 500-900 Hz.

        Reference:
        Comparable to a slightly higher-pitched children's voice, or the mid-range of a soprano pitch. 
        """

        x = self.midFilterHI.process_block(x)
        x = self.midFilterLO.process_block(x)

        return x

    def preamp_treble(self, x):

        """
        Peak-filter/boost for the high-end/treble frequencies, typically ranging from 3000 to 5000 Hz.

        Reference:
        Comparable to a bright, crisp sound, often associated with the treble range of a guitar or piano.
        Also, this is the mid-range for human consonants in our everyday speech - what makes our language make sense.
        """

        x = self.trebFilterHI.process_block(x)
        x = self.trebFilterLO.process_block(x)

        return x

    def preamp_presence(self, x):

        """
        Peak-filter/boost for the presence frequencies, typically ranging from 6000 to 10000 Hz.

        Reference:
        Comparable to crash cymbals, pinch harmonics(those iconic guitar squeals), or slightly similar to tinnitus. Birds may often chirp your ears off at such frequencies.
        """

        x = self.presFilterHI.process_block(x)
        x = self.presFilterLO.process_block(x)

        return x

    def process_sample(self, x):
        
        vol = self.params["vol"]

        x = self.preamp_gain(x)
        x = self.postdriveLPF.process_block(x)    # i found out that it's quite crackly. well, f**k me.

        x = self.preamp_tone(x)
        x = self.preamp_bass(x)
        x = self.preamp_mid(x)
        x = self.preamp_treble(x)
        x = self.preamp_presence(x)

        vol = uts.volume_gain(vol)

        return vol * x
    
    def process_block(self, x):
        x = np.asarray(x, dtype=np.float32)

        if not self.enabled:
            return x.copy()

        if x.ndim != 1:
            raise ValueError(
                f"BlockAmp expects mono input shape (n,), got {x.shape}"
            )

        pre = self.prehpf.process_block(x)

        shaped = self.oversampler.process(pre, self.preamp_gain)
        shaped = np.asarray(shaped, dtype=np.float32)

        if len(shaped) > len(x):
            shaped = shaped[:len(x)]

        if len(shaped) < len(x):
            shaped = np.pad(shaped, (0, len(x) - len(shaped)))

        y = self.postdriveLPF.process_block(shaped)

        y = self.preamp_tone(y)
        y = self.preamp_bass(y)
        y = self.preamp_mid(y)
        y = self.preamp_treble(y)
        y = self.preamp_presence(y)

        vol = uts.volume_gain(self.params["vol"])

        return vol * y

class Flanger(Effect):

    # STIR US FROM OUR
    # WANTON SLUMBER
    # MITIGATE THE RUIN, CALL US ALL TO ARMS AND ORDER

    # should i add a triangle LFO???

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
        delay: Base delay duration scaler, normalized to [0, 1].
        fs: Sample frequency, default at 44100 Hz.
    """

    real_time_safe = True

    def __init__(self, rate=2, depth=2, feedback=0, mix=3, delay=5, fs=44100):
        super().__init__(fs)

        self.params = {
            "rate": uts.normalize(rate),
            "depth": uts.normalize(depth),
            "feedback": uts.normalize(feedback),
            "mix": uts.normalize(mix),
            "delay": uts.normalize(delay),
        }

        self.buffer = np.zeros(int(0.03 * fs) + 4, dtype=np.float32)
        self.write_idx = 0
        self.phase = 0.0

    def reset(self):
        self.buffer.fill(0.0)
        self.write_idx = 0
        self.phase = 0.0

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid Flanger parameter: {name}")

        if not 0 <= val <= 10:
            raise ValueError(f"{name} must be in range [0, 10]")

        self.params[name] = uts.normalize(val)

    def process_block(self, x):
        x = np.asarray(x, dtype=np.float32)

        if not self.enabled:
            return x.copy()

        if x.ndim != 1:
            raise ValueError(f"FastFlanger expects mono input shape (n,), got {x.shape}")

        y, self.write_idx, self.phase = speed.fast_flanger_kernel(
            x,
            self.buffer,
            self.write_idx,
            self.phase,
            self.fs,
            self.params["rate"],
            self.params["depth"],
            self.params["feedback"],
            self.params["mix"],
            self.params["delay"],
        )

        return y
    
class Chorus(Effect):

    # TO ASSUME FROM IGNORANCE
    # INFLICTING WOUNDS WITH YOUR CROSS TURNED DAGGER

    # LITERALLY FLANGER BUT WITH A FEW TWEAKS

    r"""
    Using the same formula as the flanger, the chorus, as I like to call it, is a more 'vocal' and slower version of the flanger.
    The two delay lines used are usually independent from each other, creating a stereo-like effect. Chuck Schuldiner(RIP) liked to use it on his solos.

    Equation: 

       y[t] = x[t] + αx[t - 𝜏[t]] 
       𝜏[t] = 𝜏_0 + d * sin(2πf_{LFO}*t)
    α is the mix parameter, 𝜏_0 is the base delay, d is the depth of modulation, and LFO is a low-frequency oscillator (sine wave) that modulates the delay time.

    Args:
        rate: LFO rate, the speed of the sweeping effect. Input: [0, 10], scaled down to [0, 1] and then to [0.05, 5] Hz.
        depth: LFO depth, the amount of delay modulation. Input: [0, 10], scaled down to [0, 1] and then to [0.001, 0.005] seconds.
        feedback: Feedback amount, the amount of output signal fed back into the input. Input: [0, 10], scaled down to [0, 1] and then to [0, 0.95].
        delay: Base delay duration scaler, normalized to [0, 1]. (A bit less aggressive than in the flanger effect.)
        mix: Mix parameter, the balance between the dry and wet signals. Input: [0, 10], scaled down to [0, 1].
        fs: Sample frequency, default at 44100 Hz.
    """
    real_time_safe = True

    def __init__(self, rate, depth, feedback=0, mix=3, delay=5, fs=44100):
        super().__init__(fs)

        self.params = {
            "rate": uts.normalize(rate),
            "depth": uts.normalize(depth),
            "feedback": uts.normalize(feedback),  # unused for now
            "mix": uts.normalize(mix),
            "delay": uts.normalize(delay),
        }

        self.buffer = np.zeros(int(0.08 * fs) + 4, dtype=np.float32)
        self.write_idx = 0
        self.phase = 0.0

    def reset(self):
        self.buffer.fill(0.0)
        self.write_idx = 0
        self.phase = 0.0

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid Chorus parameter: {name}")

        if not 0 <= val <= 10:
            raise ValueError(f"{name} must be in range [0, 10]")

        self.params[name] = uts.normalize(val)

    def process_block(self, x):
        x = np.asarray(x, dtype=np.float32)

        if not self.enabled:
            return x.copy()

        if x.ndim != 1:
            raise ValueError(f"FastChorus expects mono input shape (n,), got {x.shape}")

        y, self.write_idx, self.phase = speed.chorus_kernel(
            x,
            self.buffer,
            self.write_idx,
            self.phase,
            self.fs,
            self.params["rate"],
            self.params["depth"],
            self.params["mix"],
            self.params["delay"],
        )

        return y

class Delay(Effect):

    # same as flanger - but delay is much longer, also no modulation and no mix

    # black then white are all I see in my infancy
    # > proceeds to change the time signature 11235813 times
    # > F E T T U C C I N E  S E Q U E N C E

    real_time_safe = True

    def __init__(
        self,
        feedback=2,
        mix=2,
        delay_ms=300,
        mode="ms",
        bpm=120,
        div="1/4",
        echo=None,
        max_delay_seconds=3.0,
        fs=44100,
    ):
        super().__init__(fs)

        self.params = {
            "feedback": uts.normalize(feedback),
            "mix": uts.normalize(mix),
            "delay_ms": float(delay_ms),
            "mode": mode,
            "bpm": float(bpm),
            "div": div,
            "echo": echo,
        }

        self.buffer = np.zeros(
            int(np.ceil(max_delay_seconds * fs)) + 4,
            dtype=np.float32,
        )
        self.write_idx = 0

    def reset(self):
        self.buffer.fill(0.0)
        self.write_idx = 0

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid Delay parameter: {name}")

        if name in ("feedback", "mix"):
            if not 0 <= val <= 10:
                raise ValueError(f"{name} must be in range [0, 10]")
            self.params[name] = uts.normalize(val)
            return

        if name == "delay_ms":
            if val <= 0:
                raise ValueError("delay_ms must be positive")
            self.params[name] = float(val)
            return

        if name == "mode":
            if val not in ("ms", "sync"):
                raise ValueError("mode must be 'ms' or 'sync'")
            self.params[name] = val
            return

        if name == "bpm":
            if val <= 0:
                raise ValueError("bpm must be positive")
            self.params[name] = float(val)
            return

        if name == "div":
            self.params[name] = val
            return

        if name == "echo":
            self.params[name] = val
            return

    def _delay_seconds(self):
        if self.params["mode"] == "ms":
            return self.params["delay_ms"] / 1000.0

        if self.params["mode"] == "sync":
            return uts.bpm_to_delay_seconds(
                self.params["bpm"],
                self.params["div"],
            )

        raise ValueError(f"Invalid delay mode: {self.params['mode']}")

    def _feedback_gain(self):
        echo = self.params["echo"]

        if echo is not None:
            return uts.feedback_from_echoes(echo)

        # Safer max feedback.
        return 0.75 * self.params["feedback"]

    def process_block(self, x):
        x = np.asarray(x, dtype=np.float32)

        if not self.enabled:
            return x.copy()

        if x.ndim != 1:
            raise ValueError(f"FastDelay expects mono input shape (n,), got {x.shape}")

        delay_samples = self._delay_seconds() * self.fs
        feedback = self._feedback_gain()

        # Cap max wet at 60% for guitar.
        mix = 0.6 * self.params["mix"]

        y, self.write_idx = speed.fast_delay_kernel(
            x,
            self.buffer,
            self.write_idx,
            delay_samples,
            feedback,
            mix,
        )

        return y
    
class Phaser(Effect):

    # i don't remember any songs using the phaser, sorry.
    # Basically an APF stack. Will come up with a meaningful docstring later.
    real_time_safe = True

    def __init__(
        self,
        rate,
        depth,
        feedback,
        mix,
        stages=4,
        min_freq=300,
        max_freq=1600,
        fs=44100,
    ):
        super().__init__(fs)

        self.params = {
            "rate": uts.normalize(rate),
            "depth": uts.normalize(depth),
            "feedback": uts.normalize(feedback),
            "mix": uts.normalize(mix),
            "stages": int(stages),
            "min_freq": float(min_freq),
            "max_freq": float(max_freq),
        }

        self.apf_states = np.zeros(int(stages), dtype=np.float32)

        self.phase = 0.0
        self.fb_state = 0.0

        mid_freq = np.sqrt(float(min_freq) * float(max_freq))
        self.current_coef = speed.phaser_coef_from_freq(mid_freq, fs)

        self.control_div = 4
        self.control_counter = 0

    def reset(self):
        self.apf_states.fill(0.0)
        self.phase = 0.0
        self.fb_state = 0.0
        self.control_counter = 0

    def _set_stage_count(self, stages):
        stages = int(stages)

        if stages < 1:
            raise ValueError("stages must be >= 1")

        old = self.apf_states
        new = np.zeros(stages, dtype=np.float32)

        n = min(len(old), stages)
        new[:n] = old[:n]

        self.apf_states = new
        self.params["stages"] = stages

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid Phaser parameter: {name}")

        if name in ("rate", "depth", "feedback", "mix"):
            if not 0 <= val <= 10:
                raise ValueError(f"{name} must be in range [0, 10]")
            self.params[name] = uts.normalize(val)
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

    def process_block(self, x):
        x = np.asarray(x, dtype=np.float32)

        if not self.enabled:
            return x.copy()

        if x.ndim != 1:
            raise ValueError(f"FastPhaser expects mono input shape (n,), got {x.shape}")

        (
            y,
            self.phase,
            self.fb_state,
            self.current_coef,
            self.control_counter,
        ) = speed.phaser_kernel(
            x,
            self.apf_states,
            self.phase,
            self.fb_state,
            self.current_coef,
            self.control_counter,
            self.fs,
            self.params["rate"],
            self.params["depth"],
            self.params["feedback"],
            self.params["mix"],
            self.params["min_freq"],
            self.params["max_freq"],
            self.control_div,
        )

        return y
        
class Reverb(Effect):
    # the big bad chonky boy, with all his bros and sisters
    """
    Description incoming.
    """
    def __init__(self, mode="schroeder", mix=2, level=10, fs=44100, ir=None, **kwargs):
        super().__init__(fs)

        if mode not in ("schroeder", "lite", "freeverb", "convolution"):
            raise ValueError("mode must be 'schroeder', 'freeverb', or 'convolution'")

        if not 0 <= mix <= 10:
            raise ValueError("mix must be in range [0, 10]")

        if not 0 <= level <= 10:
            raise ValueError("level must be in range [0, 10]")

        self.params = {
            "mode": mode,
            "mix": uts.normalize(mix),
            "level": uts.normalize(level),
        }

        self.ir = ir
        self.engine_kwargs = kwargs
        self.engine = self._make_engine(mode)

    @property
    def real_time_safe(self):
        return getattr(self.engine, "real_time_safe", True)

    def _make_engine(self, mode):
        if mode == "schroeder":
            return revb.SchroederReverb(fs=self.fs, **self.engine_kwargs)

        if mode == "freeverb":
            return revb.FreeVerb(fs=self.fs, **self.engine_kwargs)
        
        if mode == "lite":
            return revb.LiteSchroederReverb(fs=self.fs, **self.engine_kwargs)

        if mode == "convolution":
            if self.ir is None:
                raise ValueError("Convolution reverb requires an IR")
            return revb.ConvolutionReverb(ir=self.ir, fs=self.fs, **self.engine_kwargs)

        raise ValueError(f"Invalid reverb mode: {mode}")

    def reset(self):
        self.engine.reset()

    def set_param(self, name, val):
        if name == "mode":
            if val not in ("schroeder", "freeverb", "convolution"):
                raise ValueError("mode must be 'schroeder', 'freeverb', or 'convolution'")

            if val == "convolution" and self.ir is None:
                raise ValueError("Cannot switch to convolution mode without an IR")

            self.params["mode"] = val
            self.engine = self._make_engine(val)
            return

        if name in ("mix", "level"):
            if not 0 <= val <= 10:
                raise ValueError(f"{name} must be in range [0, 10]")
            self.params[name] = uts.normalize(val)
            return

        self.engine.set_param(name, val)

    def process_block(self, x):
        x = np.asarray(x, dtype=np.float32)

        if not self.enabled:
            return x.copy()

        wet = self.engine.process_block(x)

        # Max 40% wet for debugging.
        mix = 0.4 * self.params["mix"]

        # Direct normalized output level.
        level = self.params["level"]

        if x.ndim == 1 and wet.ndim == 2:
            dry = np.column_stack([x, x])
        else:
            dry = x
        
        wet_mix = 0.20 * self.params["mix"]
        wet_level = self.params["level"]

        y = dry + wet_mix * wet_level * wet

        return level * y
    
class EQ(Effect):
    # at last we got here. idk no songs for heavy EQ use, but let it be smh
    def __init__(self, fs=44100, param=True, grph=False, level=10):
        super().__init__(fs)
        self.enabled = False
        self.params = {
            "param": bool(param),
            "grph": bool(grph),
            "level": uts.normalize(level)
        }

        # let it be 5 bands only, no time to make more of them

        self.param_bands = [
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

        self.param_filters = [flt.Biquad(fs) for _ in self.param_bands]
        self.grph_filters = [flt.Biquad(fs) for _ in self.grph_freqs]

        # placeholder for filter updates

        self._update_parametric_filters()
        self._update_graphic_filters()
    
    def reset(self):
        for f in self.param_filters:
            f.reset()
        for f in self.grph_filters:
            f.reset()
    
    def set_param(self, name, val):
        if name == "level":
            if 0 <= val <= 10:
                self.params[name] = uts.normalize(val)
            else:
                raise ValueError("level must be in range [0, 10]")
            return

        if name in ("param", "grph"):
            if not isinstance(val, bool):
                raise ValueError(f"{name} must be bool")
            self.params[name] = val
            return
    
    def enable_parametric(self, enabled=True):
        self.params["param"] = bool(enabled)

    def enable_graphic(self, enabled=True):
        self.params["grph"] = bool(enabled)

    def set_parametric_eqband(self, index, *, enabled=None, type=None, fc=None, gain=None, Q=None):
        if not 0 <= index < len(self.param_bands):
            raise IndexError("parametric band index out of range")

        band = self.param_bands[index]

        if enabled is not None:
            band["enabled"] = bool(enabled)

        if type is not None:
            valid_types = {
                "peak",
                "loshelf",
                "hishelf",
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
        for i in range(len(self.param_bands)):
            self._update_parametric_filter(i)
    
    def _update_parametric_filter(self, index):
        band = self.param_bands[index]
        filt = self.param_filters[index]

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
                filt.Shelf(fc, gain, flt_type="low")
            case "hishelf":
                filt.Shelf(fc, gain, flt_type="high")
            case _:
                raise ValueError(f"Invalid band type: {band_type}")
    
    def _update_graphic_filters(self):
        for i in range(len(self.grph_filters)):
            self._update_graphic_filter(i)

    def _update_graphic_filter(self, index):
        fc = self.grph_freqs[index]
        gain = self.grph_gains[index]
        self.grph_filters[index].Peak(fc, gain, self.grph_Q)

    def process_block(self, x):
        x = np.asarray(x, dtype=np.float32)

        if not self.enabled:
            return x.copy()

        if x.ndim != 1:
            raise ValueError(f"EQ expects mono input shape (n,), got {x.shape}")

        y = x.copy()

        if self.params["param"]:
            for band, filt in zip(self.param_bands, self.param_filters):
                if band["enabled"]:
                    y = filt.process_block(y)

        if self.params["grph"]:
            for gain, filt in zip(self.grph_gains, self.grph_filters):
                if gain != 0.0:
                    y = filt.process_block(y)

        level = uts.volume_gain(self.params["level"])

        return level * y

class CabSim(Effect):
    # good cabsim = good tone. otherwise nothing, literally NOTHING will save you. you'll be subject to eternal damnation, lol
    def __init__(self, mode="filter", cab="closed_4x12", ir=None, mix=10, level=10, normalize_ir=True, fs=44100,):
        super().__init__(fs)

        if mode not in ("filter", "ir"):
            raise ValueError("mode must be 'filter' or 'ir'")

        if cab not in uts.CABSIM_PRESETS:
            raise ValueError(f"Unknown cab preset: {cab}")

        if mode == "ir" and ir is None:
            raise ValueError("mode='ir' requires an impulse response")

        self.params = {
            "mode": mode,
            "cab": cab,
            "mix": uts.normalize(mix),
            "level": uts.normalize(level),
        }

        self._ir = None
        self._normalize_ir = normalize_ir

        if ir is not None:
            self._ir = np.asarray(ir, dtype=float)

        self.engine = self._make_engine(mode)

    @property
    def real_time_safe(self):
        return getattr(self.engine, "real_time_safe", True)

    def _make_engine(self, mode):
        if mode == "filter":
            return FilterCabEngine(
                fs=self.fs,
                cab=self.params["cab"],
            )

        if mode == "ir":
            if self._ir is None:
                raise ValueError("Cannot create IR cab engine without an IR")

            return IRCabEngine(
                ir=self._ir,
                normalize_ir=self._normalize_ir,
                fs=self.fs,
            )

        raise ValueError(f"Invalid cab sim mode: {mode}")

    def reset(self):
        self.engine.reset()

    def set_param(self, name, val):
        if name not in self.params:
            raise ValueError(f"Invalid parameter name: {name}")

        if name == "mode":
            if val not in ("filter", "ir"):
                raise ValueError("mode must be 'filter' or 'ir'")

            if val == "ir" and self._ir is None:
                raise ValueError("Cannot use mode='ir' without loading an IR")

            if val != self.params["mode"]:
                self.params["mode"] = val
                self.engine = self._make_engine(val)

            return

        if name == "cab":
            if val not in uts.CABSIM_PRESETS:
                raise ValueError(f"Unknown cab preset: {val}")

            self.params["cab"] = val

            if self.params["mode"] == "filter":
                self.engine.set_cab(val)

            return

        if name in ("mix", "level"):
            if 0 <= val <= 10:
                self.params[name] = uts.normalize(val)
            else:
                raise ValueError(f"{name} must be in range [0, 10]")

            return

    def set_ir(self, ir, normalize_ir=True):
        self._ir = np.asarray(ir, dtype=float)
        self._normalize_ir = normalize_ir

        if self.params["mode"] == "ir":
            self.engine = IRCabEngine(
                ir=self._ir,
                normalize_ir=normalize_ir,
                fs=self.fs,
            )

    def process_sample(self, x):
        if self.params["mode"] != "filter":
            raise RuntimeError("CabSim.process_sample() only works in filter mode")

        wet = self.engine.process_sample(x)

        mix = self.params["mix"]
        level = uts.volume_gain(self.params["level"])

        return level * ((1.0 - mix) * x + mix * wet)

    def process_block(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        wet = self.engine.process_block(x)

        mix = self.params["mix"]
        level = uts.volume_gain(self.params["level"])

        # Mono dry + stereo wet, for mono input + stereo IR.
        if x.ndim == 1 and wet.ndim == 2:
            dry = np.column_stack([x, x])
        else:
            dry = x

        return level * ((1.0 - mix) * dry + mix * wet)
    
class GainTrim(Effect):
    real_time_safe = True

    def __init__(self, gain=1.0, fs=44100):
        super().__init__(fs)
        self.gain = float(gain)
        self.enabled = True

    def reset(self):
        pass

    def set_param(self, name, val):
        if name != "gain":
            raise ValueError("Invalid parameter name")
        self.gain = float(val)

    def process_sample(self, x):
        return self.gain * x