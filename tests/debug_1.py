import numpy as np
from audio.effects.dspeffects import (
    Overdrive,
    Distortion, 
    Flanger,
    Compressor,
    Fuzz,
    BlockAmp,
    Delay,
    Chorus,
    Phaser,
    Reverb,
    CabSim,
    EQ,
    GainTrim,
)
from audio.misc.basics import EffectChain

FS = 44100
BLOCKSIZE = 256

class DebugEffectChain:
    def __init__(self, effects):
        self.effects = effects

    def reset(self):
        for fx in self.effects:
            fx.reset()

    def process_block(self, x):
        y = x

        for fx in self.effects:
            y = fx.process_block(y)

            if y is None:
                raise RuntimeError(
                    f"{fx.__class__.__name__} returned None"
                )

            y = np.asarray(y, dtype=np.float32)

            if not np.all(np.isfinite(y)):
                raise RuntimeError(
                    f"{fx.__class__.__name__} produced NaN/Inf"
                )

            peak = float(np.max(np.abs(y)))
            rms = float(np.sqrt(np.mean(y ** 2)))

            print(
                f"{fx.__class__.__name__:20s}",
                "peak:",
                round(peak, 4),
                "rms:",
                round(rms, 4),
            )

        return y
    
comp = Compressor(
    sens=7,
    level=6,
    fs=FS,
)
comp.enabled=True

dist = Distortion(
    dist=6,
    tone=5,
    level=7,
    mode="cubic",
    fs=FS,
)
dist.enabled = True

od = Overdrive(
    drive=6,
    tone=4,
    level=8,
    mode="qcubic",
    clip="atan",
    fs=FS,
)
od.enabled = True
od.oversample = 1

flg = Flanger(
    rate=2,
    depth=6,
    feedback=6,
    mix=3
)
flg.enabled=True

fuzz = Fuzz(
    sustain=5, 
    tone=6,
    level=4,
    asym=True,
    fs=FS,
)
fuzz.enabled=True

amp = BlockAmp(
    drive=8,
    tone=5,
    bass=3,
    mid=6.4,
    treb=4.4,
    pres=2,
    vol=8.3,
    clipType="hard",
    fs=FS,
)
amp.enabled=True

delay = Delay(
    feedback=3,
    mix=4,
    mode="sync",
    bpm=111,
    div="1/4",
)
delay.enabled = True

chorus = Chorus(
    rate=6,
    depth=3,
    feedback=0,
    mix=4,
)
chorus.enabled = True

phaser = Phaser(
    rate=5,
    depth=2,
    feedback=3,
    mix=6,
)
phaser.enabled = True

reverb = Reverb(
    mode="lite",
    room_size=4,
    damping=5,
    mix=6,
    level=6,
    fs=44100,
)

reverb.enabled = True

eq = EQ(
    param=True,
    grph=False,
    level=8,
)
eq.enabled = True

cab = CabSim(
    mode="filter",
    cab="closed_4x12",
    mix=10,
    level=7,
)
cab.enabled = True

trim1 = GainTrim(
    gain=0.15,
)

trim2 = GainTrim(
    gain=0.2,
)

trim3 = GainTrim(
    gain=0.7,
)

chain = EffectChain([comp, fuzz, amp, cab, eq, chorus, delay, reverb, ])
chain.reset()

debug_chain = DebugEffectChain(chain.effects)

x = np.random.default_rng(0).normal(
    scale=0.05,
    size=256,
).astype(np.float32)

debug_chain.reset()
y = debug_chain.process_block(x)