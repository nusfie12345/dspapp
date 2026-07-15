import os
os.environ["SD_ENABLE_ASIO"] = "1"

import numpy as np
import sounddevice as sd

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

import gc


FS = 44100
BLOCKSIZE = 256


INPUT_DEVICE = "Positive Grid USB Audio Device"
OUTPUT_DEVICE = "Positive Grid USB Audio Device"

def soft_limiter(x, ceiling=0.98):
    return ceiling * np.tanh(x / ceiling)

LATENCY = (0.00001, 0.000001)

comp = Compressor(
    sens=7,
    level=6,
    fs=FS,
)
comp.enabled=True

dist = Distortion(
    dist=8,
    tone=5,
    level=7,
    mode="cubic",
    fs=FS,
)
dist.enabled = True

od = Overdrive(
    drive=8,
    tone=7,
    level=8,
    mode="clear",
    clip="atan",
    fs=FS,
)
od.enabled = True

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
    level=7,
    asym=True,
    fs=FS,
)
fuzz.enabled=True

amp = BlockAmp(
    drive=8,
    tone=5,
    bass=3,
    mid=8,
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
    bpm=133,
    div="1/8",
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
    depth=3,
    feedback=4,
    mix=6,
)
phaser.enabled = True

reverb = Reverb(
    mode="lite",
    room_size=4,
    damping=5,
    mix=6,
    level=7,
    fs=44100,
)

reverb.enabled = True

eq = EQ(
    param=True,
    grph=False,
    level=8,
)
eq.enabled = True
eq.set_parametric_eqband(
    1, 
    enabled=True,
    gain=2,
)

cab = CabSim(
    mode="filter",
    cab="closed_4x12",
    mix=10,
    level=8,
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

chain = EffectChain([comp, od, eq, delay, flg, amp, ])
chain.reset()

# running twice to "heat up" the kernels

dummy = np.zeros(BLOCKSIZE, dtype=np.float32)
chain.process_block(dummy)
chain.process_block(dummy)

# Optional: warm up/reset before starting.
for fx in chain.effects:
    fx.reset()

print("effects in chain:", [fx.__class__.__name__ for fx in chain.effects])

callback_status_count = 0
callback_error_count = 0


def callback(indata, outdata, frames, time, status):
    global callback_status_count, callback_error_count

    try:
        if status:
            callback_status_count += 1

        x = indata[:, 0].astype(np.float32)

        y = chain.process_block(x)

        y = np.asarray(y, dtype=np.float32)

        if not np.all(np.isfinite(y)):
            callback_error_count += 1
            outdata.fill(0)
            return

        # Use emergency clipping/limiting only after gain is controlled.
        y = np.clip(y, -0.98, 0.98).astype(np.float32)

        if y.ndim == 1:
            outdata[:, 0] = y
            if outdata.shape[1] > 1:
                outdata[:, 1] = y

        elif y.ndim == 2 and y.shape[1] == 2:
            outdata[:, 0] = y[:, 0]
            if outdata.shape[1] > 1:
                outdata[:, 1] = y[:, 1]

        else:
            callback_error_count += 1
            outdata.fill(0)

    except Exception:
        callback_error_count += 1
        outdata.fill(0)

stream = sd.Stream(
    samplerate=FS,
    blocksize=BLOCKSIZE,
    latency=LATENCY,
    dtype="float32",
    channels=(1, 2),
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    callback=callback,
)

gc.disable()
    
try:
    with stream:
        print("Running Overdrive test. Press Ctrl+C to stop.")
        print("Actual stream latency:", stream.latency)
        while True:
            sd.sleep(1000)
            print(
                "status_count:",
                callback_status_count,
                "error_count:",
                callback_error_count,
            )

finally:
    gc.enable()
