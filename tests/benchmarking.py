import time
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
BLOCKSIZE = 512

def benchmark_effects_in_chain(chain, fs=44100, blocksize=256, n_blocks=2000):
    x = np.random.default_rng(0).normal(
        scale=0.05,
        size=blocksize,
    ).astype(np.float32)

    chain.reset()

    # Warmup.
    for _ in range(20):
        chain.process_block(x)

    effect_times = {
        fx.__class__.__name__: []
        for fx in chain.effects
    }

    total_times = []

    for _ in range(n_blocks):
        y = x

        t_chain0 = time.perf_counter()

        for fx in chain.effects:
            name = fx.__class__.__name__

            t0 = time.perf_counter()
            y = fx.process_block(y)
            t1 = time.perf_counter()

            effect_times[name].append((t1 - t0) * 1000)

        t_chain1 = time.perf_counter()
        total_times.append((t_chain1 - t_chain0) * 1000)

    deadline_ms = 1000.0 * blocksize / fs

    print("Callback deadline:", deadline_ms, "ms")
    print()

    print("TOTAL CHAIN")
    total_times = np.asarray(total_times)
    print("mean:", np.mean(total_times))
    print("p95 :", np.percentile(total_times, 95))
    print("p99 :", np.percentile(total_times, 99))
    print("max :", np.max(total_times))
    print()

    print("PER EFFECT")
    for name, times in effect_times.items():
        times = np.asarray(times)

        print(
            f"{name:20s}",
            "mean:", round(float(np.mean(times)), 4),
            "p99:", round(float(np.percentile(times, 99)), 4),
            "max:", round(float(np.max(times)), 4),
        )

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

chain = EffectChain([comp, od, eq, delay, flg, amp, ])
chain.reset()

benchmark_effects_in_chain(chain, fs=FS, blocksize=BLOCKSIZE)