import numpy as np

from audio.effects.dspeffects import (
    Distortion, 
    Flanger, 
    Reverb,
)
from audio.misc.basics import EffectChain

dist = Distortion(
    dist=4,
    tone=5,
    level=7,
    mode="cubic",
    fs=44100,
)
dist.enabled = True

flg = Flanger(
    rate=2,
    depth=9,
    feedback=6,
    mix=3
)
flg.enabled=True

reverb = Reverb(
    mode="schroeder",
    room_size=5,
    damping=4,
    mix=3,
    level=6,
    fs=44100,
)
reverb.enabled = True


chain = EffectChain([dist, flg, reverb])
chain.reset()

x = np.random.default_rng(0).normal(scale=0.05, size=64).astype(np.float32)

print("input:", x.shape, x.ndim)

y1 = dist.process_block(x)
print("after dist:", y1.shape, y1.ndim)

y2 = flg.process_block(y1)
print("after flanger:", y2.shape, y2.ndim)

y3 = reverb.process_block(y2)
print("after reverb:", y3.shape, y3.ndim)

y4 = chain.process_block(x)
print("after chain:", y4.shape, y4.ndim)



# reverb issues

print(y3.shape)
print(np.all(np.isfinite(y3)))
print(np.max(np.abs(y3)))
print(reverb.real_time_safe)