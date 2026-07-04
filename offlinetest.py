import numpy as np

from dspeffects import Distortion, Flanger
from basics import EffectChain

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

chain = EffectChain([dist, flg])
chain.reset()

x = np.random.default_rng(0).normal(scale=0.05, size=64).astype(np.float32)

print("input:", x.shape, x.ndim)

y1 = dist.process_block(x)
print("after dist:", y1.shape, y1.ndim)

y2 = flg.process_block(y1)
print("after flanger:", y2.shape, y2.ndim)

y3 = chain.process_block(x)
print("after chain:", y3.shape, y3.ndim)