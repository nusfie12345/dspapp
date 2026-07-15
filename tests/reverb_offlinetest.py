import numpy as np
from audio.effects.dspeffects import Reverb

from audio.effects.reverbs import ZeroWetEngine

reverb = Reverb(
    mode="schroeder",
    room_size=3,
    damping=7,
    mix=2,
    level=10,
    fs=44100,
)

reverb.enabled = True

x = np.random.default_rng(0).normal(scale=0.05, size=1024).astype(np.float32)
y = reverb.process_block(x)

print(type(y))
print(y.shape)
print(np.all(np.isfinite(y)))
print(np.max(np.abs(y)))