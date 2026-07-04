import os
os.environ["SD_ENABLE_ASIO"] = "1"

import numpy as np
import sounddevice as sd

from dspeffects import Distortion, Flanger
from basics import EffectChain


FS = 44100
BLOCKSIZE = 64


INPUT_DEVICE = "Positive Grid USB Audio Device"
OUTPUT_DEVICE = "Positive Grid USB Audio Device"


LATENCY = (0.00001, 0.000001)

dist = Distortion(
    dist=4,
    tone=5,
    level=7,
    mode="cubic",
    fs=FS,
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

# Optional: warm up/reset before starting.
if hasattr(dist, "reset"):
    dist.reset()


def callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    # Mono input from Spark.
    x = indata[:, 0].astype(np.float32)

    # Process through overdrive.
    # Use process_block if you've refactored to the new parent class.
    y = chain.process_block(x)

    # Safety output limiter.
    y = np.clip(y, -1.0, 1.0).astype(np.float32)

    # Duplicate mono processed signal to stereo output.
    outdata[:, 0] = y
    outdata[:, 1] = y

    if outdata.shape[1] > 1:
        outdata[:, 1] = y


stream = sd.Stream(
    samplerate=FS,
    blocksize=BLOCKSIZE,
    latency=LATENCY,
    dtype="float32",
    channels=(1, 2),
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    callback=callback,
)

with stream:
    print("Running Overdrive test. Press Ctrl+C to stop.")
    print("Actual stream latency:", stream.latency)

    while True:
        sd.sleep(1000)