import numpy as np
import audio.misc.utils as uts

def test_oversampler_identity(oversampler, fs=44100, blocksize=512):
    t = np.arange(blocksize, dtype=np.float32) / fs
    x = 0.2 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    oversampler.reset()

    # Warm up FIR state.
    for _ in range(10):
        y = oversampler.process(x, lambda z: z)

    in_rms = float(np.sqrt(np.mean(x ** 2)))
    out_rms = float(np.sqrt(np.mean(y ** 2)))

    print("input rms :", in_rms)
    print("output rms:", out_rms)
    print("ratio     :", out_rms / (in_rms + 1e-12))
    print("input peak:", float(np.max(np.abs(x))))
    print("output peak:", float(np.max(np.abs(y))))

os2 = uts.Oversampler(factor=2, taps=63, cutoff=0.225)
test_oversampler_identity(os2)

os4 = uts.Oversampler(factor=4, taps=63, cutoff=0.225)
test_oversampler_identity(os4)