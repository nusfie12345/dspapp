# This file holds the basic classes Effect and EffectChain.
# Therefore, basics of everything. Lmao.
# FYI:
# - everything in /dspeffects.py inherits from Effect;
# - everything ties into EffectChain.

import numpy as np

class Effect:
    real_time_safe = True

    def __init__(self, fs):
        self.fs = fs
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
    
    def reset(self):
        pass
    
    def set_param(self, name, val):
        raise NotImplementedError

    def process_sample(self, x):
        return x

    def process_block(self, x):
        x = np.asarray(x, dtype=float)

        if not self.enabled:
            return x.copy()

        if x.ndim == 0:
            raise ValueError(
                f"{self.__class__.__name__}.process_block() received a scalar. "
                "Use process_sample() for one sample, or pass a 1-D block."
            )

        if x.ndim != 1:
            raise ValueError(
                f"{self.__class__.__name__} uses the default mono process_block(). "
                "Override process_block() for stereo or shape-changing effects."
            )

        y = np.zeros_like(x)

        for n, sample in enumerate(x):
            y[n] = self.process_sample(sample)

        return y

    def process(self, x):
        return self.process_block(x)

class EffectChain:
    def __init__(self, effects):
        self.effects = effects

    def reset(self):
        for fx in self.effects:
            if hasattr(fx, "reset"):
                fx.reset()

    def process_block(self, x):
        y = x

        for fx in self.effects:
            y = fx.process_block(y)

        return y

    def is_real_time_safe(self):
        return all(getattr(fx, "real_time_safe", True) for fx in self.effects)