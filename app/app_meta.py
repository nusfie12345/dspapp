from dataclasses import dataclass
from typing import Callable, Any

from audio.effects.dspeffects import (
    Compressor,
    Overdrive,
    Distortion,
    Fuzz,
    BlockAmp,
    Flanger,
    Chorus,
    Phaser,
    Delay,
    Reverb,
    CabSim,
    EQ,
)


@dataclass
class ParamSpec:
    name: str
    label: str
    kind: str               # "slider", "choice", "bool"
    default: Any
    minimum: float = 0.0
    maximum: float = 10.0
    choices: tuple = ()


@dataclass
class EffectSpec:
    name: str
    category: str
    factory: Callable
    params: list[ParamSpec]

EFFECT_REGISTRY = {
    "Compressor": EffectSpec(
        name="Compressor",
        category="Comp",
        factory=lambda fs: Compressor(
            sens=5,
            level=5,
            atk=5e-4,
            rel=0.01,
            fs=fs,
        ),
        params=[
            ParamSpec("sens", "Sensitivity", "slider", 5),
            ParamSpec("level", "Level", "slider", 5),
        ]
    ),
    "Overdrive": EffectSpec(
        name="Overdrive",
        category="Drive",
        factory=lambda fs: Overdrive(
            drive=5,
            tone=5,
            level=5,
            mode="clear",
            clip="tanh",
            fs=fs,
        ),
        params=[
            ParamSpec("drive", "Drive", "slider", 5),
            ParamSpec("tone", "Tone", "slider", 5),
            ParamSpec("level", "Level", "slider", 5),
            ParamSpec(
                "clip",
                "Clip Type",
                "choice",
                "tanh",
                choices=("tanh", "atan", "soft", "hard", "null"),
            ),
            ParamSpec(
                "mode",
                "Waveshaping Mode",
                "choice",
                "clear",
                choices=("clear", "quad", "qcubic"),
            )
        ],
    ),
    "Distortion": EffectSpec(
        name="Distortion",
        category="Drive",
        factory=lambda fs: Distortion(
            dist=5,
            tone=5,
            level=5,
            mode="standard",
            filter="classic",
            fs=fs,
        ),
        params=[
            ParamSpec("dist", "Distortion", "slider", 5),
            ParamSpec("tone", "Tone", "slider", 5),
            ParamSpec("level", "Level", "slider", 5),
            ParamSpec(
                "mode", 
                "Waveshaping Mode", 
                "choice", 
                "standard", 
                choices=("standard", "cubic", "ord5"))
        ],
    ),
    "Fuzz": EffectSpec(
        name="Fuzz",
        category="Drive",
        factory=lambda fs: Fuzz(
            sustain=5,
            tone=5,
            level=5,
            asym=True,
            fs=fs
        ),
        params=[
            ParamSpec("sustain", "Sustain", "slider", 5),
            ParamSpec("tone", "Tone", "slider", 5),
            ParamSpec("level", "Level", "slider", 5),
            ParamSpec("asym", "Asymmetry", "choice", True, choices=(True, False))
        ],
    ),
    "BlockAmp": EffectSpec(
        name="BlockAmp",
        category="Amp",
        factory=lambda fs: BlockAmp(
            drive=4,
            tone=5,
            bass=5,
            mid=5,
            treb=5,
            pres=5,
            vol=5,
            clipType="tanh",
            fs=fs,
        ),
        params=[
            ParamSpec("gain", "Gain", "slider", 4),
            ParamSpec("tone", "Tone", "slider", 5),
            ParamSpec("bass", "Bass", "slider", 5),
            ParamSpec("mid", "Mid", "slider", 5),
            ParamSpec("treb", "Treble", "slider", 5),
            ParamSpec("pres", "Presence", "slider", 5),
            ParamSpec("vol", "Volume", "slider", 5),
            ParamSpec(
                "clipType",
                "Clip Type",
                "choice",
                "tanh",
                choices=("tanh", "atan", "hard", "null"),
            ),
        ],
    ),

    "CabSim": EffectSpec(
        name="CabSim",
        category="Cab",
        factory=lambda fs: CabSim(
            mode="filter",
            cab="closed_4x12",
            mix=10,
            level=7,
            fs=fs,
        ),
        params=[
            ParamSpec(
                "cab",
                "Cab",
                "choice",
                "closed_4x12",
                choices=("open_1x12", "closed_2x12", "closed_4x12", "bright_2x12", "dark_1x12"),
            ),
            ParamSpec("mix", "Mix", "slider", 10),
            ParamSpec("level", "Level", "slider", 7),
        ],
    ),

    "EQ": EffectSpec(
        name="EQ",
        category="EQ",
        factory=lambda fs: EQ(fs=fs, param=True, grph=False, level=10),
        params=[
            ParamSpec("level", "Level", "slider", 10),
            ParamSpec("param", "Parametric", "bool", True),
            ParamSpec("grph", "Graphic", "bool", False),
        ],
    ),

    "Flanger": EffectSpec(
        name="Flanger",
        category="Modulation",
        factory=lambda fs: Flanger(
            rate=2,
            depth=2,
            feedback=0,
            mix=3,
            delay=5,
            fs=fs,
        ),
        params=[
            ParamSpec("rate", "Rate", "slider", 2),
            ParamSpec("depth", "Depth", "slider", 2),
            ParamSpec("feedback", "Feedback", "slider", 0),
            ParamSpec("mix", "Mix", "slider", 3),
            ParamSpec("delay", "Delay", "slider", 5),
        ],
    ),

    "Chorus": EffectSpec(
        name="Chorus",
        category="Modulation",
        factory=lambda fs: Chorus(
            rate=2,
            depth=2,
            feedback=0,
            mix=3,
            delay=5,
            fs=fs,
        ),
        params=[
            ParamSpec("rate", "Rate", "slider", 2),
            ParamSpec("depth", "Depth", "slider", 2),
            ParamSpec("mix", "Mix", "slider", 3),
            ParamSpec("delay", "Delay", "slider", 5),
        ],
    ),

    "Phaser": EffectSpec(
        name="Phaser",
        category="Modulation",
        factory=lambda fs: Phaser(
            rate=2,
            depth=4,
            feedback=1,
            mix=3,
            stages=4,
            min_freq=300,
            max_freq=1600,
            fs=fs,
        ),
        params=[
            ParamSpec("rate", "Rate", "slider", 2),
            ParamSpec("depth", "Depth", "slider", 4),
            ParamSpec("feedback", "Feedback", "slider", 1),
            ParamSpec("mix", "Mix", "slider", 3),
            ParamSpec("stages", "Stages", "choice", 4, choices=(2, 4, 6, 8)),
        ],
    ),

    "Delay": EffectSpec(
        name="Delay",
        category="Time",
        factory=lambda fs: Delay(
            feedback=2,
            mix=2,
            delay_ms=300,
            mode="ms",
            fs=fs,
        ),
        params=[
            ParamSpec("feedback", "Feedback", "slider", 2),
            ParamSpec("mix", "Mix", "slider", 2),
            ParamSpec("delay_ms", "Delay ms", "slider", 300, minimum=20, maximum=1000),
            ParamSpec("mode", "Mode", "choice", "ms", choices=("ms", "sync")),
        ],
    ),

    "Reverb": EffectSpec(
        name="Reverb",
        category="Time",
        factory=lambda fs: Reverb(
            mode="lite",
            room_size=2,
            damping=7,
            mix=1.5,
            level=3,
            fs=fs,
        ),
        params=[
            ParamSpec("room_size", "Room Size", "slider", 2),
            ParamSpec("damping", "Damping", "slider", 7),
            ParamSpec("mix", "Mix", "slider", 1.5),
            ParamSpec("level", "Level", "slider", 3),
        ],
    ),
}