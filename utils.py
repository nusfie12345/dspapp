# imports
# lmao no need, it's core python
#______________________

def normalize(param):
    """
    A function to properly normalize all the numeric inputs to the scale [0, 1].

    Args:
        param: designed parameter for the effect to normalize.
    """
    if not isinstance(param, (int, float)):
        raise ValueError(f"Parameter {param} must be numeric.")

    param = max(0, min(10, param))

    return param / 10

def volume_gain(param):
    """
    A scale function reserved specifically for the "volume" parameter. 
    
    Remaps the makeup(pre-master) volume from [0, 1] to [-20, +6] dB.

    Args:
        param: volume value
    """
    if not 0 <= param <= 1:
        raise ValueError("Volume parameter must be between 0 and 1.")
    
    db = -20 + 26 * param
    return 10**(db/20)

def drive_eff(param, min_val, max_val):
    """
    A drive effect rescale designed for OD, distortion, and fuzz.

    Args:
        param: initial drive value
        min_val: minimum value for the effect
        max_val: maximum value for the effect
    """
    if not isinstance(param, (int, float)):
        raise ValueError("Drive parameter must be numeric.")

    if not 0 <= param <= 1:
        raise ValueError("Drive parameter must be between 0 and 1.")

    if min_val <= 0:
        raise ValueError("min_val must be > 0.")

    if max_val <= min_val:
        raise ValueError("max_val must exceed min_val.")

    return min_val * (max_val/min_val)**param

NOTE_DIVISIONS = {
    "1/1": 4.0,
    "1/2": 2.0,
    "1/4": 1.0,
    "1/8": 0.5,
    "1/16": 0.25,
    "1/32": 0.125,

    "1/2d": 3.0,
    "1/4d": 1.5,
    "1/8d": 0.75,
    "1/16d": 0.375,

    "1/2t": 4.0 / 3.0,
    "1/4t": 2.0 / 3.0,
    "1/8t": 1.0 / 3.0,
    "1/16t": 1.0 / 6.0,
}


def bpm_to_delay_seconds(bpm, division):
    if bpm <= 0:
        raise ValueError("bpm must be positive")

    if division not in NOTE_DIVISIONS:
        raise ValueError(f"Invalid note division: {division}")

    return (60.0 / bpm) * NOTE_DIVISIONS[division]


def feedback_from_echoes(echoes, threshold=0.01, max_feedback=0.95):
    if echoes <= 0:
        return 0.0

    fb = threshold ** (1.0 / echoes)
    return min(fb, max_feedback)

CABSIM_PRESETS = {
    "open_1x12": {
        "hpf": 90,
        "lpf": 6500,
        "peaks": [
            (130, 1.5, 0.8),
            (350, -2.0, 1.0),
            (1200, 2.0, 1.0),
            (3200, -2.5, 1.2),
        ],
    },

    "closed_2x12": {
        "hpf": 80,
        "lpf": 5800,
        "peaks": [
            (110, 2.0, 0.8),
            (300, -2.5, 1.0),
            (900, 2.5, 1.0),
            (2500, -1.5, 1.3),
            (4200, -3.0, 1.0),
        ],
    },

    "closed_4x12": {
        "hpf": 75,
        "lpf": 5200,
        "peaks": [
            (100, 2.5, 0.8),
            (240, -2.0, 1.1),
            (750, 2.0, 1.0),
            (1600, 2.5, 1.0),
            (3600, -3.5, 1.2),
        ],
    },

    "bright_2x12": {
        "hpf": 75,
        "lpf": 7200,
        "peaks": [
            (120, 1.5, 0.9),
            (500, -1.5, 1.0),
            (1800, 2.5, 1.0),
            (4300, 1.5, 1.0),
        ],
    },

    "dark_1x12": {
        "hpf": 110,
        "lpf": 4300,
        "peaks": [
            (150, 2.0, 0.8),
            (450, -2.0, 1.0),
            (1000, 1.5, 1.1),
            (2800, -3.0, 1.2),
        ],
    },
}