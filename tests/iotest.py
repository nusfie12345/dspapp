# A safe debug for y'all to check if you didn't forget your audio interface or whatever.

import os
os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd

hostapis = sd.query_hostapis()
devices = sd.query_devices()

for i, dev in enumerate(devices):
    api_name = hostapis[dev["hostapi"]]["name"]

    print(
        f"{i}: {dev['name']} | API: {api_name} | "
        f"in={dev['max_input_channels']} out={dev['max_output_channels']} | "
        f"low_in={dev['default_low_input_latency']:.4f} "
        f"low_out={dev['default_low_output_latency']:.4f} | "
        f"high_in={dev['default_high_input_latency']:.4f} "
        f"high_out={dev['default_high_output_latency']:.4f}"
    )