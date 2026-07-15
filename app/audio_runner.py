import os

os.environ.setdefault("SD_ENABLE_ASIO", "1")

import gc
import traceback
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi: str


class AudioRunner:
    def __init__(
        self,
        rack,
        fs=44100,
        blocksize=256,
        input_device=None,
        output_device=None,
        input_gain=1.0,
        output_gain=1.0,
        latency=None,
    ):
        self.rack = rack

        self.fs = int(fs)
        self.blocksize = int(blocksize)

        self.input_device = input_device
        self.output_device = output_device

        self.input_gain = float(input_gain)
        self.output_gain = float(output_gain)

        # For ASIO, explicit low latency can help.
        # Example: latency=(0.003, 0.0035)
        self.latency = latency

        self.stream = None
        self.running = False

        self.status_count = 0
        self.error_count = 0
        self.last_error = None

        self.last_input_peak = 0.0
        self.last_output_peak = 0.0
        self.last_chain_peak = 0.0

    # -------------------------
    # Device discovery
    # -------------------------

    @staticmethod
    def list_devices():
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        result = []

        for idx, dev in enumerate(devices):
            hostapi_name = hostapis[dev["hostapi"]]["name"]

            result.append(
                AudioDevice(
                    index=idx,
                    name=dev["name"],
                    max_input_channels=int(dev["max_input_channels"]),
                    max_output_channels=int(dev["max_output_channels"]),
                    default_samplerate=float(dev["default_samplerate"]),
                    hostapi=hostapi_name,
                )
            )

        return result

    @staticmethod
    def list_input_devices():
        return [
            dev for dev in AudioRunner.list_devices()
            if dev.max_input_channels > 0
        ]

    @staticmethod
    def list_output_devices():
        return [
            dev for dev in AudioRunner.list_devices()
            if dev.max_output_channels > 0
        ]

    @staticmethod
    def find_device_by_name(name, *, input_required=False, output_required=False):
        name = name.lower()

        for dev in AudioRunner.list_devices():
            if name in dev.name.lower():
                if input_required and dev.max_input_channels <= 0:
                    continue
                if output_required and dev.max_output_channels <= 0:
                    continue
                return dev.index

        return None

    def set_devices(self, input_device=None, output_device=None):
        was_running = self.running

        if was_running:
            self.stop()

        self.input_device = input_device
        self.output_device = output_device

        if was_running:
            self.start()

    def set_audio_config(self, fs=None, blocksize=None, latency=None):
        was_running = self.running

        if was_running:
            self.stop()

        if fs is not None:
            self.fs = int(fs)

        if blocksize is not None:
            self.blocksize = int(blocksize)

        if latency is not None:
            self.latency = latency

        if was_running:
            self.start()

    def set_input_gain(self, gain):
        self.input_gain = float(gain)

    def set_output_gain(self, gain):
        self.output_gain = float(gain)

    # -------------------------
    # Stream lifecycle
    # -------------------------

    def start(self):
        if self.running:
            return
        
        # heating up the kernels
        dummy = np.zeros(self.blocksize, dtype=np.float32)
        self.rack.process_block(dummy)
        self.rack.process_block(dummy)

        self.status_count = 0
        self.error_count = 0
        self.last_error = None

        self.rack.reset()

        # Prevent random GC pauses during live audio.
        gc.disable()

        stream_kwargs = dict(
            samplerate=self.fs,
            blocksize=self.blocksize,
            dtype="float32",
            channels=(1, 2),  # mono input, stereo output
            callback=self._callback,
            device=(self.input_device, self.output_device),
        )

        if self.latency is not None:
            stream_kwargs["latency"] = self.latency

        self.stream = sd.Stream(**stream_kwargs)
        self.stream.start()

        self.running = True

    def stop(self):
        if not self.running:
            return

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
            self.running = False
            gc.enable()

    def restart(self):
        self.stop()
        self.start()

    # -------------------------
    # Callback
    # -------------------------

    def _callback(self, indata, outdata, frames, time_info, status):
        try:
            if status:
                self.status_count += 1

            # Mono guitar input.
            x = indata[:, 0].astype(np.float32, copy=True)

            x *= self.input_gain

            self.last_input_peak = float(np.max(np.abs(x))) if len(x) else 0.0

            y = self.rack.process_block(x)

            if y is None:
                raise RuntimeError("rack.process_block() returned None")

            y = np.asarray(y, dtype=np.float32)

            if not np.all(np.isfinite(y)):
                raise RuntimeError("rack produced NaN/Inf")

            self.last_chain_peak = float(np.max(np.abs(y))) if y.size else 0.0

            y *= self.output_gain

            # Emergency hard safety. Keep gain staging low enough that this rarely hits.
            y = np.clip(y, -0.98, 0.98).astype(np.float32)

            self.last_output_peak = float(np.max(np.abs(y))) if y.size else 0.0

            if y.ndim == 1:
                outdata[:, 0] = y

                if outdata.shape[1] > 1:
                    outdata[:, 1] = y

            elif y.ndim == 2 and y.shape[1] == 2:
                outdata[:, 0] = y[:, 0]

                if outdata.shape[1] > 1:
                    outdata[:, 1] = y[:, 1]

            else:
                raise ValueError(f"Unsupported rack output shape: {y.shape}")

        except Exception:
            self.error_count += 1
            self.last_error = traceback.format_exc()
            outdata.fill(0.0)

    # -------------------------
    # Status
    # -------------------------

    def get_status(self):
        return {
            "running": self.running,
            "fs": self.fs,
            "blocksize": self.blocksize,
            "input_device": self.input_device,
            "output_device": self.output_device,
            "status_count": self.status_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "input_peak": self.last_input_peak,
            "chain_peak": self.last_chain_peak,
            "output_peak": self.last_output_peak,
        }