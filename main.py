# main.py

import sys
from PySide6.QtWidgets import QApplication

from app.live_rack import LiveRack
from app.audio_runner import AudioRunner
from app.gui import MainWindow


FS = 44100
BLOCKSIZE = 256


def main():
    qt_app = QApplication(sys.argv)

    rack = LiveRack()

    audio_runner = AudioRunner(
        rack=rack,
        fs=FS,
        blocksize=BLOCKSIZE,
        input_gain=0.5,
        output_gain=0.4,
        latency=(0.003, 0.0035),
    )

    window = MainWindow(
        rack=rack,
        audio_runner=audio_runner,
        fs=FS,
    )

    window.resize(1000, 650)
    window.show()

    exit_code = qt_app.exec()

    audio_runner.stop()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()