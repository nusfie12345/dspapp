from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QSlider,
    QCheckBox,
    QScrollArea,
    QGroupBox,
    QGridLayout,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QTimer

from app.app_meta import EFFECT_REGISTRY
from app.rack_builder import create_rack_item
from app.audio_runner import AudioRunner


class MainWindow(QMainWindow):
    def __init__(self, rack, audio_runner, fs=44100):
        super().__init__()

        self.rack = rack
        self.audio_runner = audio_runner
        self.fs = fs

        self.item_by_list_row = []

        self.setWindowTitle("Effect Rack")

        root = QWidget()
        self.setCentralWidget(root)

        # Main root layout is vertical:
        # top audio controls + bottom rack/effect editor.
        root_layout = QVBoxLayout(root)

        # -------------------------------------------------
        # Top audio bar
        # -------------------------------------------------
        top_bar = QHBoxLayout()
        root_layout.addLayout(top_bar)

        self.input_device_combo = QComboBox()
        self.output_device_combo = QComboBox()

        self.blocksize_combo = QComboBox()
        for bs in (64, 128, 256, 512, 1024):
            self.blocksize_combo.addItem(str(bs), userData=bs)
        self.blocksize_combo.setCurrentText("256")

        self.refresh_devices_button = QPushButton("Refresh Devices")
        self.start_audio_button = QPushButton("Start Audio")
        self.stop_audio_button = QPushButton("Stop Audio")

        self.refresh_devices_button.clicked.connect(self.refresh_audio_devices)
        self.start_audio_button.clicked.connect(self.start_audio)
        self.stop_audio_button.clicked.connect(self.stop_audio)

        top_bar.addWidget(QLabel("Input"))
        top_bar.addWidget(self.input_device_combo, stretch=2)

        top_bar.addWidget(QLabel("Output"))
        top_bar.addWidget(self.output_device_combo, stretch=2)

        top_bar.addWidget(QLabel("Block"))
        top_bar.addWidget(self.blocksize_combo)

        top_bar.addWidget(self.refresh_devices_button)
        top_bar.addWidget(self.start_audio_button)
        top_bar.addWidget(self.stop_audio_button)

        # -------------------------------------------------
        # Gain bar
        # -------------------------------------------------
        gain_bar = QHBoxLayout()
        root_layout.addLayout(gain_bar)

        self.input_gain_label = QLabel("Input Gain: 0.50")
        self.input_gain_slider = QSlider(Qt.Horizontal)
        self.input_gain_slider.setMinimum(0)
        self.input_gain_slider.setMaximum(200)
        self.input_gain_slider.setValue(50)
        self.input_gain_slider.valueChanged.connect(self.on_input_gain_changed)

        self.output_gain_label = QLabel("Output Gain: 0.40")
        self.output_gain_slider = QSlider(Qt.Horizontal)
        self.output_gain_slider.setMinimum(0)
        self.output_gain_slider.setMaximum(200)
        self.output_gain_slider.setValue(40)
        self.output_gain_slider.valueChanged.connect(self.on_output_gain_changed)

        gain_bar.addWidget(self.input_gain_label)
        gain_bar.addWidget(self.input_gain_slider)

        gain_bar.addWidget(self.output_gain_label)
        gain_bar.addWidget(self.output_gain_slider)

        # -------------------------------------------------
        # Status bar
        # -------------------------------------------------
        self.audio_status_label = QLabel("Audio stopped")
        root_layout.addWidget(self.audio_status_label)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_audio_status)
        self.status_timer.start(500)

        # -------------------------------------------------
        # Main rack/editor area
        # -------------------------------------------------
        main_layout = QHBoxLayout()
        root_layout.addLayout(main_layout, stretch=1)

        # Left side: rack controls.
        left = QVBoxLayout()
        main_layout.addLayout(left, stretch=1)

        self.effect_combo = QComboBox()
        self.effect_combo.addItems(sorted(EFFECT_REGISTRY.keys()))

        self.add_button = QPushButton("Add Effect")
        self.add_button.clicked.connect(self.add_effect)

        self.chain_list = QListWidget()
        self.chain_list.currentRowChanged.connect(self.on_selected_effect_changed)
        self.chain_list.itemChanged.connect(self.on_chain_item_changed)

        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_selected)

        self.up_button = QPushButton("Move Up")
        self.up_button.clicked.connect(lambda: self.move_selected(-1))

        self.down_button = QPushButton("Move Down")
        self.down_button.clicked.connect(lambda: self.move_selected(1))

        left.addWidget(QLabel("Add effect"))
        left.addWidget(self.effect_combo)
        left.addWidget(self.add_button)

        left.addWidget(QLabel("Chain"))
        left.addWidget(self.chain_list, stretch=1)

        left.addWidget(self.up_button)
        left.addWidget(self.down_button)
        left.addWidget(self.remove_button)

        # Right side: parameter editor.
        self.param_container = QWidget()
        self.param_panel = QVBoxLayout(self.param_container)
        main_layout.addWidget(self.param_container, stretch=2)

        self.refresh_audio_devices()

    # -------------------------------------------------
    # Audio controls
    # -------------------------------------------------

    def refresh_audio_devices(self):
        self.input_device_combo.clear()
        self.output_device_combo.clear()

        input_devices = AudioRunner.list_input_devices()
        output_devices = AudioRunner.list_output_devices()

        for dev in input_devices:
            label = f"{dev.index}: {dev.name} [{dev.hostapi}]"
            self.input_device_combo.addItem(label, userData=dev.index)

        for dev in output_devices:
            label = f"{dev.index}: {dev.name} [{dev.hostapi}]"
            self.output_device_combo.addItem(label, userData=dev.index)

        # Try to auto-select Spark if present.
        for i in range(self.input_device_combo.count()):
            text = self.input_device_combo.itemText(i).lower()
            if "positive grid" in text or "spark" in text:
                self.input_device_combo.setCurrentIndex(i)
                break

        for i in range(self.output_device_combo.count()):
            text = self.output_device_combo.itemText(i).lower()
            if "positive grid" in text or "spark" in text:
                self.output_device_combo.setCurrentIndex(i)
                break

    def start_audio(self):
        input_dev = self.input_device_combo.currentData()
        output_dev = self.output_device_combo.currentData()
        blocksize = self.blocksize_combo.currentData()

        self.audio_runner.set_devices(
            input_device=input_dev,
            output_device=output_dev,
        )

        self.audio_runner.set_audio_config(
            fs=self.fs,
            blocksize=blocksize,
            latency=(0.003, 0.0035),
        )

        self.audio_runner.start()

    def stop_audio(self):
        self.audio_runner.stop()

    def on_input_gain_changed(self, value):
        gain = value / 100.0
        self.audio_runner.set_input_gain(gain)
        self.input_gain_label.setText(f"Input Gain: {gain:.2f}")

    def on_output_gain_changed(self, value):
        gain = value / 100.0
        self.audio_runner.set_output_gain(gain)
        self.output_gain_label.setText(f"Output Gain: {gain:.2f}")

    def update_audio_status(self):
        status = self.audio_runner.get_status()

        self.audio_status_label.setText(
            f"Running: {status['running']} | "
            f"XRuns: {status['status_count']} | "
            f"Errors: {status['error_count']} | "
            f"In: {status['input_peak']:.3f} | "
            f"Chain: {status['chain_peak']:.3f} | "
            f"Out: {status['output_peak']:.3f}"
        )

    # -------------------------------------------------
    # Rack controls
    # -------------------------------------------------

    def add_effect(self):
        effect_name = self.effect_combo.currentText()
        item = create_rack_item(effect_name, self.fs)

        self.rack.enqueue_add(item)

        self.item_by_list_row.append(item)

        list_item = QListWidgetItem(effect_name)
        list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
        list_item.setCheckState(Qt.Checked)

        self.chain_list.addItem(list_item)
        self.chain_list.setCurrentRow(self.chain_list.count() - 1)

    def remove_selected(self):
        row = self.chain_list.currentRow()

        if row < 0:
            return

        item = self.item_by_list_row[row]

        self.rack.enqueue_remove(item.id)

        self.item_by_list_row.pop(row)
        self.chain_list.takeItem(row)

        self.clear_param_panel()

    def move_selected(self, direction):
        row = self.chain_list.currentRow()

        if row < 0:
            return

        new_row = row + direction

        if not 0 <= new_row < self.chain_list.count():
            return

        item = self.item_by_list_row[row]
        self.rack.enqueue_move(item.id, direction)

        self.item_by_list_row[row], self.item_by_list_row[new_row] = (
            self.item_by_list_row[new_row],
            self.item_by_list_row[row],
        )

        taken = self.chain_list.takeItem(row)
        self.chain_list.insertItem(new_row, taken)
        self.chain_list.setCurrentRow(new_row)

    def on_chain_item_changed(self, list_item):
        row = self.chain_list.row(list_item)

        if row < 0 or row >= len(self.item_by_list_row):
            return

        item = self.item_by_list_row[row]
        enabled = list_item.checkState() == Qt.Checked

        self.rack.enqueue_toggle(item.id, enabled)

        # Keep GUI-side copy in sync for the parameter panel checkbox.
        item.effect.enabled = enabled

    def on_selected_effect_changed(self, row):
        self.clear_param_panel()

        if row < 0:
            return

        if row >= len(self.item_by_list_row):
            return

        item = self.item_by_list_row[row]
        self.build_param_panel(item)

    # -------------------------------------------------
    # Parameter panel
    # -------------------------------------------------

    def clear_param_panel(self):
        while self.param_panel.count():
            child = self.param_panel.takeAt(0)

            if child.widget():
                child.widget().deleteLater()

    def build_param_panel(self, item):
        if item.name == "EQ":
            self.build_eq_panel(item)
            return

        self.param_panel.addWidget(QLabel(f"<b>{item.name}</b>"))

        enabled_box = QCheckBox("Enabled")
        enabled_box.setChecked(item.effect.enabled)

        enabled_box.stateChanged.connect(
            lambda state, item_id=item.id: self.rack.enqueue_toggle(
                item_id,
                state == Qt.Checked,
            )
        )

        self.param_panel.addWidget(enabled_box)

        for spec in item.spec.params:
            if spec.kind == "slider":
                self.add_slider_control(item, spec)
            elif spec.kind == "choice":
                self.add_choice_control(item, spec)
            elif spec.kind == "bool":
                self.add_bool_control(item, spec)

        self.param_panel.addStretch()
    
    def build_eq_panel(self, item):
        self.param_panel.addWidget(QLabel("<b>EQ</b>"))

        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        enabled_box = QCheckBox("Enabled")
        enabled_box.setChecked(item.effect.enabled)
        enabled_box.stateChanged.connect(
            lambda state, item_id=item.id: self.rack.enqueue_toggle(
                item_id,
                state == Qt.Checked,
            )
        )

        param_box = QCheckBox("Parametric")
        param_box.setChecked(item.effect.params.get("param", True))
        param_box.stateChanged.connect(
            lambda state, item_id=item.id: self.rack.enqueue_param(
                item_id,
                "param",
                state == Qt.Checked,
            )
        )

        grph_box = QCheckBox("Graphic")
        grph_box.setChecked(item.effect.params.get("grph", False))
        grph_box.stateChanged.connect(
            lambda state, item_id=item.id: self.rack.enqueue_param(
                item_id,
                "grph",
                state == Qt.Checked,
            )
        )

        top_layout.addWidget(enabled_box)
        top_layout.addWidget(param_box)
        top_layout.addWidget(grph_box)
        top_layout.addStretch()

        self.param_panel.addWidget(top_row)

        self.param_panel.addWidget(QLabel("<b>Parametric bands</b>"))

        for i, band in enumerate(item.effect.param_bands):
            self.add_eq_band_controls(item, i, band)

        self.param_panel.addWidget(QLabel("<b>Graphic EQ</b>"))

        for i, freq in enumerate(item.effect.grph_freqs):
            gain = item.effect.grph_gains[i]
            self.add_graphic_eq_slider(item, i, freq, gain)

        self.param_panel.addStretch()


    def add_eq_band_controls(self, item, index, band):
        group = QGroupBox(f"Band {index + 1}")
        grid = QGridLayout(group)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        enabled_box = QCheckBox("On")
        enabled_box.setChecked(band["enabled"])
        enabled_box.stateChanged.connect(
            lambda state, item_id=item.id, idx=index: self.rack.enqueue_eq_band(
                item_id,
                idx,
                enabled=(state == Qt.Checked),
            )
        )

        type_combo = QComboBox()
        for band_type in ("hpf", "lpf", "peak", "loshelf", "hishelf"):
            type_combo.addItem(band_type)

        type_combo.setCurrentText(band["type"])
        type_combo.currentTextChanged.connect(
            lambda text, item_id=item.id, idx=index: self.rack.enqueue_eq_band(
                item_id,
                idx,
                type=text,
            )
        )

        grid.addWidget(enabled_box, 0, 0)
        grid.addWidget(QLabel("Type"), 0, 1)
        grid.addWidget(type_combo, 0, 2)

        self.add_eq_spinbox(
            grid,
            row=1,
            label_text="Freq",
            value=band["fc"],
            minimum=20.0,
            maximum=16000.0,
            step=10.0,
            suffix=" Hz",
            on_change=lambda value, item_id=item.id, idx=index: self.rack.enqueue_eq_band(
                item_id,
                idx,
                fc=value,
            ),
        )

        self.add_eq_spinbox(
            grid,
            row=2,
            label_text="Gain",
            value=band["gain"],
            minimum=-24.0,
            maximum=24.0,
            step=0.5,
            suffix=" dB",
            on_change=lambda value, item_id=item.id, idx=index: self.rack.enqueue_eq_band(
                item_id,
                idx,
                gain=value,
            ),
        )

        self.add_eq_spinbox(
            grid,
            row=3,
            label_text="Q",
            value=band["Q"],
            minimum=0.1,
            maximum=10.0,
            step=0.1,
            suffix="",
            on_change=lambda value, item_id=item.id, idx=index: self.rack.enqueue_eq_band(
                item_id,
                idx,
                Q=value,
            ),
        )

        self.param_panel.addWidget(group)
    
    def add_eq_spinbox(
        self,
        grid,
        row,
        label_text,
        value,
        minimum,
        maximum,
        step,
        suffix,
        on_change,
    ):
        label = QLabel(label_text)

        spin = QDoubleSpinBox()
        spin.setMinimum(minimum)
        spin.setMaximum(maximum)
        spin.setSingleStep(step)
        spin.setValue(float(value))
        spin.setSuffix(suffix)
        spin.setDecimals(2)

        spin.valueChanged.connect(on_change)

        grid.addWidget(label, row, 0)
        grid.addWidget(spin, row, 1, 1, 2)


    def add_eq_band_slider(self, item, index, param_name, label_text, value, minimum, maximum):
        label = QLabel(f"{label_text}: {value:.2f}")

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(1000)

        def value_to_slider(v):
            return int(1000 * (float(v) - minimum) / (maximum - minimum))

        def slider_to_value(s):
            return minimum + (maximum - minimum) * (s / 1000.0)

        slider.setValue(value_to_slider(value))

        def on_change(s, item_id=item.id, idx=index, name=param_name):
            new_value = slider_to_value(s)
            label.setText(f"{label_text}: {new_value:.2f}")

            self.rack.enqueue_eq_band(
                item_id,
                idx,
                **{name: new_value},
            )

        slider.valueChanged.connect(on_change)

        self.param_panel.addWidget(label)
        self.param_panel.addWidget(slider)


    def add_graphic_eq_slider(self, item, index, freq, gain):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel(f"{freq} Hz")
        label.setMinimumWidth(70)

        value_label = QLabel(f"{gain:.1f} dB")
        value_label.setMinimumWidth(60)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(-240)
        slider.setMaximum(240)
        slider.setValue(int(gain * 10.0))

        def on_change(v, item_id=item.id, idx=index):
            new_gain = v / 10.0
            value_label.setText(f"{new_gain:.1f} dB")
            self.rack.enqueue_graphic_eq_band(item_id, idx, new_gain)

        slider.valueChanged.connect(on_change)

        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_label)

        self.param_panel.addWidget(row)

    def add_slider_control(self, item, spec):
        initial = self._get_current_param_value(item, spec)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        label = QLabel(f"{spec.label}: {initial:.2f}")
        label.setMinimumWidth(110)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(1000)

        def value_to_slider(v):
            return int(
                1000
                * (float(v) - spec.minimum)
                / (spec.maximum - spec.minimum)
            )

        def slider_to_value(s):
            return spec.minimum + (spec.maximum - spec.minimum) * (s / 1000.0)

        slider.setValue(value_to_slider(initial))

        def on_change(s, item_id=item.id, name=spec.name):
            value = slider_to_value(s)
            label.setText(f"{spec.label}: {value:.2f}")
            self.rack.enqueue_param(item_id, name, value)

        slider.valueChanged.connect(on_change)

        row_layout.addWidget(label)
        row_layout.addWidget(slider)

        self.param_panel.addWidget(row)

    def add_choice_control(self, item, spec):
        label = QLabel(spec.label)
        combo = QComboBox()

        for choice in spec.choices:
            combo.addItem(str(choice), userData=choice)

        current = self._get_current_param_value(item, spec)

        if current in spec.choices:
            combo.setCurrentIndex(list(spec.choices).index(current))
        else:
            combo.setCurrentIndex(0)

        def on_change(index, item_id=item.id, name=spec.name):
            value = combo.itemData(index)
            self.rack.enqueue_param(item_id, name, value)

        combo.currentIndexChanged.connect(on_change)

        self.param_panel.addWidget(label)
        self.param_panel.addWidget(combo)

    def add_bool_control(self, item, spec):
        current = self._get_current_param_value(item, spec)

        checkbox = QCheckBox(spec.label)
        checkbox.setChecked(bool(current))

        checkbox.stateChanged.connect(
            lambda state, item_id=item.id, name=spec.name: self.rack.enqueue_param(
                item_id,
                name,
                state == Qt.Checked,
            )
        )

        self.param_panel.addWidget(checkbox)

    def _get_current_param_value(self, item, spec):
        """
        Prefer the actual effect param if available.
        Fall back to the registry default.
        """
        if hasattr(item.effect, "params") and spec.name in item.effect.params:
            value = item.effect.params[spec.name]

            # Most DSP params are stored normalized 0..1 internally,
            # but GUI sliders are 0..10. Convert only for standard sliders.
            if spec.kind == "slider" and spec.minimum == 0.0 and spec.maximum == 10.0:
                if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
                    return float(value) * 10.0

            return value

        return spec.default