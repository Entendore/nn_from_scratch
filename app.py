"""
Neural Networks from Scratch — PySide6 Learning App
=====================================================
A comprehensive interactive desktop application for learning
neural networks from first principles using only NumPy.
"""

import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QSlider, QGroupBox, QFormLayout, QScrollArea, QSplitter,
    QFrame, QSizePolicy, QTextEdit, QGridLayout, QCheckBox,
    QStatusBar, QProgressBar, QTabWidget, QLineEdit
)
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QSize, QRect, QPointF, QRectF
)
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
    QLinearGradient, QRadialGradient, QIcon
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("QtAgg")

from nn_engine import (
    NeuralNetwork, ACTIVATION_MAP, LOSS_MAP, DATASETS,
    Activation, Loss, SEQUENCE_DATASETS, CONV_DATASETS,
    DenseLayer, DropoutLayer, BatchNormLayer, Conv1DLayer,
    SimpleRNNLayer, LSTMLayer, SelfAttentionLayer,
    make_sine_sequence, make_counter_sequence,
    compute_self_attention, compute_multi_head_attention, ATTENTION_PRESETS,
    conv2d, generate_test_image, KERNEL_PRESETS,
    run_scaling_experiment,
    AdaBoostClassifier, DecisionStump,
    analyze_gradient_flow, analyze_gradient_flow_with_weights,
    Diffusion1D, Diffusion2D, make_diffusion_data,
    TransformerEncoder, Autoencoder, GAN, ResidualDemo,
    compare_normalizations, normalization_gradient_flow,
    compute_confusion_matrix, compute_metrics_from_cm, compute_roc_curve,
    train_and_evaluate, compute_pca, compute_tsne_simple,
)


# ════════════════════════════════════════════════════════════════
#  STYLESHEET
# ════════════════════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f1117;
    color: #e1e4e8;
    font-family: 'Segoe UI', 'Noto Sans', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 18px;
    font-weight: bold;
    font-size: 13px;
    color: #58a6ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 16px;
    color: #e1e4e8;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #58a6ff;
}
QPushButton:pressed {
    background-color: #0f1117;
}
QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #21262d;
}
QPushButton#accent {
    background-color: #238636;
    border-color: #2ea043;
    color: #ffffff;
}
QPushButton#accent:hover {
    background-color: #2ea043;
}
QPushButton#danger {
    background-color: #da3633;
    border-color: #f85149;
    color: #ffffff;
}
QPushButton#danger:hover {
    background-color: #f85149;
}
QPushButton#warning {
    background-color: #9e6a03;
    border-color: #d29922;
    color: #ffffff;
}
QPushButton#warning:hover {
    background-color: #d29922;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 5px 10px;
    color: #e1e4e8;
    min-height: 24px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #21262d;
    border: 1px solid #30363d;
    color: #e1e4e8;
    selection-background-color: #30363d;
}
QSlider::groove:horizontal {
    border: 1px solid #30363d;
    height: 6px;
    background: #21262d;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #58a6ff;
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #79c0ff;
}
QTextEdit {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #e1e4e8;
    padding: 8px;
    font-size: 13px;
}
QScrollArea {
    border: none;
}
QTabWidget::pane {
    border: 1px solid #30363d;
    border-radius: 6px;
    background: #161b22;
}
QTabBar::tab {
    background: #21262d;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #8b949e;
}
QTabBar::tab:selected {
    background: #161b22;
    color: #58a6ff;
    font-weight: bold;
}
QTabBar::tab:hover {
    color: #e1e4e8;
}
QLabel#sectionTitle {
    font-size: 18px;
    font-weight: bold;
    color: #58a6ff;
    padding: 8px 0;
}
QLabel#hint {
    color: #8b949e;
    font-size: 12px;
}
QProgressBar {
    border: 1px solid #30363d;
    border-radius: 4px;
    background: #21262d;
    text-align: center;
    color: #e1e4e8;
    height: 20px;
}
QProgressBar::chunk {
    background: #238636;
    border-radius: 3px;
}
"""


# ════════════════════════════════════════════════════════════════
#  NETWORK DIAGRAM WIDGET (QPainter-based)
# ════════════════════════════════════════════════════════════════

class NetworkDiagramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layer_sizes = []
        self.activations = None
        self.gradients = None
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_network(self, layer_sizes, activations=None, gradients=None):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.gradients = gradients
        self.update()

    def paintEvent(self, event):
        if not self.layer_sizes:
            painter = QPainter(self)
            painter.setPen(QColor("#8b949e"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Build a network to see the diagram")
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        n_layers = len(self.layer_sizes)
        margin_x, margin_y = 80, 40
        layer_spacing = (w - 2 * margin_x) / max(n_layers - 1, 1)
        max_neurons = max(s[0] for s in self.layer_sizes)

        positions = []
        for i, (n_neurons, _) in enumerate(self.layer_sizes):
            if n_neurons == 0:
                positions.append([])
                continue
            x = margin_x + i * layer_spacing
            neuron_spacing = min(40, (h - 2 * margin_y) / max(n_neurons, 1))
            total_h = (n_neurons - 1) * neuron_spacing
            y_start = (h - total_h) / 2
            layer_pos = [QPointF(x, y_start + j * neuron_spacing) for j in range(n_neurons)]
            positions.append(layer_pos)

        # Draw connections
        for i in range(len(positions) - 1):
            if not positions[i] or not positions[i + 1]:
                continue
            for j, p1 in enumerate(positions[i]):
                for k, p2 in enumerate(positions[i + 1]):
                    pen = QPen(QColor(60, 70, 90), 1.0)
                    if self.activations and i + 1 < len(self.activations):
                        val = abs(self.activations[i + 1][k]) if k < len(self.activations[i + 1]) else 0
                        intensity = min(255, int(val * 255))
                        pen = QPen(QColor(intensity, 80, 255 - intensity, 120), 1.0)
                    if self.gradients and i < len(self.gradients):
                        gw = self.gradients[i].get("dW", None)
                        if gw is not None and j < gw.shape[0] and k < gw.shape[1]:
                            gval = min(1.0, abs(gw[j, k]) * 500)
                            pen = QPen(QColor(255, int(120 * (1 - gval)), 50, 140), 1.5)
                    painter.setPen(pen)
                    painter.drawLine(p1, p2)

        # Draw neurons
        radius = 16
        for i, layer_pos in enumerate(positions):
            if not layer_pos:
                # Special layer type — draw a box instead
                x = margin_x + i * layer_spacing
                _, name = self.layer_sizes[i]
                painter.setPen(QPen(QColor("#f0883e"), 1.5))
                painter.setBrush(QBrush(QColor(33, 38, 45, 200)))
                painter.drawRoundedRect(QRectF(x - 35, h / 2 - 20, 70, 40), 8, 8)
                painter.setPen(QColor("#f0883e"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.drawText(QRectF(x - 35, h / 2 - 20, 70, 40), Qt.AlignCenter, name)
                continue
            _, act_name = self.layer_sizes[i]
            for j, pos in enumerate(layer_pos):
                base_color = QColor("#1f6feb")
                if i == 0:
                    base_color = QColor("#238636")
                elif i == len(positions) - 1:
                    base_color = QColor("#da3633")
                if self.activations and i < len(self.activations):
                    val = self.activations[i][j] if j < len(self.activations[i]) else 0
                    val = np.clip(val, -1, 1)
                    if val >= 0:
                        base_color = QColor(int(40 + val * 160), int(100 + val * 155), int(60 + val * 100))
                    else:
                        base_color = QColor(int(60 + abs(val) * 195), int(40 + abs(val) * 60), int(40 + abs(val) * 60))
                glow = QRadialGradient(pos, radius * 1.5)
                gc = QColor(base_color); gc.setAlpha(40)
                glow.setColorAt(0, gc); glow.setColorAt(1, QColor(0, 0, 0, 0))
                painter.setBrush(QBrush(glow)); painter.setPen(Qt.NoPen)
                painter.drawEllipse(pos, radius * 1.5, radius * 1.5)
                painter.setBrush(QBrush(base_color))
                painter.setPen(QPen(QColor("#e1e4e8"), 1.5))
                painter.drawEllipse(pos, radius, radius)

        # Draw layer labels
        painter.setPen(QColor("#8b949e"))
        painter.setFont(QFont("Segoe UI", 10))
        for i, (_, act_name) in enumerate(self.layer_sizes):
            x = margin_x + i * layer_spacing
            label = f"{'Input' if i == 0 else 'Output' if i == n_layers - 1 else f'Hidden {i}'}"
            if self.layer_sizes[i][0] > 0:
                label2 = f"({self.layer_sizes[i][0]})"
                painter.drawText(QPointF(x - 30, h - 10), label)
                painter.drawText(QPointF(x - 15, h - 26), label2)
            if i > 0:
                painter.setPen(QColor("#58a6ff"))
                painter.drawText(QPointF(x - 25, h - 42), act_name)
                painter.setPen(QColor("#8b949e"))
        painter.end()


# ════════════════════════════════════════════════════════════════
#  MATPLOTLIB CANVAS
# ════════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#0f1117")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._default_title = "plot"

    def save_plot(self, filepath=None):
        """Save current figure to PNG. Returns the path used."""
        import os
        if filepath is None:
            d = os.path.expanduser("~/Downloads")
            os.makedirs(d, exist_ok=True)
            safe = re.sub(r'[^\w\-]+', '_', self._default_title.lower())
            filepath = os.path.join(d, f"nn_{safe}.png")
        self.fig.savefig(filepath, dpi=150, facecolor=self.fig.get_facecolor(),
                         edgecolor='none', bbox_inches='tight')
        return filepath

    def style_axes(self, ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.xaxis.label.set_color("#e1e4e8")
        ax.yaxis.label.set_color("#e1e4e8")
        ax.title.set_color("#e1e4e8")
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
            self._default_title = title
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)


# ════════════════════════════════════════════════════════════════
#  TRAINING THREAD
# ════════════════════════════════════════════════════════════════

class TrainingThread(QThread):
    epoch_done = Signal(int, float, float, object, object)
    finished_signal = Signal()

    def __init__(self, nn, X, y, X_val=None, y_val=None, epochs=100, batch_size=32,
                 lr_scheduler=None, scheduler_kwargs=None):
        super().__init__()
        self.nn = nn
        self.X, self.y = X, y
        self.X_val, self.y_val = X_val, y_val
        self.epochs, self.batch_size = epochs, batch_size
        self.lr_scheduler = lr_scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self._stop = False
        self.nn.accuracy_history = []

    def stop(self):
        self._stop = True

    def run(self):
        n = self.X.shape[0]
        for epoch in range(self.epochs):
            if self._stop:
                break
            # LR scheduling
            if self.lr_scheduler == "step":
                self.nn.step_lr(epoch, **self.scheduler_kwargs)
            elif self.lr_scheduler == "exp":
                self.nn.exp_lr(epoch, **self.scheduler_kwargs)
            elif self.lr_scheduler == "cosine":
                self.nn.cosine_lr(epoch, **self.scheduler_kwargs)
            elif self.lr_scheduler == "warmup_cosine":
                self.nn.warmup_cosine_lr(epoch, **self.scheduler_kwargs)

            indices = np.random.permutation(n)
            total_loss, n_batches = 0, 0
            for start in range(0, n, self.batch_size):
                if self._stop:
                    break
                idx = indices[start:start + self.batch_size]
                loss = self.nn.train_step(self.X[idx], self.y[idx])
                total_loss += loss; n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self.nn.loss_history.append(avg_loss)

            if self.X_val is not None:
                val_pred = self.nn.predict(self.X_val)
                loss_fn, _ = LOSS_MAP[self.nn.loss_name]
                self.nn.val_loss_history.append(loss_fn(val_pred, self.y_val))

            pred = self.nn.predict(self.X)
            acc = float(np.mean(((pred > 0.5).astype(float)) == self.y))
            self.nn.accuracy_history.append(acc)
            fwd = self.nn.get_forward_data(self.X) if hasattr(self.nn, 'get_forward_data') else None
            self.epoch_done.emit(epoch + 1, avg_loss, acc, pred, fwd)
        self.finished_signal.emit()


# ════════════════════════════════════════════════════════════════
#  PAGE 1: PLAYGROUND
# ════════════════════════════════════════════════════════════════

class PlaygroundPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nn = None; self.thread = None; self.X = None; self.y = None
        self.is_training = False
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget(); left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)

        ds_group = QGroupBox("Dataset")
        ds_form = QFormLayout()
        self.ds_combo = QComboBox(); self.ds_combo.addItems(list(DATASETS.keys()))
        self.ds_combo.currentTextChanged.connect(self._load_dataset)
        ds_form.addRow("Dataset:", self.ds_combo)
        ds_group.setLayout(ds_form); left_layout.addWidget(ds_group)

        arch_group = QGroupBox("Architecture")
        arch_form = QFormLayout()
        self.hidden_spin = QSpinBox(); self.hidden_spin.setRange(0, 5); self.hidden_spin.setValue(2)
        arch_form.addRow("Hidden Layers:", self.hidden_spin)
        self.neurons_spin = QSpinBox(); self.neurons_spin.setRange(1, 64); self.neurons_spin.setValue(8)
        arch_form.addRow("Neurons/Layer:", self.neurons_spin)
        self.act_combo = QComboBox(); self.act_combo.addItems(["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "GELU", "Swish"])
        arch_form.addRow("Activation:", self.act_combo)
        self.output_act_combo = QComboBox(); self.output_act_combo.addItems(["Sigmoid", "Linear", "Softmax"])
        arch_form.addRow("Output Act:", self.output_act_combo)
        self.loss_combo = QComboBox(); self.loss_combo.addItems(["MSE", "Binary Cross-Entropy", "Cross-Entropy"])
        self.loss_combo.setCurrentIndex(1); arch_form.addRow("Loss:", self.loss_combo)
        self.opt_combo = QComboBox(); self.opt_combo.addItems(["SGD", "Adam"]); self.opt_combo.setCurrentIndex(1)
        arch_form.addRow("Optimizer:", self.opt_combo)
        self.dropout_spin = QDoubleSpinBox(); self.dropout_spin.setRange(0, 0.8); self.dropout_spin.setValue(0)
        self.dropout_spin.setSingleStep(0.1); self.dropout_spin.setDecimals(2)
        arch_form.addRow("Dropout:", self.dropout_spin)
        self.l2_spin = QDoubleSpinBox(); self.l2_spin.setRange(0, 0.1); self.l2_spin.setValue(0)
        self.l2_spin.setSingleStep(0.001); self.l2_spin.setDecimals(4)
        arch_form.addRow("L2 lambda:", self.l2_spin)
        arch_group.setLayout(arch_form); left_layout.addWidget(arch_group)

        hp_group = QGroupBox("Hyperparameters")
        hp_form = QFormLayout()
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(0.0001, 1.0); self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4); self.lr_spin.setSingleStep(0.001)
        hp_form.addRow("Learning Rate:", self.lr_spin)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(10, 10000); self.epochs_spin.setValue(500)
        self.epochs_spin.setSingleStep(100); hp_form.addRow("Epochs:", self.epochs_spin)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 256); self.batch_spin.setValue(32)
        hp_form.addRow("Batch Size:", self.batch_spin)
        hp_group.setLayout(hp_form); left_layout.addWidget(hp_group)

        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("▶  Train"); self.train_btn.setObjectName("accent")
        self.train_btn.clicked.connect(self._toggle_training); btn_layout.addWidget(self.train_btn)
        self.reset_btn = QPushButton("↺  Reset"); self.reset_btn.setObjectName("danger")
        self.reset_btn.clicked.connect(self._reset); btn_layout.addWidget(self.reset_btn)
        self.save_btn = QPushButton("Save Plot"); self.save_btn.clicked.connect(self._save_plots)
        btn_layout.addWidget(self.save_btn)
        left_layout.addLayout(btn_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background-color: #21262d; border: 1px solid #30363d; border-radius: 4px; text-align: center; color: #e1e4e8; font-size: 10px; }
            QProgressBar::chunk { background-color: #1f6feb; border-radius: 3px; }
        """)
        left_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready. Select a dataset and build a network.")
        self.status_label.setWordWrap(True); self.status_label.setObjectName("hint")
        left_layout.addWidget(self.status_label); left_layout.addStretch()

        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel)
        top_splitter = QSplitter(Qt.Horizontal)
        self.loss_canvas = MplCanvas(self, width=4, height=3)
        self.loss_ax = self.loss_canvas.fig.add_subplot(111); self.style_loss_chart()
        top_splitter.addWidget(self.loss_canvas)
        self.network_diagram = NetworkDiagramWidget()
        top_splitter.addWidget(self.network_diagram); top_splitter.setSizes([500, 500])
        right_layout.addWidget(top_splitter, stretch=1)
        self.boundary_canvas = MplCanvas(self, width=5, height=4)
        self.boundary_ax = self.boundary_canvas.fig.add_subplot(111); self._init_boundary_chart()
        right_layout.addWidget(self.boundary_canvas, stretch=1)
        main_layout.addWidget(left_panel); main_layout.addWidget(right_panel, stretch=1)
        self._load_dataset()

    def style_loss_chart(self):
        self.loss_canvas.fig.clear()
        self.loss_ax = self.loss_canvas.fig.add_subplot(111)
        self.loss_canvas.style_axes(self.loss_ax, "Training Loss", "Epoch", "Loss")
        self.loss_canvas.fig.tight_layout(); self.loss_canvas.draw()

    def _init_boundary_chart(self):
        self.boundary_canvas.fig.clear()
        self.boundary_ax = self.boundary_canvas.fig.add_subplot(111)
        self.loss_canvas.style_axes(self.boundary_ax, "Decision Boundary", "x1", "x2")
        self.boundary_canvas.fig.tight_layout(); self.boundary_canvas.draw()

    def _load_dataset(self):
        self.X, self.y = DATASETS[self.ds_combo.currentText()]()
        self._plot_data_only()

    def _plot_data_only(self):
        ax = self.boundary_ax; self.boundary_canvas.fig.clear()
        self.boundary_ax = self.boundary_canvas.fig.add_subplot(111); ax = self.boundary_ax
        self.boundary_canvas.style_axes(ax, "Decision Boundary", "x1", "x2")
        c0, c1 = self.y.flatten() == 0, self.y.flatten() == 1
        ax.scatter(self.X[c0, 0], self.X[c0, 1], c="#58a6ff", s=20, alpha=0.7, edgecolors="none", label="Class 0")
        ax.scatter(self.X[c1, 0], self.X[c1, 1], c="#f85149", s=20, alpha=0.7, edgecolors="none", label="Class 1")
        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8")
        self.boundary_canvas.fig.tight_layout(); self.boundary_canvas.draw()

    def _build_network(self):
        n_hidden = self.hidden_spin.value(); n_neurons = self.neurons_spin.value()
        hidden_act = self.act_combo.currentText(); output_act = self.output_act_combo.currentText()
        drop_p = self.dropout_spin.value(); l2 = self.l2_spin.value()
        layer_sizes = [(self.X.shape[1], "Input")]
        for _ in range(n_hidden):
            layer_sizes.append((n_neurons, hidden_act))
        layer_sizes.append((1, output_act))
        self.nn = NeuralNetwork(loss_name=self.loss_combo.currentText(), lr=self.lr_spin.value(),
                                optimizer=self.opt_combo.currentText().lower(), weight_init="he", l2_lambda=l2)
        self.nn.build_dense(layer_sizes)
        # Insert dropout after each hidden Dense layer
        if drop_p > 0:
            new_layers = []
            for i, l in enumerate(self.nn.layers):
                new_layers.append(l)
                if i < len(self.nn.layers) - 1:  # not after output
                    new_layers.append(DropoutLayer(drop_p))
            self.nn.layers = new_layers
        self.network_diagram.set_network(self.nn.layer_sizes)
        self.status_label.setText(f"Network: {' -> '.join(f'{n}({a})' for n, a in self.nn.layer_sizes)}")

    def _toggle_training(self):
        self._stop_training() if self.is_training else self._start_training()

    def _start_training(self):
        self._build_network(); self.is_training = True
        self.train_btn.setText("⏹  Stop")
        for w in [self.ds_combo, self.hidden_spin, self.neurons_spin, self.dropout_spin, self.l2_spin]:
            w.setEnabled(False)
        self.thread = TrainingThread(self.nn, self.X, self.y, epochs=self.epochs_spin.value(),
                                     batch_size=self.batch_spin.value())
        self.thread.epoch_done.connect(self._on_epoch)
        self.thread.finished_signal.connect(self._on_training_done)
        self.thread.start()

    def _stop_training(self):
        self.is_training = False
        if self.thread: self.thread.stop(); self.thread.wait(2000)
        self._on_training_done()

    def _reset(self):
        self._stop_training(); self.nn = None; self._plot_data_only(); self.style_loss_chart()
        self.network_diagram.set_network([]); self.train_btn.setText("▶  Train")
        for w in [self.ds_combo, self.hidden_spin, self.neurons_spin, self.dropout_spin, self.l2_spin]:
            w.setEnabled(True)
        self.status_label.setText("Reset. Ready to train.")

    def _save_plots(self):
        import os
        saved = []
        try: saved.append(self.loss_canvas.save_plot("training_loss"))
        except Exception: pass
        try: saved.append(self.boundary_canvas.save_plot("decision_boundary"))
        except Exception: pass
        self.status_label.setText(f"Saved {len(saved)} plot(s) to Downloads/" if saved else "No plots yet.")

    def _on_epoch(self, epoch, loss, accuracy, pred, fwd_data):
        self.status_label.setText(f"Epoch {epoch}  |  Loss: {loss:.6f}  |  Acc: {accuracy:.1%}")
        self.progress_bar.setValue(int(100 * epoch / max(self.epochs_spin.value(), 1)))
        if epoch % 5 == 0 or epoch == 1:
            self.loss_canvas.fig.clear()
            if hasattr(self.nn, 'accuracy_history') and len(self.nn.accuracy_history) > 1:
                ax1 = self.loss_canvas.fig.add_subplot(211)
                ax2 = self.loss_canvas.fig.add_subplot(212)
                self.loss_canvas.style_axes(ax1, "Training Loss", "Epoch", "Loss")
                ax1.plot(self.nn.loss_history, color="#58a6ff", linewidth=1.2, label="Train")
                if self.nn.val_loss_history:
                    ax1.plot(self.nn.val_loss_history, color="#f0883e", linewidth=1.2, linestyle="--", label="Val")
                ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
                ax1.grid(True, alpha=0.15, color="#8b949e")
                self.loss_canvas.style_axes(ax2, "Accuracy", "Epoch", "Acc")
                ax2.plot(self.nn.accuracy_history, color="#3fb950", linewidth=1.5)
                ax2.set_ylim(0, 1.05)
                ax2.grid(True, alpha=0.15, color="#8b949e")
            else:
                ax1 = self.loss_canvas.fig.add_subplot(111)
                self.loss_canvas.style_axes(ax1, "Training Loss", "Epoch", "Loss")
                ax1.plot(self.nn.loss_history, color="#58a6ff", linewidth=1.5, label="Train")
                ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
                ax1.grid(True, alpha=0.15, color="#8b949e")
                self.loss_ax = ax1
            self.loss_canvas.fig.tight_layout(); self.loss_canvas.draw()
        if epoch % 10 == 0 or epoch == 1:
            self._plot_decision_boundary(pred)
        if fwd_data:
            mean_acts = [fwd_data[0]["a_prev"].mean(axis=0)]
            for fd in fwd_data: mean_acts.append(fd["a"].mean(axis=0))
            self.network_diagram.set_network(self.nn.layer_sizes, activations=mean_acts)

    def _plot_decision_boundary(self, pred):
        ax = self.boundary_ax; self.boundary_canvas.fig.clear()
        self.boundary_ax = self.boundary_canvas.fig.add_subplot(111); ax = self.boundary_ax
        self.boundary_canvas.style_axes(ax, "Decision Boundary", "x1", "x2")
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
        Z = self.nn.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors=["#e1e4e8"], linewidths=1.5)
        c0, c1 = self.y.flatten() == 0, self.y.flatten() == 1
        ax.scatter(self.X[c0, 0], self.X[c0, 1], c="#58a6ff", s=20, alpha=0.8, edgecolors="white", linewidths=0.3)
        ax.scatter(self.X[c1, 0], self.X[c1, 1], c="#f85149", s=20, alpha=0.8, edgecolors="white", linewidths=0.3)
        self.boundary_canvas.fig.tight_layout(); self.boundary_canvas.draw()

    def _on_training_done(self):
        self.is_training = False
        self.progress_bar.setValue(100)
        self.train_btn.setText("▶  Train")
        for w in [self.ds_combo, self.hidden_spin, self.neurons_spin, self.dropout_spin, self.l2_spin]:
            w.setEnabled(True)
        if self.nn and self.nn.loss_history:
            if hasattr(self.nn, 'accuracy_history') and self.nn.accuracy_history:
                self.status_label.setText(f"Done. Loss: {self.nn.loss_history[-1]:.6f}  |  Acc: {self.nn.accuracy_history[-1]:.1%}")
            else:
                self.status_label.setText(f"Done. Loss: {self.nn.loss_history[-1]:.6f}")


# ════════════════════════════════════════════════════════════════
#  PAGE 2: FORWARD PASS EXPLORER
# ════════════════════════════════════════════════════════════════

class ForwardPassPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nn = None; self.fwd_data = None; self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Network Architecture"))
        self.network_diagram = NetworkDiagramWidget(); left_layout.addWidget(self.network_diagram)
        main_layout.addWidget(left_panel, stretch=2)
        right_panel = QWidget(); right_panel.setFixedWidth(420); right_layout = QVBoxLayout(right_panel)
        ctrl_group = QGroupBox("Controls"); ctrl_layout = QFormLayout()
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["2->4->1 (XOR Simple)", "2->8->4->1 (XOR Deep)", "2->6->3->1 (Custom)"])
        self.arch_combo.currentIndexChanged.connect(self._build_and_forward)
        ctrl_layout.addRow("Architecture:", self.arch_combo)
        self.input_x = QDoubleSpinBox(); self.input_x.setRange(-2, 2); self.input_x.setValue(0.5)
        self.input_x.setSingleStep(0.1); self.input_x.valueChanged.connect(self._run_forward)
        ctrl_layout.addRow("Input x1:", self.input_x)
        self.input_y = QDoubleSpinBox(); self.input_y.setRange(-2, 2); self.input_y.setValue(0.8)
        self.input_y.setSingleStep(0.1); self.input_y.valueChanged.connect(self._run_forward)
        ctrl_layout.addRow("Input x2:", self.input_y)
        ctrl_group.setLayout(ctrl_layout); right_layout.addWidget(ctrl_group)
        nav_group = QGroupBox("Layer Steps"); nav_layout = QVBoxLayout()
        self.step_combo = QComboBox(); self.step_combo.currentIndexChanged.connect(self._show_step)
        nav_layout.addWidget(self.step_combo); nav_group.setLayout(nav_layout); right_layout.addWidget(nav_group)
        detail_group = QGroupBox("Computation Details"); detail_layout = QVBoxLayout()
        self.detail_text = QTextEdit(); self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("Consolas", 11)); detail_layout.addWidget(self.detail_text)
        detail_group.setLayout(detail_layout); right_layout.addWidget(detail_group)
        right_layout.addStretch(); main_layout.addWidget(right_panel)
        self._build_and_forward()

    def _build_and_forward(self):
        architectures = [
            [(2, "Input"), (4, "ReLU"), (1, "Sigmoid")],
            [(2, "Input"), (8, "ReLU"), (4, "ReLU"), (1, "Sigmoid")],
            [(2, "Input"), (6, "Tanh"), (3, "Tanh"), (1, "Sigmoid")],
        ]
        self.nn = NeuralNetwork(loss_name="MSE", lr=0.01, weight_init="xavier")
        self.nn.build_dense(architectures[self.arch_combo.currentIndex()])
        self.network_diagram.set_network(self.nn.layer_sizes); self._run_forward()

    def _run_forward(self):
        if self.nn is None: return
        X = np.array([[self.input_x.value(), self.input_y.value()]])
        self.nn.predict(X)
        self.fwd_data = self.nn.get_forward_data(X)
        self.step_combo.blockSignals(True); self.step_combo.clear()
        self.step_combo.addItem("0: Input Layer")
        for i, fd in enumerate(self.fwd_data):
            self.step_combo.addItem(f"{i+1}: {fd['activation_name']} ({fd['W'].shape[1]} neurons)")
        self.step_combo.addItem(f"{len(self.fwd_data)+1}: Output")
        self.step_combo.blockSignals(False); self.step_combo.setCurrentIndex(len(self.fwd_data))
        mean_acts = [self.fwd_data[0]["a_prev"][0]]
        for fd in self.fwd_data: mean_acts.append(fd["a"][0])
        self.network_diagram.set_network(self.nn.layer_sizes, activations=mean_acts)

    def _show_step(self, idx):
        if not self.fwd_data: return
        text = ""
        if idx == 0:
            text = f"=== INPUT LAYER ===\n\n  x1 = {self.input_x.value():.4f}\n  x2 = {self.input_y.value():.4f}\n\nInput vector: a0 = [x1, x2]\nThis is fed directly to the first hidden layer."
        elif idx <= len(self.fwd_data):
            fd = self.fwd_data[idx - 1]
            text = f"=== LAYER {idx}: {fd['activation_name'].upper()} ===\n\n"
            text += f"  Input (a^{idx-1}):  {fd['a_prev'][0]}\n\n  Weights (W^{idx}):\n"
            for row in fd['W']: text += "    [" + "  ".join(f"{v:+.4f}" for v in row) + "]\n"
            text += f"\n  Biases (b^{idx}):  {fd['b'][0]}\n\n"
            text += f"  z^{idx} = a^{idx-1} . W^{idx} + b^{idx}\n  z^{idx} = {fd['z'][0]}\n\n"
            text += f"  a^{idx} = {fd['activation_name']}(z^{idx})\n  a^{idx} = {fd['a'][0]}"
        else:
            pred = self.fwd_data[-1]["a"][0]
            text = f"=== OUTPUT ===\n\n  Network output: {pred}\n\nFor XOR:\n  Output > 0.5 -> Class 1\n  Output <= 0.5 -> Class 0\n\nPrediction: Class {1 if pred > 0.5 else 0}\nConfidence: {abs(pred - 0.5) * 200:.1f}%"
        self.detail_text.setText(text)


# ════════════════════════════════════════════════════════════════
#  PAGE 3: BACKPROPAGATION VISUALIZER
# ════════════════════════════════════════════════════════════════

class BackpropPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.nn = None; self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Gradient Flow (color = gradient magnitude)"))
        self.gradient_diagram = NetworkDiagramWidget(); left_layout.addWidget(self.gradient_diagram)
        main_layout.addWidget(left_panel, stretch=2)
        right_panel = QWidget(); right_panel.setFixedWidth(420); right_layout = QVBoxLayout(right_panel)
        ctrl_group = QGroupBox("Train & Inspect"); ctrl_form = QFormLayout()
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["2->4->1", "2->8->4->1", "2->16->8->4->1"])
        ctrl_form.addRow("Architecture:", self.arch_combo)
        self.dataset_combo = QComboBox(); self.dataset_combo.addItems(list(DATASETS.keys()))
        ctrl_form.addRow("Dataset:", self.dataset_combo)
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(0.0001, 1.0); self.lr_spin.setValue(0.05)
        self.lr_spin.setDecimals(4); self.lr_spin.setSingleStep(0.01)
        ctrl_form.addRow("Learning Rate:", self.lr_spin)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 5000); self.epochs_spin.setValue(100)
        ctrl_form.addRow("Epochs:", self.epochs_spin)
        ctrl_group.setLayout(ctrl_form); right_layout.addWidget(ctrl_group)
        self.train_btn = QPushButton("Train"); self.train_btn.setObjectName("accent")
        self.train_btn.clicked.connect(self._train); right_layout.addWidget(self.train_btn)
        detail_group = QGroupBox("Gradient Details"); detail_layout = QVBoxLayout()
        self.detail_text = QTextEdit(); self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("Consolas", 10)); detail_layout.addWidget(self.detail_text)
        detail_group.setLayout(detail_layout); right_layout.addWidget(detail_group)
        right_layout.addStretch(); main_layout.addWidget(right_panel)

    def _train(self):
        archs = [
            [(2, "Input"), (4, "ReLU"), (1, "Sigmoid")],
            [(2, "Input"), (8, "ReLU"), (4, "ReLU"), (1, "Sigmoid")],
            [(2, "Input"), (16, "ReLU"), (8, "ReLU"), (4, "ReLU"), (1, "Sigmoid")],
        ]
        self.nn = NeuralNetwork(loss_name="MSE", lr=self.lr_spin.value(), optimizer="adam")
        self.nn.build_dense(archs[self.arch_combo.currentIndex()])
        X, y = DATASETS[self.dataset_combo.currentText()]()
        for epoch in range(self.epochs_spin.value()): self.nn.train_step(X, y)
        fwd_data = self.nn.get_forward_data(X)
        grads = [{"dW": fd["dW"], "db": fd["db"]} for fd in fwd_data]
        self.gradient_diagram.set_network(self.nn.layer_sizes, gradients=grads)
        text = f"Trained for {self.epochs_spin.value()} epochs\nFinal loss: {self.nn.loss_history[-1]:.6f}\n\n"
        for fd in fwd_data:
            text += f"=== Layer {fd['index']+1} ({fd['activation_name']}) ===\n"
            text += f"  dW mean: {np.mean(np.abs(fd['dW'])):.6f}  max: {np.max(np.abs(fd['dW'])):.6f}\n"
            text += f"  W  mean: {np.mean(fd['W']):.4f}  std: {np.std(fd['W']):.4f}\n"
            gm = np.mean(np.abs(fd['dW']))
            if gm < 1e-6: text += "  !! Vanishing gradient!\n"
            elif gm > 10: text += "  !! Potentially exploding gradient!\n"
            else: text += "  Gradient magnitude healthy\n"
            text += "\n"
        self.detail_text.setText(text)


# ════════════════════════════════════════════════════════════════
#  PAGE 4: ACTIVATION FUNCTIONS
# ════════════════════════════════════════════════════════════════

ACT_DESCRIPTIONS = {
    "Sigmoid": "<h3 style='color:#58a6ff'>Sigmoid</h3><p><b>Formula:</b> s(x) = 1 / (1 + e<sup>-x</sup>)</p><p><b>Range:</b> (0, 1) - Output not zero-centered. Suffers from <span style='color:#f85149'>vanishing gradients</span> for large |x|. Good for binary output gates.</p>",
    "Tanh": "<h3 style='color:#58a6ff'>Tanh</h3><p><b>Formula:</b> tanh(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)</p><p><b>Range:</b> (-1, 1) - Zero-centered, better than sigmoid. Still <span style='color:#f85149'>vanishes</span> at extremes.</p>",
    "ReLU": "<h3 style='color:#58a6ff'>ReLU</h3><p><b>Formula:</b> f(x) = max(0, x)</p><p><b>Range:</b> [0, inf) - Most popular. Fast, <span style='color:#238636'>sparse activations</span>. Watch for <span style='color:#f0883e'>dying ReLU</span>.</p>",
    "LeakyReLU": "<h3 style='color:#58a6ff'>Leaky ReLU</h3><p><b>Formula:</b> f(x) = x if x > 0, else 0.01x</p><p><b>Range:</b> (-inf, inf) - Fixes dying ReLU by allowing small gradient for negative inputs.</p>",
    "ELU": "<h3 style='color:#58a6ff'>ELU</h3><p><b>Formula:</b> f(x) = x if x > 0, else a(e<sup>x</sup>-1)</p><p><b>Range:</b> (-a, inf) - Smooth for negative values, pushes mean activations toward zero.</p>",
    "GELU": "<h3 style='color:#58a6ff'>GELU</h3><p><b>Formula:</b> f(x) = 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x<sup>3</sup>)))</p><p>Used in <b>Transformers, BERT, GPT</b>. Smooth approximation of ReLU with non-zero gradient everywhere.</p>",
    "Swish": "<h3 style='color:#58a6ff'>Swish (SiLU)</h3><p><b>Formula:</b> f(x) = x * sigmoid(x)</p><p>Used in <b>EfficientNet, LLaMA</b>. Self-gated, non-monotonic. Often outperforms ReLU in deep networks.</p>",
    "Linear": "<h3 style='color:#58a6ff'>Linear</h3><p><b>Formula:</b> f(x) = x</p><p>No non-linearity. Stacking linear layers is equivalent to a single linear layer.</p>",
}

class ActivationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Select Activation:"))
        self.act_combo = QComboBox()
        self.act_combo.addItems(["Sigmoid", "Tanh", "ReLU", "LeakyReLU", "ELU", "GELU", "Swish", "Linear"])
        self.act_combo.currentTextChanged.connect(self._update_plot); ctrl_layout.addWidget(self.act_combo)
        self.show_deriv_cb = QCheckBox("Show Derivative"); self.show_deriv_cb.setChecked(True)
        self.show_deriv_cb.stateChanged.connect(self._update_plot); ctrl_layout.addWidget(self.show_deriv_cb)
        ctrl_layout.addStretch(); layout.addLayout(ctrl_layout)
        self.canvas = MplCanvas(self, width=8, height=5); layout.addWidget(self.canvas)
        self.desc_text = QTextEdit(); self.desc_text.setReadOnly(True); self.desc_text.setMaximumHeight(100)
        layout.addWidget(self.desc_text); self._update_plot()

    def _update_plot(self):
        name = self.act_combo.currentText(); show_deriv = self.show_deriv_cb.isChecked()
        act_fn, deriv_fn = ACTIVATION_MAP[name]
        self.canvas.fig.clear(); ax = self.canvas.fig.add_subplot(111)
        self.canvas.style_axes(ax, f"{name} Activation Function", "x", "f(x)")
        x = np.linspace(-6, 6, 500); y = act_fn(x)
        ax.plot(x, y, color="#58a6ff", linewidth=2.5, label=f"{name}(x)")
        ax.axhline(y=0, color="#30363d", linewidth=0.8); ax.axvline(x=0, color="#30363d", linewidth=0.8)
        if show_deriv and deriv_fn:
            if name in ("Swish", "GELU"):
                dy = deriv_fn(x, y)
            else:
                dy = deriv_fn(y)
            ax.plot(x, dy, color="#f85149", linewidth=2, linestyle="--", label=f"{name}'(x)")
        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=11)
        ax.grid(True, alpha=0.15, color="#8b949e")
        self.canvas.fig.tight_layout(); self.canvas.draw()
        self.desc_text.setHtml(ACT_DESCRIPTIONS.get(name, ""))


# ════════════════════════════════════════════════════════════════
#  PAGE 5: WEIGHT INITIALIZATION
# ════════════════════════════════════════════════════════════════

class WeightInitPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Architecture:"))
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["2->256->256->1 (Deep)", "2->32->1 (Shallow)", "2->64->64->64->1 (Very Deep)"])
        ctrl_layout.addWidget(self.arch_combo)
        self.run_btn = QPushButton("Train & Compare"); self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._run_comparison); ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addStretch(); layout.addLayout(ctrl_layout)
        self.canvas = MplCanvas(self, width=8, height=5); layout.addWidget(self.canvas)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setMaximumHeight(60)
        self.info_text.setHtml("<i style='color:#8b949e'>Compare initialization strategies on gradient flow.</i>")
        layout.addWidget(self.info_text); self._run_comparison()

    def _run_comparison(self):
        archs = [
            [(2,"Input"),(256,"Tanh"),(256,"Tanh"),(1,"Sigmoid")],
            [(2,"Input"),(32,"ReLU"),(1,"Sigmoid")],
            [(2,"Input"),(64,"ReLU"),(64,"ReLU"),(64,"ReLU"),(1,"Sigmoid")],
        ]
        arch = archs[self.arch_combo.currentIndex()]
        inits = ["he", "xavier", "random (0.01)"]
        init_map = {"he": "he", "xavier": "xavier", "random (0.01)": "random"}
        X, y = DATASETS["Circle"]()
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(14, 4)
        axes = self.canvas.fig.subplots(1, 3)
        lcolors = ["#58a6ff", "#238636", "#f0883e", "#bc8cff", "#f85149"]
        for i, init_name in enumerate(inits):
            nn = NeuralNetwork(loss_name="MSE", lr=0.01, optimizer="adam", weight_init=init_map[init_name])
            nn.build_dense(arch); grad_norms = {j: [] for j in range(len(nn.layers))}
            for epoch in range(200):
                nn.train_step(X, y)
                for j, layer in enumerate(nn.layers):
                    if isinstance(layer, DenseLayer): grad_norms[j].append(np.mean(np.abs(layer.dW)))
            ax = axes[i]; self.canvas.style_axes(ax, f"{init_name}", "Epoch", "|dW| mean")
            for j in range(len(nn.layers)):
                if j in grad_norms: ax.plot(grad_norms[j], color=lcolors[j % len(lcolors)], linewidth=1.2, label=f"Layer {j+1}")
            ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
            ax.grid(True, alpha=0.15, color="#8b949e")
            fl = nn.loss_history[-1] if nn.loss_history else float("inf")
            ax.set_title(f"{init_name}\nLoss: {fl:.4f}", color="#e1e4e8", fontsize=10, fontweight="bold")
        self.canvas.fig.tight_layout(); self.canvas.draw()
        self.info_text.setHtml("<p style='color:#e1e4e8'>He init works best with ReLU; Xavier with Tanh/Sigmoid. Small random causes vanishing signals in deep nets.</p>")


# ════════════════════════════════════════════════════════════════
#  PAGE 6: RNN SEQUENCE PREDICTOR
# ════════════════════════════════════════════════════════════════

class RNNPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget(); left_panel.setFixedWidth(340); left_layout = QVBoxLayout(left_panel)

        ctrl_group = QGroupBox("RNN Configuration"); ctrl_form = QFormLayout()
        self.dataset_combo = QComboBox(); self.dataset_combo.addItems(list(SEQUENCE_DATASETS.keys()))
        ctrl_form.addRow("Task:", self.dataset_combo)
        self.model_combo = QComboBox(); self.model_combo.addItems(["LSTM", "Simple RNN"])
        ctrl_form.addRow("Model:", self.model_combo)
        self.hidden_spin = QSpinBox(); self.hidden_spin.setRange(4, 128); self.hidden_spin.setValue(32)
        ctrl_form.addRow("Hidden Size:", self.hidden_spin)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(10, 2000); self.epochs_spin.setValue(200)
        ctrl_form.addRow("Epochs:", self.epochs_spin)
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(0.0001, 0.1); self.lr_spin.setValue(0.005)
        self.lr_spin.setDecimals(4); ctrl_form.addRow("Learning Rate:", self.lr_spin)
        ctrl_group.setLayout(ctrl_form); left_layout.addWidget(ctrl_group)

        self.train_btn = QPushButton("Train RNN"); self.train_btn.setObjectName("accent")
        self.train_btn.clicked.connect(self._train); left_layout.addWidget(self.train_btn)
        self.status_label = QLabel("Configure and train a recurrent network."); self.status_label.setWordWrap(True)
        self.status_label.setObjectName("hint"); left_layout.addWidget(self.status_label)
        left_layout.addStretch(); main_layout.addWidget(left_panel)

        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel)
        self.loss_canvas = MplCanvas(self, width=6, height=3)
        self.pred_canvas = MplCanvas(self, width=6, height=3)
        right_layout.addWidget(self.loss_canvas, stretch=1)
        right_layout.addWidget(self.pred_canvas, stretch=1)
        main_layout.addWidget(right_panel, stretch=1)

    def _train(self):
        self.status_label.setText("Training... please wait")
        QApplication.processEvents()
        X, y = SEQUENCE_DATASETS[self.dataset_combo.currentText()]()
        input_size = X.shape[2]
        output_size = y.shape[-1] if y.ndim == 3 else y.shape[1]
        seq_len = X.shape[1]
        hidden_size = self.hidden_spin.value()

        # Build model: RNN/LSTM -> Dense
        ModelClass = LSTMLayer if self.model_combo.currentText() == "LSTM" else SimpleRNNLayer
        act = "Tanh" if self.model_combo.currentText() == "Simple RNN" else None
        if act:
            rnn = ModelClass(input_size, hidden_size, activation=act, weight_init="xavier")
        else:
            rnn = ModelClass(input_size, hidden_size, weight_init="xavier")
        dense = DenseLayer(hidden_size, output_size, "Linear", "xavier")

        nn = NeuralNetwork(loss_name="MSE", lr=self.lr_spin.value(), optimizer="adam")
        nn.build_custom([rnn, dense])

        epochs = self.epochs_spin.value()
        # Target: last timestep of y for next-step prediction
        if y.ndim == 3:
            y_flat = y[:, -1, :]
        else:
            y_flat = y

        for epoch in range(epochs):
            loss = nn.train_step(X, y_flat)
            if (epoch + 1) % 50 == 0:
                self.status_label.setText(f"Epoch {epoch+1}/{epochs}  Loss: {loss:.6f}")
                QApplication.processEvents()

        if hasattr(nn, 'accuracy_history') and nn.accuracy_history:
            self.status_label.setText(f"Done. Loss: {nn.loss_history[-1]:.6f}  |  Acc: {nn.accuracy_history[-1]:.1%}")
        else:
            self.status_label.setText(f"Done. Loss: {nn.loss_history[-1]:.6f}")

        # Plot loss
        ax = self.loss_canvas.fig.clear()
        ax = self.loss_canvas.fig.add_subplot(111)
        self.loss_canvas.style_axes(ax, "Training Loss", "Epoch", "Loss")
        ax.plot(nn.loss_history, color="#58a6ff", linewidth=1.5)
        ax.grid(True, alpha=0.15, color="#8b949e")
        self.loss_canvas.fig.tight_layout(); self.loss_canvas.draw()

        # Plot predictions
        ax = self.pred_canvas.fig.clear()
        ax = self.pred_canvas.fig.add_subplot(111)
        self.pred_canvas.style_axes(ax, "Sequence Predictions", "Time Step", "Value")

        pred = nn.predict(X)
        if pred.ndim == 3:
            pred = pred[:, -1, :]
        n_show = min(5, X.shape[0])
        true_colors = ["#238636", "#58a6ff", "#f0883e", "#bc8cff", "#f85149"]
        pred_colors = ["#3fb950", "#79c0ff", "#f778ba", "#d2a8ff", "#ff7b72"]

        for k in range(n_show):
            if y.ndim == 3:
                true_seq = y[k, :, 0]
                pred_seq = pred[k].reshape(-1) if pred[k].ndim > 1 else np.full(seq_len, pred[k, 0])
            else:
                true_seq = y[k, 0] * np.ones(seq_len)
                pred_seq = np.full(seq_len, pred[k, 0])
            ax.plot(range(seq_len), true_seq, color=true_colors[k], linewidth=1.5, alpha=0.7, label=f"True {k+1}")
            ax.plot(range(seq_len), pred_seq, color=pred_colors[k], linewidth=1.5, linestyle="--", alpha=0.7, label=f"Pred {k+1}")

        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.15, color="#8b949e")
        self.pred_canvas.fig.tight_layout(); self.pred_canvas.draw()


# ════════════════════════════════════════════════════════════════
#  PAGE 7: CNN / CONV1D EXPLORER
# ════════════════════════════════════════════════════════════════

class CNNPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget(); left_panel.setFixedWidth(300); left_layout = QVBoxLayout(left_panel)
        ctrl_group = QGroupBox("Conv1D Configuration"); ctrl_form = QFormLayout()
        self.out_channels_spin = QSpinBox(); self.out_channels_spin.setRange(1, 16); self.out_channels_spin.setValue(4)
        ctrl_form.addRow("Filters:", self.out_channels_spin)
        self.kernel_spin = QSpinBox(); self.kernel_spin.setRange(2, 9); self.kernel_spin.setValue(3)
        ctrl_form.addRow("Kernel Size:", self.kernel_spin)
        self.stride_spin = QSpinBox(); self.stride_spin.setRange(1, 3); self.stride_spin.setValue(1)
        ctrl_form.addRow("Stride:", self.stride_spin)
        self.padding_spin = QSpinBox(); self.padding_spin.setRange(0, 5); self.padding_spin.setValue(0)
        ctrl_form.addRow("Padding:", self.padding_spin)
        self.dense_spin = QSpinBox(); self.dense_spin.setRange(4, 64); self.dense_spin.setValue(16)
        ctrl_form.addRow("Dense Neurons:", self.dense_spin)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(10, 500); self.epochs_spin.setValue(50)
        ctrl_form.addRow("Epochs:", self.epochs_spin)
        ctrl_group.setLayout(ctrl_form); left_layout.addWidget(ctrl_group)
        self.run_btn = QPushButton("Train Conv1D"); self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._train); left_layout.addWidget(self.run_btn)
        self.status_label = QLabel("Configure and train a 1D CNN."); self.status_label.setWordWrap(True)
        self.status_label.setObjectName("hint"); left_layout.addWidget(self.status_label)
        left_layout.addStretch(); main_layout.addWidget(left_panel)
        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel)
        self.signal_canvas = MplCanvas(self, width=8, height=3)
        self.filter_canvas = MplCanvas(self, width=8, height=3)
        right_layout.addWidget(self.signal_canvas, stretch=1)
        right_layout.addWidget(self.filter_canvas, stretch=1)
        main_layout.addWidget(right_panel, stretch=1)

    def _train(self):
        self.status_label.setText("Training..."); QApplication.processEvents()
        X, y = make_conv_signal(n=80, seq_len=64)
        in_ch = X.shape[1]; out_ch = self.out_channels_spin.value()
        ks = self.kernel_spin.value(); stride = self.stride_spin.value()
        pad = self.padding_spin.value(); dense_n = self.dense_spin.value()

        conv = Conv1DLayer(in_ch, out_ch, ks, stride, pad, "ReLU", "he")
        out_len = (64 + 2 * pad - ks) // stride + 1
        flatten_dim = out_ch * out_len
        dense = DenseLayer(flatten_dim, dense_n, "ReLU", "he")
        output = DenseLayer(dense_n, 1, "Sigmoid", "xavier")

        nn = NeuralNetwork(loss_name="Binary Cross-Entropy", lr=0.005, optimizer="adam")
        nn.build_custom([conv, dense, output])

        for epoch in range(self.epochs_spin.value()):
            loss = nn.train_step(X, y)
            if (epoch + 1) % 10 == 0:
                self.status_label.setText(f"Epoch {epoch+1}  Loss: {loss:.6f}")
                QApplication.processEvents()

        self.status_label.setText(f"Done. Loss: {nn.loss_history[-1]:.6f} | Acc: {np.mean(((nn.predict(X) > 0.5).astype(float)) == y):.2%}")

        # Plot signals
        ax = self.signal_canvas.fig.clear()
        ax = self.signal_canvas.fig.add_subplot(111)
        self.signal_canvas.style_axes(ax, "Sample Signals with Detected Pulses", "Time", "Amplitude")
        pred = nn.predict(X)
        for k in range(min(4, X.shape[0])):
            ax.plot(X[k, 0, :], color="#58a6ff" if y[k, 0] < 0.5 else "#f85149", alpha=0.6, linewidth=0.8)
            ax.set_title(f"Sample (pred={pred[k, 0]:.3f})", color="#8b949e", fontsize=9)
        self.signal_canvas.fig.tight_layout(); self.signal_canvas.draw()

        # Plot filter weights
        ax = self.filter_canvas.fig.clear()
        n_filters = min(out_ch, 8)
        axes = self.filter_canvas.fig.subplots(n_filters, 1, sharex=True)
        if n_filters == 1: axes = [axes]
        for i in range(n_filters):
            self.filter_canvas.style_axes(axes[i], f"Filter {i+1}", "", "Weight")
            axes[i].stem(range(ks), conv.K[i, 0, :], linefmt="-", markerfmt="o", basefmt="-")
            axes[i].grid(True, alpha=0.15, color="#8b949e")
        self.filter_canvas.fig.tight_layout(); self.filter_canvas.draw()


# ════════════════════════════════════════════════════════════════
#  PAGE 8: LOSS LANDSCAPE EXPLORER
# ════════════════════════════════════════════════════════════════

class LossLandscapePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget(); left_panel.setFixedWidth(300); left_layout = QVBoxLayout(left_panel)
        ctrl_group = QGroupBox("Configuration"); ctrl_form = QFormLayout()
        self.ds_combo = QComboBox(); self.ds_combo.addItems(list(DATASETS.keys()))
        self.ds_combo.setCurrentText("Circle"); ctrl_form.addRow("Dataset:", self.ds_combo)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["2->4->1", "2->8->1", "2->4->4->1"])
        ctrl_form.addRow("Network:", self.arch_combo)
        self.resolution_spin = QSpinBox(); self.resolution_spin.setRange(20, 80); self.resolution_spin.setValue(40)
        ctrl_form.addRow("Resolution:", self.resolution_spin)
        ctrl_group.setLayout(ctrl_form); left_layout.addWidget(ctrl_group)
        self.run_btn = QPushButton("Compute Landscape"); self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._compute); left_layout.addWidget(self.run_btn)
        self.info_label = QLabel("Visualize the 2D loss surface by varying two weight parameters.\nRed = low loss, Blue = high loss.")
        self.info_label.setWordWrap(True); self.info_label.setObjectName("hint")
        left_layout.addWidget(self.info_label); left_layout.addStretch()
        main_layout.addWidget(left_panel)
        self.canvas = MplCanvas(self, width=8, height=6)
        main_layout.addWidget(self.canvas, stretch=1)

    def _compute(self):
        self.info_label.setText("Computing... (this may take a moment)"); QApplication.processEvents()
        X, y = DATASETS[self.ds_combo.currentText()]()
        archs = {
            "2->4->1": [(2,"Input"),(4,"ReLU"),(1,"Sigmoid")],
            "2->8->1": [(2,"Input"),(8,"ReLU"),(1,"Sigmoid")],
            "2->4->4->1": [(2,"Input"),(4,"ReLU"),(4,"ReLU"),(1,"Sigmoid")],
        }
        nn = NeuralNetwork(loss_name="MSE", lr=0.01, optimizer="sgd")
        nn.build_dense(archs[self.arch_combo.currentText()])

        # Choose 2 weights to vary: first layer W[0,0] and W[1,0]
        target_layer = nn.layers[0]
        w0_orig, w1_orig = target_layer.W[0, 0], target_layer.W[1, 0]

        res = self.resolution_spin.value()
        # First train a bit to find good center
        for _ in range(20): nn.train_step(X, y)
        w0_center, w1_center = target_layer.W[0, 0], target_layer.W[1, 0]

        range_val = max(abs(w0_center), abs(w1_center), 0.5) * 3
        w0_range = np.linspace(w0_center - range_val, w0_center + range_val, res)
        w1_range = np.linspace(w1_center - range_val, w1_center + range_val, res)
        loss_grid = np.zeros((res, res))

        loss_fn = LOSS_MAP["MSE"][0]
        for i, w0 in enumerate(w0_range):
            for j, w1 in enumerate(w1_range):
                target_layer.W[0, 0] = w0
                target_layer.W[1, 0] = w1
                pred = nn.forward(X)
                loss_grid[j, i] = loss_fn(pred, y)

        target_layer.W[0, 0] = w0_orig
        target_layer.W[1, 0] = w1_orig

        # Plot
        ax = self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self.canvas.style_axes(ax, "2D Loss Landscape", f"W[0,0] (first layer)", "W[1,0] (first layer)")
        im = ax.contourf(w0_range, w1_range, loss_grid, levels=30, cmap="RdYlBu_r")
        ax.contour(w0_range, w1_range, loss_grid, levels=15, colors="white", alpha=0.2, linewidths=0.5)
        ax.plot(w0_center, w1_center, "w*", markersize=15, label="Trained position")
        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8")
        self.canvas.fig.colorbar(im, ax=ax, label="Loss", shrink=0.8)
        self.canvas.fig.tight_layout(); self.canvas.draw()
        self.info_label.setText(f"Landscape computed. White star = trained weight position.\nLoss range: [{loss_grid.min():.4f}, {loss_grid.max():.4f}]")


# ════════════════════════════════════════════════════════════════
#  PAGE 9: REGULARIZATION EXPLORER
# ════════════════════════════════════════════════════════════════

class RegularizationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget(); left_panel.setFixedWidth(280); left_layout = QVBoxLayout(left_panel)
        ctrl_group = QGroupBox("Configuration"); ctrl_form = QFormLayout()
        self.ds_combo = QComboBox(); self.ds_combo.addItems(list(DATASETS.keys()))
        ctrl_form.addRow("Dataset:", self.ds_combo)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(50, 2000); self.epochs_spin.setValue(300)
        ctrl_form.addRow("Epochs:", self.epochs_spin)
        self.neurons_spin = QSpinBox(); self.neurons_spin.setRange(8, 128); self.neurons_spin.setValue(32)
        ctrl_form.addRow("Hidden Neurons:", self.neurons_spin)
        ctrl_group.setLayout(ctrl_form); left_layout.addWidget(ctrl_group)
        self.run_btn = QPushButton("Compare Regularization"); self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._run); left_layout.addWidget(self.run_btn)
        self.info_label = QLabel("Compare no reg vs L2 vs Dropout vs L2+Dropout.\nWatch for overfitting in the 'None' case.")
        self.info_label.setWordWrap(True); self.info_label.setObjectName("hint")
        left_layout.addWidget(self.info_label); left_layout.addStretch()
        main_layout.addWidget(left_panel)
        self.canvas = MplCanvas(self, width=8, height=5)
        main_layout.addWidget(self.canvas, stretch=1)

    def _run(self):
        self.info_label.setText("Training 4 models..."); QApplication.processEvents()
        X, y = DATASETS[self.ds_combo.currentText()]()
        n = len(X); n_train = n * 3 // 4
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]
        epochs = self.epochs_spin.value(); hidden = self.neurons_spin.value()

        configs = [
            ("None", 0.0, 0.0), ("L2 (0.01)", 0.0, 0.01),
            ("Dropout (0.3)", 0.3, 0.0), ("L2+Dropout", 0.3, 0.01),
        ]
        colors = ["#f85149", "#58a6ff", "#238636", "#f0883e"]

        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(12, 5)
        axes = self.canvas.fig.subplots(1, 2)

        for i, (name, drop, l2) in enumerate(configs):
            nn = NeuralNetwork(loss_name="MSE", lr=0.01, optimizer="adam", l2_lambda=l2)
            nn.build_dense([(X.shape[1], "Input"), (hidden, "ReLU"), (hidden // 2, "ReLU"), (1, "Sigmoid")])
            if drop > 0:
                new_layers = []
                for j, l in enumerate(nn.layers):
                    new_layers.append(l)
                    if j < len(nn.layers) - 1: new_layers.append(DropoutLayer(drop))
                nn.layers = new_layers

            for epoch in range(epochs):
                nn.train_step(X_train, y_train)
                val_pred = nn.predict(X_val)
                nn.val_loss_history.append(Loss.mse(val_pred, y_val))

            ax = axes[0]
            ax.plot(nn.loss_history, color=colors[i], linewidth=1.2, alpha=0.7, label=f"{name}")
            ax = axes[1]
            ax.plot(nn.val_loss_history, color=colors[i], linewidth=1.2, label=f"{name}")

        self.canvas.style_axes(axes[0], "Training Loss", "Epoch", "Loss")
        self.canvas.style_axes(axes[1], "Validation Loss", "Epoch", "Loss")
        for ax in axes:
            ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=9)
            ax.grid(True, alpha=0.15, color="#8b949e")
        self.canvas.fig.tight_layout(); self.canvas.draw()
        self.info_label.setText("Done. Compare train vs val loss. Large gap = overfitting.\nRegularization reduces the gap.")


# ════════════════════════════════════════════════════════════════
#  PAGE 10: LR SCHEDULER COMPARISON
# ════════════════════════════════════════════════════════════════

class LRSchedulerPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget(); left_panel.setFixedWidth(280); left_layout = QVBoxLayout(left_panel)
        ctrl_group = QGroupBox("Configuration"); ctrl_form = QFormLayout()
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(50, 2000); self.epochs_spin.setValue(300)
        ctrl_form.addRow("Epochs:", self.epochs_spin)
        ctrl_group.setLayout(ctrl_form); left_layout.addWidget(ctrl_group)
        self.run_btn = QPushButton("Compare Schedules"); self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._run); left_layout.addWidget(self.run_btn)
        self.info_label = QLabel("Compare Constant, Step, Exponential, Cosine,\nand Warmup+Cosine LR schedules.")
        self.info_label.setWordWrap(True); self.info_label.setObjectName("hint")
        left_layout.addWidget(self.info_label); left_layout.addStretch()
        main_layout.addWidget(left_panel)
        self.canvas = MplCanvas(self, width=8, height=5)
        main_layout.addWidget(self.canvas, stretch=1)

    def _run(self):
        self.info_label.setText("Training 5 models..."); QApplication.processEvents()
        X, y = DATASETS["Circle"]()
        epochs = self.epochs_spin.value()
        max_epochs = epochs

        schedulers = [
            ("Constant", None, {}),
            ("Step (50, 0.5)", "step", {"step_size": 50, "gamma": 0.5}),
            ("Exp (0.995)", "exp", {"gamma": 0.995}),
            ("Cosine", "cosine", {"max_epochs": max_epochs}),
            ("Warmup+Cosine", "warmup_cosine", {"max_epochs": max_epochs, "warmup_epochs": 20}),
        ]
        colors = ["#f85149", "#58a6ff", "#238636", "#f0883e", "#bc8cff"]

        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(14, 5)
        axes = self.canvas.fig.subplots(1, 3)

        for i, (name, sched, kwargs) in enumerate(schedulers):
            nn = NeuralNetwork(loss_name="MSE", lr=0.01, optimizer="sgd")
            nn.build_dense([(2, "Input"), (16, "ReLU"), (1, "Sigmoid")])
            for epoch in range(epochs):
                if sched == "step": nn.step_lr(epoch, **kwargs)
                elif sched == "exp": nn.exp_lr(epoch, **kwargs)
                elif sched == "cosine": nn.cosine_lr(epoch, **kwargs)
                elif sched == "warmup_cosine": nn.warmup_cosine_lr(epoch, **kwargs)
                nn.train_step(X, y)
            axes[0].plot(nn.lr_history, color=colors[i], linewidth=1.5, label=name)
            axes[1].plot(nn.loss_history, color=colors[i], linewidth=1.2, label=name)
            # Final accuracy
            pred = nn.predict(X)
            acc = np.mean(((pred > 0.5).astype(float)) == y)
            axes[2].bar(i, acc * 100, color=colors[i], alpha=0.8)

        self.canvas.style_axes(axes[0], "Learning Rate Schedule", "Epoch", "LR")
        self.canvas.style_axes(axes[1], "Training Loss", "Epoch", "Loss")
        self.canvas.style_axes(axes[2], "Final Accuracy", "Schedule", "Accuracy (%)")
        axes[2].set_xticks(range(len(schedulers)))
        axes[2].set_xticklabels([s[0] for s in schedulers], rotation=30, ha="right", fontsize=8)
        for ax in axes[:2]:
            ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
            ax.grid(True, alpha=0.15, color="#8b949e")
        self.canvas.fig.tight_layout(); self.canvas.draw()
        self.info_label.setText("Done. Cosine and Warmup+Cosine often converge best.\nStep decay is simple but effective.")


# ════════════════════════════════════════════════════════════════
#  PAGE 11: NETWORK ZOO (Architecture Comparison)
# ════════════════════════════════════════════════════════════════

class NetworkZooPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        left_panel = QWidget(); left_panel.setFixedWidth(280); left_layout = QVBoxLayout(left_panel)
        ctrl_group = QGroupBox("Compare Architectures"); ctrl_form = QFormLayout()
        self.ds_combo = QComboBox(); self.ds_combo.addItems(list(DATASETS.keys()))
        ctrl_form.addRow("Dataset:", self.ds_combo)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(50, 2000); self.epochs_spin.setValue(200)
        ctrl_form.addRow("Epochs:", self.epochs_spin)
        ctrl_group.setLayout(ctrl_form); left_layout.addWidget(ctrl_group)
        self.run_btn = QPushButton("Compare All"); self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._run); left_layout.addWidget(self.run_btn)
        self.info_label = QLabel("Compare different architectures on the same task.\nWatch how depth, width, and activation affect learning.")
        self.info_label.setWordWrap(True); self.info_label.setObjectName("hint")
        left_layout.addWidget(self.info_label); left_layout.addStretch()
        main_layout.addWidget(left_panel)
        self.canvas = MplCanvas(self, width=8, height=5)
        main_layout.addWidget(self.canvas, stretch=1)

    def _run(self):
        self.info_label.setText("Training multiple architectures..."); QApplication.processEvents()
        X, y = DATASETS[self.ds_combo.currentText()]()
        epochs = self.epochs_spin.value()

        architectures = [
            ("Linear (2->1)", [(2,"Input"),(1,"Linear")], "MSE", 0.01, "sgd"),
            ("Shallow ReLU (2->8->1)", [(2,"Input"),(8,"ReLU"),(1,"Sigmoid")], "MSE", 0.01, "adam"),
            ("Deep ReLU (2->16->8->1)", [(2,"Input"),(16,"ReLU"),(8,"ReLU"),(1,"Sigmoid")], "MSE", 0.01, "adam"),
            ("Wide Tanh (2->32->1)", [(2,"Input"),(32,"Tanh"),(1,"Sigmoid")], "MSE", 0.01, "adam"),
            ("Very Deep (2->8->8->8->8->1)", [(2,"Input"),(8,"ReLU"),(8,"ReLU"),(8,"ReLU"),(8,"ReLU"),(1,"Sigmoid")], "MSE", 0.005, "adam"),
            ("GELU (2->16->4->1)", [(2,"Input"),(16,"GELU"),(4,"GELU"),(1,"Sigmoid")], "MSE", 0.01, "adam"),
            ("Swish Wide (2->64->16->1)", [(2,"Input"),(64,"Swish"),(16,"Swish"),(1,"Sigmoid")], "MSE", 0.005, "adam"),
            ("ELU Tower (2->16x3->1)", [(2,"Input"),(16,"ELU"),(16,"ELU"),(16,"ELU"),(1,"Sigmoid")], "MSE", 0.008, "adam"),
        ]
        colors = ["#484f58", "#f85149", "#58a6ff", "#238636", "#f0883e", "#bc8cff", "#e3b341", "#79c0ff"]

        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(14, 5)
        axes = self.canvas.fig.subplots(1, 2)

        results = []
        for i, (name, arch, loss_name, lr, opt) in enumerate(architectures):
            nn = NeuralNetwork(loss_name=loss_name, lr=lr, optimizer=opt, weight_init="he")
            nn.build_dense(arch)
            for epoch in range(epochs): nn.train_step(X, y)
            pred = nn.predict(X); acc = np.mean(((pred > 0.5).astype(float)) == y)
            axes[0].plot(nn.loss_history, color=colors[i], linewidth=1.2, label=name)
            results.append((name, acc, nn.loss_history[-1]))

        # Bar chart of final accuracy
        names = [r[0] for r in results]; accs = [r[1] * 100 for r in results]
        bars = axes[1].barh(range(len(names)), accs, color=colors, alpha=0.85)
        axes[1].set_yticks(range(len(names))); axes[1].set_yticklabels(names, fontsize=9)
        for bar, acc in zip(bars, accs):
            axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                         f"{acc:.1f}%", va="center", color="#e1e4e8", fontsize=10)

        self.canvas.style_axes(axes[0], "Training Loss", "Epoch", "Loss")
        self.canvas.style_axes(axes[1], "Final Accuracy", "Accuracy (%)", "")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=7)
        for ax in axes:
            ax.grid(True, alpha=0.15, color="#8b949e")
        self.canvas.fig.tight_layout(); self.canvas.draw()
        text = "Results:\n" + "\n".join(f"  {n}: acc={a:.1%}  loss={l:.4f}" for n, a, l in results)
        self.info_label.setText(text)


# ════════════════════════════════════════════════════════════════
#  PAGE 12: LESSONS
# ════════════════════════════════════════════════════════════════

LESSONS = [
    {"title": "1. The Perceptron", "content": """
<h3>The Perceptron</h3>
<p>A <b>perceptron</b> multiplies inputs by <b>weights</b>, sums them, adds a <b>bias</b>, and passes through an <b>activation function</b>.</p>
<h4>Mathematical Formulation</h4>
<p style="color:#58a6ff; font-family:monospace; padding:8px; background:#161b22; border-radius:6px;">
  z = w1*x1 + w2*x2 + ... + wn*xn + b<br>a = activation(z)
</p>
<p>A single perceptron can only learn <b>linearly separable</b> problems. It cannot solve XOR!</p>
<h4>Try It</h4>
<p>Go to <b>Forward Pass Explorer</b> to see how a perceptron transforms inputs. Then visit <b>Playground</b> to see why one layer is not enough for XOR.</p>"""},
    {"title": "2. Multi-Layer Networks", "content": """
<h3>Multi-Layer Neural Networks</h3>
<p>Stacking layers creates <b>non-linear</b> decision boundaries. Each layer transforms the representation, extracting more abstract features.</p>
<h4>Universal Approximation Theorem</h4>
<p>A network with just <b>one hidden layer</b> can approximate any continuous function. But deeper networks learn more efficiently.</p>
<h4>Architecture Choices</h4>
<ul><li><b>Width vs. Depth:</b> Wide shallow networks can theoretically do anything, but deep networks are more parameter-efficient.</li>
<li><b>Activations:</b> Non-linear activations (ReLU, Tanh, GELU) give networks their power. Without them, stacking linear layers is useless.</li></ul>"""},
    {"title": "3. Backpropagation", "content": """
<h3>Backpropagation - The Chain Rule in Action</h3>
<p>Backprop computes <b>gradients</b> of the loss w.r.t. every weight using the <b>chain rule</b>, layer by layer, from output to input.</p>
<h4>Key Insights</h4>
<ul><li><b>Efficiency:</b> O(n) time (one forward + one backward pass).</li>
<li><b>Vanishing gradients:</b> In deep networks with sigmoid/tanh, gradients shrink exponentially. ReLU helps.</li>
<li><b>Exploding gradients:</b> Gradients can grow exponentially. Use proper initialization and gradient clipping.</li></ul>
<h4>Try It</h4>
<p>Visit the <b>Backpropagation Visualizer</b> to see gradient magnitudes at each layer. Try deep networks with different activations.</p>"""},
    {"title": "4. Loss Functions", "content": """
<h3>Loss Functions</h3>
<p><b>MSE:</b> L = (1/n) Sum((y_pred - y_true)^2) - Good for regression.<br>
<b>Binary Cross-Entropy:</b> L = -(1/n) Sum[y*log(y_hat) + (1-y)*log(1-y_hat)] - Standard for binary classification.<br>
<b>Cross-Entropy:</b> For multi-class with softmax output.</p>
<p>The shape of the loss landscape determines optimization difficulty. A well-chosen loss provides smoother gradients and faster convergence.</p>"""},
    {"title": "5. Optimizers", "content": """
<h3>Optimizers</h3>
<p><b>SGD:</b> W = W - lr * grad. Simple but can be slow.<br>
<b>Adam:</b> Combines momentum and adaptive learning rate per parameter. Generally the best default choice.</p>
<h4>Learning Rate</h4>
<ul><li>Too high: loss oscillates or diverges</li><li>Too low: very slow convergence</li></ul>
<p>Try the <b>LR Scheduler</b> page to compare constant, step, cosine, and warmup schedules.</p>"""},
    {"title": "6. Weight Initialization", "content": """
<h3>Weight Initialization</h3>
<p><b>Xavier/Glorot:</b> W ~ N(0, sqrt(1/n_in)) - For sigmoid/tanh.<br>
<b>He:</b> W ~ N(0, sqrt(2/n_in)) - For ReLU. Accounts for zeroing half the neurons.<br>
<b>Small Random:</b> W ~ N(0, 0.01) - Causes vanishing activations in deep networks.</p>
<p>Visit the <b>Weight Init Explorer</b> to compare their effects on gradient flow.</p>"""},
    {"title": "7. RNNs & LSTMs", "content": """
<h3>Recurrent Neural Networks</h3>
<p>RNNs process <b>sequences</b> by maintaining a hidden state that carries information across time steps.</p>
<p><b>Vanilla RNN:</b> h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b) - Simple but suffers from vanishing gradients over long sequences.</p>
<p><b>LSTM:</b> Uses forget, input, and output <b>gates</b> to control information flow. The cell state acts as a "highway" for gradients, solving the vanishing gradient problem.</p>
<h4>Try It</h4>
<p>Go to the <b>RNN</b> page to train a sequence predictor. Compare Simple RNN vs LSTM on sine wave prediction.</p>"""},
    {"title": "8. CNNs", "content": """
<h3>Convolutional Neural Networks</h3>
<p>CNNs use <b>convolutional filters</b> that slide across the input, detecting local patterns. Key ideas:</p>
<ul><li><b>Local connectivity:</b> Each neuron sees only a small region (the receptive field)</li>
<li><b>Weight sharing:</b> The same filter is applied everywhere, greatly reducing parameters</li>
<li><b>Translation invariance:</b> Patterns are detected regardless of position</li></ul>
<p>The <b>CNN</b> page lets you train a Conv1D network on pulse detection and visualize the learned filters.</p>"""},
    {"title": "9. Regularization", "content": """
<h3>Regularization Techniques</h3>
<p><b>L2 Weight Decay:</b> Adds (lambda/2) * Sum(W^2) to the loss. Penalizes large weights, encouraging simpler models.</p>
<p><b>Dropout:</b> Randomly zeros neurons during training (with probability p). Forces the network to learn redundant representations, reducing overfitting.</p>
<p><b>Batch Normalization:</b> Normalizes layer inputs to zero mean and unit variance. Reduces internal covariate shift, allows higher learning rates.</p>
<p>Visit the <b>Regularization</b> page to compare their effects on train vs. validation loss.</p>"""},
    {"title": "10. Practical Tips", "content": """
<h3>Practical Tips</h3>
<ul><li>Start simple: 1-2 hidden layers, ReLU + He init, Adam with lr=0.001</li>
<li>Always watch the loss curve</li>
<li>If loss is noisy: increase batch size or decrease lr</li>
<li>If loss plateaus: try more capacity or adjust lr</li>
<li>Use the <b>Network Zoo</b> to quickly compare architectures</li>
<li>Use the <b>Loss Landscape</b> to understand optimization difficulty</li></ul>
<p>After mastering these basics, explore CNNs for images, Transformers for sequences, and reinforcement learning.</p>"""},
    {"title": "11. Self-Attention", "content": """
<h3>Self-Attention Mechanism</h3>
<p>Self-attention lets every token in a sequence attend to every other token, computing a weighted combination of values based on query-key compatibility.</p>
<h4>The Three Matrices</h4>
<ul>
<li><b>Q (Query):</b> "What am I looking for?" Each token projects its embedding through Wq.</li>
<li><b>K (Key):</b> "What do I contain?" Each token projects through Wk.</li>
<li><b>V (Value):</b> "What information do I provide?" Each token projects through Wv.</li>
</ul>
<h4>Computation</h4>
<p style="color:#58a6ff; font-family:monospace; padding:8px; background:#161b22; border-radius:6px;">
  scores = Q * K^T / sqrt(d_k)<br>
  attention_weights = softmax(scores, axis=-1)<br>
  output = attention_weights * V
</p>
<p>The <b>1/sqrt(d_k)</b> scaling prevents dot products from growing too large, which would push softmax into regions with tiny gradients. This is the key insight of the <i>scaled</i> dot-product attention.</p>
<h4>Multi-Head Attention</h4>
<p>Instead of one attention function, run <b>H</b> parallel attention heads, each with its own Wq, Wk, Wv (projected into d_k = d_model / H dimensions). Concatenate outputs and project with Wo. Different heads learn different relationships: some attend to syntax, others to semantics.</p>
<p>Visit the <b>Attention</b> page to see interactive heatmaps of attention weights for different sentences and head counts.</p>"""},
    {"title": "12. Convolution Kernels", "content": """
<h3>Convolution Kernels (Filters)</h3>
<p>A convolution kernel is a small matrix that slides across an image, computing dot products at each position to produce a <b>feature map</b>. Each kernel detects a specific local pattern.</p>
<h4>Classic Kernels and What They Detect</h4>
<ul>
<li><b>Edge Detection (Sobel X/Y):</b> Responds strongly to vertical/horizontal intensity changes. The kernel values sum to zero, so uniform regions produce zero output.</li>
<li><b>Laplacian:</b> A second-order derivative detector that responds to edges in all directions. Sums to zero.</li>
<li><b>Gaussian Blur:</b> All-positive weights that average neighboring pixels, smoothing the image. The 5x5 version has larger receptive field and stronger smoothing.</li>
<li><b>Sharpen:</b> Enhances edges by subtracting a blurred version from the original. The center weight is large and positive, neighbors are negative.</li>
<li><b>Emboss:</b> Creates a 3D relief effect by emphasizing directional intensity differences.</li>
</ul>
<h4>Output Size Formula</h4>
<p style="color:#58a6ff; font-family:monospace; padding:8px; background:#161b22; border-radius:6px;">
  out_size = (W - K + 2*P) / S + 1<br>
  W=input size, K=kernel size, P=padding, S=stride
</p>
<h4>Key Properties</h4>
<p><b>Weight sharing:</b> The same kernel is applied at every position, drastically reducing parameters compared to dense layers. A 3x3 kernel on a 1000x1000 image has only 9 parameters but processes 1M pixels.</p>
<p><b>Translation equivariance:</b> If an object moves in the input, its response moves in the feature map. This is why CNNs are so effective for images.</p>
<p>Visit the <b>Kernels</b> page to apply different kernels to test images and design your own custom kernel.</p>"""},
    {"title": "13. Scaling Laws", "content": """
<h3>Neural Network Scaling Laws</h3>
<p>Scaling laws describe how model performance improves as you increase <b>model size</b> (parameters), <b>dataset size</b>, and <b>compute budget</b>. These relationships are remarkably predictable.</p>
<h4>The Power Law Relationship</h4>
<p style="color:#58a6ff; font-family:monospace; padding:8px; background:#161b22; border-radius:6px;">
  L(N) ~ (N_c / N)^alpha_N<br>
  L(D) ~ (D_c / D)^alpha_D<br>
  L(C) ~ (C_c / C)^alpha_C
</p>
<p>Where L is loss, N is parameters, D is data size, C is compute. The exponents are typically: alpha_N ~ 0.076, alpha_D ~ 0.095, alpha_C ~ 0.050 (Kaplan et al., 2020).</p>
<h4>Chinchilla Optimal Training</h4>
<p>The <b>Chinchilla laws</b> (Hoffmann et al., 2022) showed that most large models were significantly undertrained. For compute-optimal training, you should scale data and parameters roughly equally: a 10x bigger model needs ~10x more data.</p>
<h4>Implications</h4>
<ul>
<li><b>More parameters always help</b> — but with diminishing returns. The log-log plot shows a straight line, meaning exponential increases in N give linear decreases in log-loss.</li>
<li><b>Depth vs. Width:</b> Deeper networks are more parameter-efficient but harder to train (vanishing gradients). Width scales more reliably but uses more parameters for the same performance.</li>
<li><b>Data scaling:</b> More data reduces overfitting and improves generalization. But data quality matters as much as quantity.</li>
</ul>
<p>Visit the <b>Scaling Laws</b> page to run experiments comparing depth, width, and data scaling on real training tasks.</p>"""},
    {"title": "14. Transformers", "content": """<h3>The Transformer Architecture</h3>
<h4>What Are Transformers?</h4>
<p>Transformers are a neural network architecture introduced in <b>"Attention Is All You Need"</b> (Vaswani et al., 2017) that replaced recurrence with <b>self-attention</b> mechanisms. They have become the dominant architecture in NLP, vision, and beyond.</p>
<h4>The Encoder Block</h4>
<p>A single transformer encoder block consists of:</p>
<ul>
<li><b>Multi-Head Self-Attention:</b> The input sequence is projected into Q (Query), K (Key), V (Value) matrices. Attention scores are computed as <code>softmax(QK^T / sqrt(d_k)) V</code>, allowing each token to attend to all other tokens. Multiple heads capture different relationship types simultaneously.</li>
<li><b>Position-wise Feed-Forward Network:</b> A two-layer MLP (usually with GELU activation) applied independently to each position: <code>FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2</code>.</li>
<li><b>Layer Normalization:</b> Applied before (pre-norm) or after (post-norm) each sub-layer to stabilize training.</li>
<li><b>Residual Connections:</b> Each sub-layer output is added back to its input: <code>x = x + SubLayer(x)</code>, enabling gradient flow through deep stacks.</li>
</ul>
<h4>Positional Encoding</h4>
<p>Since transformers have no recurrence, they need explicit position information. The original paper uses sinusoidal encodings: <code>PE(pos, 2i) = sin(pos / 10000^(2i/d))</code> and <code>PE(pos, 2i+1) = cos(pos / 10000^(2i/d))</code>. Modern variants often use learned positional embeddings.</p>
<h4>Why Transformers Win</h4>
<ul>
<li><b>Parallelism:</b> Unlike RNNs, every position is computed simultaneously — enabling massive GPU speedup.</li>
<li><b>Long-range dependencies:</b> Every token directly attends to every other token with O(1) path length, solving the vanishing gradient problem for sequences.</li>
<li><b>Scalability:</b> Transformers scale predictably with data and compute, following clear scaling laws.</li>
</ul>
<h4>Key Hyperparameters</h4>
<ul>
<li><b>d_model:</b> Embedding dimension (e.g., 512, 768, 1024). Larger = more expressive but slower.</li>
<li><b>n_heads:</b> Number of attention heads (e.g., 8, 12). d_model must be divisible by n_heads; each head has dimension d_model/n_heads.</li>
<li><b>d_ff:</b> Feed-forward hidden dimension (typically 4 × d_model).</li>
<li><b>n_layers:</b> Number of encoder/decoder blocks (e.g., 6, 12, 24, 96 for GPT-3).</li>
<li><b>dropout:</b> Applied to attention weights and feed-forward output for regularization.</li>
</ul>
<p>Visit the <b>Attention</b> page to see self-attention computed step-by-step with visual attention weight matrices.</p>"""},
    {"title": "15. GANs", "content": """<h3>Generative Adversarial Networks</h3>
<h4>The Minimax Game</h4>
<p>GANs (Goodfellow et al., 2014) consist of two networks locked in a competitive game:</p>
<ul>
<li><b>Generator (G):</b> Takes random noise z ~ N(0, I) and produces fake samples G(z) that mimic the real data distribution.</li>
<li><b>Discriminator (D):</b> A binary classifier that tries to distinguish real samples from fake ones. Output D(x) ∈ [0, 1] represents probability that x is real.</li>
</ul>
<p>The training objective is a <b>minimax game</b>: G minimizes while D maximizes: <code>min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]</code>. At equilibrium, G produces perfect samples and D outputs 0.5 for everything.</p>
<h4>Training Challenges</h4>
<ul>
<li><b>Mode Collapse:</b> The generator may learn to produce only a few types of outputs, ignoring the full diversity of the real data. For example, on MNIST, G might only generate 1s and 3s.</li>
<li><b>Training Instability:</b> If D becomes too strong, G gets no useful gradients (vanishing gradients). If G fools D too easily, it stops improving. Balancing G and D is notoriously difficult.</li>
<li><b>Non-convergence:</b> Unlike supervised learning, GAN training has no convergence guarantee. Losses oscillate rather than monotonically decreasing.</li>
<li><b>Hyperparameter sensitivity:</b> Learning rates, architecture balance, and training tricks dramatically affect outcomes.</li>
</ul>
<h4>GAN Variants</h4>
<ul>
<li><b>WGAN (Wasserstein GAN):</b> Replaces the JS divergence with the Wasserstein (Earth Mover) distance. The discriminator becomes a "critic" that outputs real-valued scores instead of probabilities. This provides smoother gradients and more stable training. Requires weight clipping or gradient penalty.</li>
<li><b>Conditional GAN (cGAN):</b> Both G and D receive class labels as additional input, enabling controlled generation (e.g., "generate a 7").</li>
<li><b>StyleGAN:</b> Uses a mapping network to transform z into an intermediate style vector w, then injects style at multiple layers via adaptive instance normalization (AdaIN). Produces high-resolution, photorealistic images with controllable attributes.</li>
<li><b>DCGAN:</b> Uses transposed convolutions in G and strided convolutions in D, enabling image generation without fully connected layers.</li>
</ul>
<p>Visit the <b>GAN</b> page to train a GAN interactively and observe mode collapse and training dynamics.</p>"""},
    {"title": "16. Diffusion Models", "content": """<h3>Denoising Diffusion Probabilistic Models (DDPM)</h3>
<h4>What Are Diffusion Models?</h4>
<p>Diffusion models (Ho et al., 2020) are generative models that learn to reverse a gradual noising process. They have become the state-of-the-art for image, audio, and video generation, powering models like DALL-E, Stable Diffusion, and Midjourney.</p>
<h4>The Forward Process</h4>
<p>Starting from a real data point x_0, the forward process adds Gaussian noise over T timesteps:</p>
<ul>
<li>At each step: <code>x_t = sqrt(1 - β_t) * x_{t-1} + sqrt(β_t) * ε</code>, where ε ~ N(0, I).</li>
<li>This can be done in one step: <code>x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε</code>, where ᾱ_t = Π(1 - β_i).</li>
<li>After T steps (typically T=1000), x_T is nearly pure Gaussian noise — the data is completely destroyed.</li>
<li>The noise schedule β_t is usually linearly or cosinely increasing from a small value to near 1.0.</li>
</ul>
<h4>The Reverse Process</h4>
<p>A neural network ε_θ(x_t, t) learns to predict the noise that was added at each timestep:</p>
<ul>
<li>Training objective (simplified): <code>L = E[||ε - ε_θ(x_t, t)||²]</code> — just predict the noise!</li>
<li>Sampling starts from x_T ~ N(0, I) and iteratively denoises: <code>x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-ᾱ_t)) * ε_θ(x_t, t)) + σ_t * z</code>.</li>
<li>The network is typically a U-Net with time embedding injected via adaptive layer normalization.</li>
</ul>
<h4>Why Diffusion Beats GANs</h4>
<ul>
<li><b>Training stability:</b> Diffusion models have a fixed, stable training objective (simple MSE). No adversarial game, no mode collapse, no delicate balancing act.</li>
<li><b>Diversity:</b> Diffusion models naturally cover the full data distribution. Every mode is reached through the iterative denoising process.</li>
<li><b>Quality:</b> Modern diffusion models produce higher-fidelity images than GANs, especially at high resolutions.</li>
<li><b>Trade-off:</b> Sampling is slow (hundreds of network evaluations) compared to GANs (single forward pass). Techniques like DDIM and classifier-free guidance speed this up.</li>
</ul>
<p>Visit the <b>Diffusion</b> page to watch the forward and reverse diffusion process in real time.</p>"""},
    {"title": "17. Normalization Methods", "content": """<h3>Comparing Normalization Techniques</h3>
<h4>Why Normalize?</h4>
<p>Normalization layers stabilize and accelerate deep network training by keeping activation magnitudes in a reasonable range. Without normalization, activations can grow or shrink exponentially through layers, causing exploding/vanishing gradients.</p>
<h4>Batch Normalization (BatchNorm)</h4>
<ul>
<li><b>Computation:</b> For each feature, subtract the <b>mean</b> and divide by the <b>standard deviation</b> computed across the <b>batch dimension</b>. Then apply learned scale γ and shift β: <code>y = γ * (x - μ_batch) / σ_batch + β</code>.</li>
<li><b>At inference:</b> Uses running estimates of μ and σ accumulated during training (exponential moving average).</li>
<li><b>Pros:</b> Very effective for CNNs; acts as a mild regularizer (noise from batch statistics); enables higher learning rates.</li>
<li><b>Cons:</b> Batch-dependent — behaves differently with different batch sizes. Fails or degrades with very small batches. Not well-suited for RNNs or variable-length sequences. Adds synchronization overhead in distributed training.</li>
</ul>
<h4>Layer Normalization (LayerNorm)</h4>
<ul>
<li><b>Computation:</b> Normalizes across the <b>feature dimension</b> for each sample independently: <code>y = γ * (x - μ_features) / σ_features + β</code>.</li>
<li><b>Pros:</b> Independent of batch size — works with batch_size=1. Well-suited for RNNs, Transformers, and sequence models. No running statistics needed — identical behavior at train and test time.</li>
<li><b>Cons:</b> Less effective for CNNs where spatial statistics matter. Doesn't provide the same regularization benefit as BatchNorm.</li>
</ul>
<h4>Group Normalization (GroupNorm)</h4>
<ul>
<li><b>Computation:</b> Divides channels into <b>groups</b> and normalizes within each group. <code>GroupNorm(32 groups, 64 channels)</code> computes 32 separate normalizations over 2 channels each.</li>
<li><b>Pros:</b> Batch-size independent like LayerNorm, but preserves some spatial/statistical structure like BatchNorm. Default in many vision transformers (ViT, DETR). Works well for detection and segmentation tasks with small batches.</li>
<li><b>Cons:</b> Performance depends on the number of groups. More groups ≈ closer to LayerNorm; fewer groups ≈ closer to BatchNorm. Requires tuning the group count.</li>
</ul>
<h4>When to Use What</h4>
<ul>
<li><b>CNNs with decent batch size (≥16):</b> BatchNorm is still the default and works best.</li>
<li><b>Transformers / NLP:</b> LayerNorm (or RMSNorm) is standard.</li>
<li><b>CNNs with small batches or detection/segmentation:</b> GroupNorm is the go-to choice.</li>
<li><b>Online learning / batch_size=1:</b> LayerNorm or GroupNorm (BatchNorm won't work).</li>
</ul>
<p>Visit the <b>Normalization</b> page to compare all three methods interactively with gradient flow visualization.</p>"""},
]

class LessonsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        sidebar = QWidget(); sidebar.setFixedWidth(250); sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.addWidget(QLabel("Lessons"))
        self.lesson_list = QVBoxLayout()
        for i, lesson in enumerate(LESSONS):
            btn = QPushButton(lesson["title"]); btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton { text-align: left; padding: 10px 14px; font-size: 13px; }
                QPushButton:checked { background-color: #1f6feb; border-color: #388bfd; }
            """)
            btn.clicked.connect(lambda checked, idx=i: self._show_lesson(idx))
            self.lesson_list.addWidget(btn)
        sidebar_layout.addLayout(self.lesson_list); sidebar_layout.addStretch()
        layout.addWidget(sidebar)
        self.content = QTextEdit(); self.content.setReadOnly(True)
        self.content.setStyleSheet("""
            QTextEdit { background-color: #161b22; border: none; padding: 20px; font-size: 14px; line-height: 1.6; }
            h3 { color: #58a6ff; margin-top: 16px; } h4 { color: #e1e4e8; margin-top: 12px; }
            p  { color: #c9d1d9; margin: 8px 0; } ul, ol { color: #c9d1d9; margin: 8px 0; padding-left: 24px; }
            li { margin: 4px 0; }
        """)
        layout.addWidget(self.content, stretch=1)
        if LESSONS:
            self.lesson_buttons = sidebar.findChildren(QPushButton)
            if self.lesson_buttons: self.lesson_buttons[0].setChecked(True)
            self._show_lesson(0)

    def _show_lesson(self, idx):
        btns = self.findChildren(QPushButton)
        for btn in btns:
            if btn.isCheckable(): btn.setChecked(False)
        lesson_btns = []
        for i in range(self.lesson_list.count()):
            item = self.lesson_list.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QPushButton):
                lesson_btns.append(item.widget())
        if idx < len(lesson_btns): lesson_btns[idx].setChecked(True)
        self.content.setHtml(LESSONS[idx]["content"])


# ════════════════════════════════════════════════════════════════
#  PAGE 13: ATTENTION EXPLORER
# ════════════════════════════════════════════════════════════════

class AttentionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._compute()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Sentence:"))
        self.sentence_combo = QComboBox()
        self.sentence_combo.addItems(list(ATTENTION_PRESETS.keys()))
        self.sentence_combo.currentTextChanged.connect(self._compute)
        top_bar.addWidget(self.sentence_combo, stretch=1)

        top_bar.addWidget(QLabel("Heads:"))
        self.heads_combo = QComboBox()
        self.heads_combo.addItems(["1 (Single Head)", "2 Heads", "4 Heads", "8 Heads"])
        self.heads_combo.currentIndexChanged.connect(self._compute)
        top_bar.addWidget(self.heads_combo)

        self.scale_check = QCheckBox("Show Q/K/V Detail")
        self.scale_check.setChecked(True)
        self.scale_check.stateChanged.connect(self._compute)
        top_bar.addWidget(self.scale_check)

        layout.addLayout(top_bar)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(80)
        self.info_text.setHtml("<i style='color:#8b949e'>Select a sentence and number of heads to visualize attention.</i>")
        layout.addWidget(self.info_text)

        self.canvas = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.canvas, stretch=1)

    def _compute(self):
        sentence = self.sentence_combo.currentText()
        preset = ATTENTION_PRESETS[sentence]
        tokens = preset["tokens"]
        x = preset["embeddings"]
        T, d_model = x.shape

        n_heads_idx = self.heads_combo.currentIndex()
        n_heads = [1, 2, 4, 8][n_heads_idx]
        show_detail = self.scale_check.isChecked()

        if n_heads == 1:
            result = compute_self_attention(x, d_k=d_model)
            self._plot_single(tokens, result, show_detail, d_model)
        else:
            if d_model % n_heads != 0:
                new_d = ((d_model // n_heads) + 1) * n_heads
                x_pad = np.zeros((T, new_d))
                x_pad[:, :d_model] = x
                result = compute_multi_head_attention(x_pad, n_heads)
            else:
                result = compute_multi_head_attention(x, n_heads)
            self._plot_multi(tokens, result, show_detail, n_heads)

        html = f"""<p style='color:#e1e4e8'>
        <b>Sequence:</b> &quot;<i>{sentence}</i>&quot; ({T} tokens, d_model={d_model})<br>
        <b>Heads:</b> {n_heads} | <b>d_k per head:</b> {result.get('d_k', d_model)}<br>
        <b>Formula:</b> <span style='color:#58a6ff; font-family:monospace'>
        Attention(Q,K,V) = softmax(QK<sup>T</sup> / sqrt(d_k)) * V</span></p>"""
        self.info_text.setHtml(html)

    def _plot_single(self, tokens, result, show_detail, d_model):
        self.canvas.fig.clear()
        if show_detail:
            axes = self.canvas.fig.subplots(2, 2)
            self.canvas.style_axes(axes[0, 0], "Attention Weights (Heatmap)", "", "")
            self._draw_heatmap(axes[0, 0], result["attn_weights"], tokens)

            self.canvas.style_axes(axes[0, 1], "Raw Scores (before softmax)", "", "")
            self._draw_heatmap(axes[0, 1], result["scores"], tokens, cmap="RdBu_r")

            self.canvas.style_axes(axes[1, 0], "Q vectors (each token)", "Token", "Dimension")
            im = axes[1, 0].imshow(result["Q"], aspect="auto", cmap="coolwarm")
            axes[1, 0].set_yticks(range(len(tokens)))
            axes[1, 0].set_yticklabels(tokens, fontsize=9)
            self.canvas.fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

            self.canvas.style_axes(axes[1, 1], "V vectors (each token)", "Token", "Dimension")
            im = axes[1, 1].imshow(result["V"], aspect="auto", cmap="PiYG")
            axes[1, 1].set_yticks(range(len(tokens)))
            axes[1, 1].set_yticklabels(tokens, fontsize=9)
            self.canvas.fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
        else:
            axes = self.canvas.fig.subplots(1, 2)
            self.canvas.style_axes(axes[0], "Attention Weights", "", "")
            self._draw_heatmap(axes[0], result["attn_weights"], tokens)
            self.canvas.style_axes(axes[1], "Output Vectors", "Token", "Dimension")
            im = axes[1].imshow(result["output"], aspect="auto", cmap="viridis")
            axes[1].set_yticks(range(len(tokens)))
            axes[1].set_yticklabels(tokens, fontsize=9)
            self.canvas.fig.colorbar(im, ax=axes[1], shrink=0.8)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _plot_multi(self, tokens, result, show_detail, n_heads):
        heads = result["heads"]
        n_h = len(heads)

        if show_detail:
            nrows = 2
            ncols = n_h
        else:
            nrows = 1
            ncols = n_h

        self.canvas.fig.clear()
        axes = self.canvas.fig.subplots(nrows, ncols)

        if ncols == 1 and nrows > 1:
            pass
        elif nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)

        for h in range(n_h):
            ax = axes[0, h] if show_detail else axes[0, h]
            self.canvas.style_axes(ax, f"Head {h+1} Attention", "", "")
            self._draw_heatmap(ax, heads[h]["attn_weights"], tokens, vmin=0, vmax=1)

            if show_detail:
                ax2 = axes[1, h]
                self.canvas.style_axes(ax2, f"Head {h+1} Scores", "", "")
                self._draw_heatmap(ax2, heads[h]["scores"], tokens, cmap="RdBu_r")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _draw_heatmap(self, ax, matrix, tokens, cmap="YlOrRd", vmin=None, vmax=None):
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if abs(val) > (matrix.max() - matrix.min()) * 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
        self.canvas.fig.colorbar(im, ax=ax, shrink=0.8)


# ════════════════════════════════════════════════════════════════
#  PAGE 14: KERNEL EXPLORER
# ════════════════════════════════════════════════════════════════

class KernelPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.custom_kernel = np.zeros((3, 3))
        self._setup_ui()
        self._apply_kernel()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()

        top_bar.addWidget(QLabel("Image:"))
        self.image_combo = QComboBox()
        self.image_combo.addItems(["checkerboard", "circles", "gradient", "edges_h", "diagonal", "noise"])
        self.image_combo.currentTextChanged.connect(self._apply_kernel)
        top_bar.addWidget(self.image_combo)

        top_bar.addWidget(QLabel("Kernel:"))
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(list(KERNEL_PRESETS.keys()))
        self.kernel_combo.currentTextChanged.connect(self._apply_kernel)
        top_bar.addWidget(self.kernel_combo, stretch=1)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(16, 64)
        self.size_spin.setValue(32)
        self.size_spin.setSingleStep(8)
        self.size_spin.valueChanged.connect(self._apply_kernel)
        top_bar.addWidget(QLabel("Size:"))
        top_bar.addWidget(self.size_spin)

        layout.addLayout(top_bar)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(60)
        self.info_text.setHtml("<i style='color:#8b949e'>A convolution kernel slides over the image, computing dot products to detect local patterns.</i>")
        layout.addWidget(self.info_text)

        self.canvas = MplCanvas(self, width=12, height=4)
        layout.addWidget(self.canvas, stretch=1)

        kernel_bar = QHBoxLayout()
        kernel_bar.addWidget(QLabel("Custom 3x3 Kernel:"))
        self.kernel_edits = []
        kernel_grid = QGridLayout()
        kernel_grid.setSpacing(2)
        for i in range(3):
            for j in range(3):
                spin = QDoubleSpinBox()
                spin.setRange(-5, 5)
                spin.setValue(0)
                spin.setSingleStep(0.5)
                spin.setDecimals(2)
                spin.setFixedWidth(60)
                spin.valueChanged.connect(self._on_custom_kernel_changed)
                kernel_grid.addWidget(spin, i, j)
                self.kernel_edits.append(spin)
        kernel_bar.addLayout(kernel_grid)
        apply_btn = QPushButton("Apply Custom")
        apply_btn.setObjectName("warning")
        apply_btn.clicked.connect(self._apply_custom_kernel)
        kernel_bar.addWidget(apply_btn)
        kernel_bar.addStretch()
        layout.addLayout(kernel_bar)

    def _on_custom_kernel_changed(self):
        for i in range(3):
            for j in range(3):
                self.custom_kernel[i, j] = self.kernel_edits[i * 3 + j].value()

    def _apply_custom_kernel(self):
        self._on_custom_kernel_changed()
        KERNEL_PRESETS["Custom (zeros)"] = self.custom_kernel.copy()
        self.kernel_combo.setCurrentText("Custom (zeros)")
        self._apply_kernel()

    def _apply_kernel(self):
        img_type = self.image_combo.currentText()
        kernel_name = self.kernel_combo.currentText()
        size = self.size_spin.value()

        image = generate_test_image(img_type, size)
        kernel = KERNEL_PRESETS[kernel_name]

        feature_map = conv2d(image, kernel, stride=1, padding=0)

        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(15, 4.5)
        axes = self.canvas.fig.subplots(1, 4)

        self.canvas.style_axes(axes[0], "Input Image", "", "")
        axes[0].imshow(image, cmap="gray", vmin=-1, vmax=1)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        self.canvas.style_axes(axes[1], f"Kernel ({kernel_name})", "", "")
        kh, kw = kernel.shape
        k_norm = kernel / max(np.abs(kernel).max(), 1e-8)
        axes[1].imshow(k_norm, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(kh):
            for j in range(kw):
                val = kernel[i, j]
                color = "white" if abs(val) > np.abs(kernel).max() * 0.5 else "black"
                axes[1].text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")
        axes[1].set_xticks([]); axes[1].set_yticks([])

        self.canvas.style_axes(axes[2], "Feature Map (output)", "", "")
        vmax = max(np.abs(feature_map).max(), 1e-8)
        axes[2].imshow(feature_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[2].set_xticks([]); axes[2].set_yticks([])

        self.canvas.style_axes(axes[3], "|Feature Map| (magnitude)", "", "")
        axes[3].imshow(np.abs(feature_map), cmap="hot", vmin=0, vmax=vmax)
        axes[3].set_xticks([]); axes[3].set_yticks([])

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        n_params_kernel = kernel.size
        out_h, out_w = feature_map.shape
        html = f"""<p style='color:#e1e4e8'>
        <b>Image:</b> {size}x{size} | <b>Kernel:</b> {kh}x{kw} ({n_params_kernel} params) |
        <b>Output:</b> {out_h}x{out_w}<br>
        <b>Output size formula:</b> <span style='color:#58a6ff; font-family:monospace'>
        (W - K + 2P) / S + 1 = ({size} - {kh} + 0) / 1 + 1 = {out_h}</span>
        | Feature map range: [{feature_map.min():.2f}, {feature_map.max():.2f}]</p>"""
        self.info_text.setHtml(html)


# ════════════════════════════════════════════════════════════════
#  PAGE 15: SCALING LAWS EXPLORER
# ════════════════════════════════════════════════════════════════

class ScalingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()

        top_bar.addWidget(QLabel("Dataset:"))
        self.ds_combo = QComboBox()
        self.ds_combo.addItems(list(DATASETS.keys()))
        top_bar.addWidget(self.ds_combo)

        top_bar.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Depth", "Width", "Data"])
        self.scale_combo.setCurrentIndex(1)
        top_bar.addWidget(self.scale_combo)

        top_bar.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(50, 1000)
        self.epochs_spin.setValue(200)
        self.epochs_spin.setSingleStep(50)
        top_bar.addWidget(self.epochs_spin)

        self.run_btn = QPushButton("Run Scaling Experiment")
        self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._run)
        top_bar.addWidget(self.run_btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(70)
        self.info_text.setHtml("""<p style='color:#e1e4e8'>Scaling laws reveal how model performance improves with more <b>parameters</b>,
        <b>data</b>, and <b>compute</b>. The relationship often follows a <span style='color:#58a6ff'>power law</span>.</p>""")
        layout.addWidget(self.info_text)

        self.canvas = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.canvas, stretch=1)

    def _run(self):
        self.info_text.setText("<p style='color:#f0883e'>Running experiment... please wait.</p>")
        QApplication.processEvents()

        dataset_name = self.ds_combo.currentText()
        scale_dim = self.scale_combo.currentText().lower()
        epochs = self.epochs_spin.value()

        if scale_dim == "depth":
            values = [1, 2, 3, 4, 5, 6]
        elif scale_dim == "width":
            values = [4, 8, 16, 32, 64, 128]
        else:
            values = [40, 80, 160, 300, 600]

        ds_fn = DATASETS[dataset_name]
        results = run_scaling_experiment(ds_fn, scale_dim=scale_dim, values=values, epochs=epochs)

        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(14, 5)
        axes = self.canvas.fig.subplots(1, 3)

        labels = [r[0] for r in results]
        n_params = [r[1] for r in results]
        final_accs = [r[3] * 100 for r in results]
        colors = ["#f85149", "#58a6ff", "#238636", "#f0883e", "#bc8cff", "#3fb950", "#79c0ff", "#d2a8ff"]

        self.canvas.style_axes(axes[0], "Training Loss", "Epoch", "Loss")
        for i, (label, _, loss_hist, _) in enumerate(results):
            axes[0].plot(loss_hist, color=colors[i % len(colors)], linewidth=1.2, alpha=0.8, label=label)
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[0].grid(True, alpha=0.15, color="#8b949e")

        self.canvas.style_axes(axes[1], "Final Accuracy vs Configuration", "Configuration", "Accuracy (%)")
        x_pos = range(len(results))
        bars = axes[1].bar(x_pos, final_accs, color=[colors[i % len(colors)] for i in range(len(results))], alpha=0.85)
        axes[1].set_xticks(list(x_pos))
        axes[1].set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
        for bar, acc in zip(bars, final_accs):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{acc:.0f}%", ha="center", va="bottom", color="#e1e4e8", fontsize=9, fontweight="bold")
        axes[1].grid(True, alpha=0.15, color="#8b949e", axis="y")

        self.canvas.style_axes(axes[2], "Scaling Law (log-log)", "Parameters (log scale)", "Loss (log scale)")
        final_losses = [r[2][-1] for r in results]
        valid = [(n, l) for n, l in zip(n_params, final_losses) if n > 0 and l > 0]
        if len(valid) > 2:
            ns = np.array([v[0] for v in valid])
            ls = np.array([v[1] for v in valid])
            log_n = np.log(ns)
            log_l = np.log(ls)
            coeffs = np.polyfit(log_n, log_l, 1)
            a, b = coeffs
            fit_n = np.linspace(ns.min(), ns.max(), 100)
            fit_l = np.exp(a * np.log(fit_n) + b)
            axes[2].loglog(ns, ls, 'o-', color="#58a6ff", markersize=8, label="Observed", linewidth=1.5)
            axes[2].loglog(fit_n, fit_l, '--', color="#f85149", linewidth=2, label=f"Fit: L ~ N^{a:.2f}")
            axes[2].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=9)
        axes[2].grid(True, alpha=0.15, color="#8b949e", which="both")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        best_idx = int(np.argmax(final_accs))
        html = f"""<p style='color:#e1e4e8'>
        <b>Experiment:</b> Scaling {scale_dim} on {dataset_name} | <b>Epochs:</b> {epochs}<br>
        <b>Best config:</b> {labels[best_idx]} with {final_accs[best_idx]:.1f}% accuracy ({n_params[best_idx]} params)<br>
        <b>Scaling law:</b> <span style='color:#58a6ff; font-family:monospace'>
        L(N) ~ N<sup>a</sup></span> where a is the scaling exponent (steeper = more benefit from scaling).
        Typically a ~ -0.07 to -0.5 for large models.</p>"""
        self.info_text.setHtml(html)




# ════════════════════════════════════════════════════════════════
#  PAGE 16: ADABOOST EXPLORER
# ════════════════════════════════════════════════════════════════

class AdaBoostPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.X = None
        self.y = None
        self.boost = None
        self.round_info = None
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()

        top_bar.addWidget(QLabel("Dataset:"))
        self.ds_combo = QComboBox()
        self.ds_combo.addItems(["XOR", "Circle", "Moon", "Gaussian", "Spiral"])
        self.ds_combo.currentTextChanged.connect(self._load_data)
        top_bar.addWidget(self.ds_combo)

        top_bar.addWidget(QLabel("Rounds:"))
        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(1, 50)
        self.rounds_spin.setValue(10)
        top_bar.addWidget(self.rounds_spin)

        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setRange(0, 10)
        self.step_slider.setValue(10)
        self.step_slider.setFixedWidth(200)
        self.step_slider.valueChanged.connect(self._update_viz)
        top_bar.addWidget(QLabel("Step:"))
        top_bar.addWidget(self.step_slider)
        self.step_label = QLabel("10/10")
        top_bar.addWidget(self.step_label)

        self.run_btn = QPushButton("Run AdaBoost")
        self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._run)
        top_bar.addWidget(self.run_btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(70)
        self.info_text.setHtml("<i style='color:#8b949e'>AdaBoost combines weak learners (decision stumps) into a strong ensemble by focusing on misclassified samples.</i>")
        layout.addWidget(self.info_text)

        self.canvas = MplCanvas(self, width=14, height=5)
        layout.addWidget(self.canvas, stretch=1)

    def _load_data(self):
        ds_name = self.ds_combo.currentText()
        self.X, self.y_01 = DATASETS[ds_name]()
        self.y = 2.0 * self.y_01.flatten() - 1.0  # Convert to {-1, +1}
        self.round_info = None
        self._plot_data_only()

    def _plot_data_only(self):
        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(15, 5)
        axes = self.canvas.fig.subplots(1, 3)
        self.canvas.style_axes(axes[0], "Data (with sample weights)", "x1", "x2")
        c0, c1 = self.y == -1, self.y == 1
        axes[0].scatter(self.X[c0, 0], self.X[c0, 1], c="#58a6ff", s=20, alpha=0.7, label="Class -1")
        axes[0].scatter(self.X[c1, 0], self.X[c1, 1], c="#f85149", s=20, alpha=0.7, label="Class +1")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        self.canvas.style_axes(axes[1], "Ensemble Error", "Round", "Error")
        self.canvas.style_axes(axes[2], "Learner Weights (alpha)", "Round", "Alpha")
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _run(self):
        self.info_text.setText("<p style='color:#f0883e'>Training AdaBoost...</p>")
        QApplication.processEvents()

        n_rounds = self.rounds_spin.value()
        self.boost = AdaBoostClassifier(n_estimators=n_rounds)
        self.round_info = self.boost.fit(self.X, self.y)

        self.step_slider.setMaximum(n_rounds)
        self.step_slider.setValue(n_rounds)
        self._update_viz(n_rounds)

    def _update_viz(self, step=None):
        if self.round_info is None:
            return
        if step is None:
            step = self.step_slider.value()

        self.step_label.setText(f"{step}/{self.rounds_spin.value()}")
        n_rounds = len(self.round_info)
        use_rounds = min(step, n_rounds)

        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(15, 5)
        axes = self.canvas.fig.subplots(1, 3)

        # Plot 1: Data with sample weights
        self.canvas.style_axes(axes[0], f"Data (after round {use_rounds})", "x1", "x2")
        c0, c1 = self.y == -1, self.y == 1

        if use_rounds > 0:
            weights = self.round_info[use_rounds - 1]["sample_weights"]
            w_norm = weights / weights.max() * 80 + 5
            sizes_0 = w_norm[c0]
            sizes_1 = w_norm[c1]
        else:
            sizes_0 = np.full(c0.sum(), 20)
            sizes_1 = np.full(c1.sum(), 20)

        axes[0].scatter(self.X[c0, 0], self.X[c0, 1], c="#58a6ff", s=sizes_0, alpha=0.7, label="Class -1")
        axes[0].scatter(self.X[c1, 0], self.X[c1, 1], c="#f85149", s=sizes_1, alpha=0.7, label="Class +1")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)

        # Draw decision boundaries for individual stumps (up to 4)
        if use_rounds > 0:
            for r_idx in range(min(use_rounds, 4)):
                ri = self.round_info[r_idx]
                feat = ri["feature"]
                thresh = ri["threshold"]
                pol = ri["polarity"]
                alpha = ri["alpha"]
                if feat == 0:
                    if pol == 1:
                        axes[0].axvline(thresh, color="#f0883e", alpha=min(alpha / 2, 0.8), 
                                       linestyle="--", linewidth=1.5)
                    else:
                        axes[0].axvline(thresh, color="#bc8cff", alpha=min(alpha / 2, 0.8),
                                       linestyle="--", linewidth=1.5)
                else:
                    if pol == 1:
                        axes[0].axhline(thresh, color="#f0883e", alpha=min(alpha / 2, 0.8),
                                       linestyle="--", linewidth=1.5)
                    else:
                        axes[0].axhline(thresh, color="#bc8cff", alpha=min(alpha / 2, 0.8),
                                       linestyle="--", linewidth=1.5)

        # Plot 2: Ensemble error over rounds
        self.canvas.style_axes(axes[1], "Ensemble Error", "Round", "Error")
        if use_rounds > 0:
            errors = [self.round_info[i]["ensemble_error"] for i in range(use_rounds)]
            rounds_x = list(range(1, use_rounds + 1))
            axes[1].plot(rounds_x, errors, "o-", color="#58a6ff", markersize=4, linewidth=1.5)
            axes[1].axhline(0, color="#30363d", linestyle=":")
            # Also show individual stump errors
            stump_errs = [self.round_info[i]["weighted_error"] for i in range(use_rounds)]
            axes[1].plot(rounds_x, stump_errs, "s--", color="#f85149", markersize=3, alpha=0.6, label="Stump error")
            axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[1].set_xlim(0.5, max(use_rounds + 0.5, 2))
        axes[1].grid(True, alpha=0.15, color="#8b949e")

        # Plot 3: Alpha (learner weights) as bar chart
        self.canvas.style_axes(axes[2], "Learner Weights (alpha)", "Round", "Alpha")
        if use_rounds > 0:
            alphas = [self.round_info[i]["alpha"] for i in range(use_rounds)]
            colors_bar = ["#238636" if a > 0 else "#f85149" for a in alphas]
            axes[2].bar(range(1, use_rounds + 1), alphas, color=colors_bar, alpha=0.85)
            axes[2].axhline(0, color="#30363d", linestyle=":")
        axes[2].grid(True, alpha=0.15, color="#8b949e", axis="y")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        # Update info
        if use_rounds > 0:
            ri = self.round_info[use_rounds - 1]
            ensemble_acc = (1.0 - ri["ensemble_error"]) * 100
            html = f"""<p style='color:#e1e4e8'>
            <b>Round {use_rounds}/{n_rounds}</b> |
            <b>Stump {use_rounds}:</b> feature={ri["feature"]}, threshold={ri["threshold"]:.3f}, polarity={ri["polarity"]:+d}, alpha={ri["alpha"]:.3f} |
            <b>Ensemble Accuracy:</b> <span style='color:#3fb950'>{ensemble_acc:.1f}%</span><br>
            <b>Formula:</b> <span style='color:#58a6ff; font-family:monospace'>
            alpha_t = 0.5 * ln((1 - error_t) / error_t)</span> |
            <b>Key idea:</b> Misclassified samples get <b>higher weights</b> in the next round.</p>"""
            self.info_text.setHtml(html)


# ════════════════════════════════════════════════════════════════
#  PAGE 17: VANISHING / EXPLODING GRADIENT DEEP DIVE
# ════════════════════════════════════════════════════════════════

class VanishingGradPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._compute()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()

        top_bar.addWidget(QLabel("Activation:"))
        self.act_combo = QComboBox()
        self.act_combo.addItems(["Sigmoid", "Tanh", "ReLU", "LeakyReLU", "GELU", "Swish"])
        self.act_combo.currentTextChanged.connect(self._compute)
        top_bar.addWidget(self.act_combo)

        top_bar.addWidget(QLabel("Depth:"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(5, 100)
        self.depth_spin.setValue(30)
        self.depth_spin.setSingleStep(5)
        self.depth_spin.valueChanged.connect(self._compute)
        top_bar.addWidget(self.depth_spin)

        self.weight_check = QCheckBox("With Weight Matrices")
        self.weight_check.setChecked(True)
        self.weight_check.stateChanged.connect(self._compute)
        top_bar.addWidget(self.weight_check)

        top_bar.addWidget(QLabel("Init:"))
        self.init_combo = QComboBox()
        self.init_combo.addItems(["he", "xavier", "random(0.01)"])
        self.init_combo.currentTextChanged.connect(self._compute)
        top_bar.addWidget(self.init_combo)

        top_bar.addWidget(QLabel("Fan-in:"))
        self.fanin_spin = QSpinBox()
        self.fanin_spin.setRange(4, 512)
        self.fanin_spin.setValue(64)
        self.fanin_spin.setSingleStep(16)
        self.fanin_spin.valueChanged.connect(self._compute)
        top_bar.addWidget(self.fanin_spin)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(80)
        self.info_text.setHtml("<i style='color:#8b949e'>Vanishing/exploding gradients are one of the fundamental challenges in training deep networks.</i>")
        layout.addWidget(self.info_text)

        self.canvas = MplCanvas(self, width=14, height=5)
        layout.addWidget(self.canvas, stretch=1)

    def _compute(self):
        act = self.act_combo.currentText()
        depth = self.depth_spin.value()
        use_weights = self.weight_check.isChecked()
        init = self.init_combo.currentText()
        fan_in = self.fanin_spin.value()

        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(16, 5.5)

        if use_weights:
            results = analyze_gradient_flow_with_weights(act, depth, fan_in, init)
            self._plot_with_weights(results)
        else:
            results = analyze_gradient_flow(act, depth)
            self._plot_activation_chain(results)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _plot_activation_chain(self, results):
        axes = self.canvas.fig.subplots(1, 3)
        grads = results["gradient_magnitudes"]
        acts = results["activations"]
        n = results["n_layers"]

        # Gradient flow
        self.canvas.style_axes(axes[0], "Gradient Magnitude (pure activations)", "Layer", "|Gradient|")
        axes[0].semilogy(range(n + 1), grads, "o-", color="#58a6ff", markersize=3, linewidth=1.5)
        axes[0].axhline(1e-3, color="#f85149", linestyle="--", alpha=0.6, label="1e-3 (vanishing)")
        axes[0].axhline(1e3, color="#f0883e", linestyle="--", alpha=0.6, label="1e3 (exploding)")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[0].grid(True, alpha=0.15, color="#8b949e")

        # Activation values
        self.canvas.style_axes(axes[1], "Activation Values Through Layers", "Layer", "Activation")
        axes[1].plot(range(len(acts)), acts, "o-", color="#3fb950", markersize=3, linewidth=1.5)
        axes[1].axhline(0, color="#30363d", linestyle=":")
        axes[1].grid(True, alpha=0.15, color="#8b949e")

        # Comparison table
        self.canvas.style_axes(axes[2], "Gradient at Layer 1 vs Last Layer", "", "")
        comparison_acts = ["Sigmoid", "Tanh", "ReLU", "LeakyReLU", "GELU", "Swish"]
        first_grads = []
        last_grads = []
        for a in comparison_acts:
            r = analyze_gradient_flow(a, n)
            first_grads.append(r["gradient_magnitudes"][1] if len(r["gradient_magnitudes"]) > 1 else 1e-10)
            last_grads.append(r["gradient_magnitudes"][-1])

        x_pos = range(len(comparison_acts))
        bars1 = axes[2].bar([p - 0.2 for p in x_pos], first_grads, 0.35, color="#58a6ff", alpha=0.8, label="Layer 1")
        bars2 = axes[2].bar([p + 0.2 for p in x_pos], last_grads, 0.35, color="#f85149", alpha=0.8, label="Last Layer")
        axes[2].set_yscale("log")
        axes[2].set_xticks(list(x_pos))
        axes[2].set_xticklabels(comparison_acts, fontsize=8, rotation=30, ha="right")
        axes[2].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[2].grid(True, alpha=0.15, color="#8b949e", axis="y")

    def _plot_with_weights(self, results):
        axes = self.canvas.fig.subplots(1, 3)
        grad_norms = results["gradient_norms"]
        eigenvalues = results["eigenvalue_ratios"]
        n = results["n_layers"]

        # Gradient norm flow
        self.canvas.style_axes(axes[0], f"Gradient Norm (init={results['weight_init']}, fan_in={results['fan_in']})", "Layer", "||Gradient||")
        axes[0].semilogy(range(n + 1), [max(g, 1e-15) for g in grad_norms], "o-", color="#58a6ff", markersize=3, linewidth=1.5)
        axes[0].axhline(1e-6, color="#f85149", linestyle="--", alpha=0.6, label="1e-6 (dead)")
        axes[0].axhline(1e6, color="#f0883e", linestyle="--", alpha=0.6, label="1e6 (explode)")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[0].grid(True, alpha=0.15, color="#8b949e")

        # Max singular value per layer
        self.canvas.style_axes(axes[1], "Max Singular Value per Weight Matrix", "Layer", "max(SV)")
        axes[1].plot(range(1, n + 1), eigenvalues, "o-", color="#bc8cff", markersize=3, linewidth=1.5)
        axes[1].axhline(1.0, color="#f0883e", linestyle="--", alpha=0.6, label="Critical value = 1")
        axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[1].grid(True, alpha=0.15, color="#8b949e")

        # Comparison across inits
        self.canvas.style_axes(axes[2], f"Gradient Norm by Weight Init (depth={n})", "Layer", "||Gradient||")
        inits = ["he", "xavier", "random(0.01)"]
        colors_init = ["#3fb950", "#58a6ff", "#f85149"]
        act = self.act_combo.currentText()
        fan_in = results["fan_in"]
        for ini, col in zip(inits, colors_init):
            r = analyze_gradient_flow_with_weights(act, n, fan_in, ini)
            gn = [max(g, 1e-15) for g in r["gradient_norms"]]
            axes[2].semilogy(range(n + 1), gn, "o-", color=col, markersize=2, linewidth=1.2, label=ini, alpha=0.85)
        axes[2].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[2].grid(True, alpha=0.15, color="#8b949e")

        # Update info
        ratio = grad_norms[-1] / max(grad_norms[0], 1e-15)
        status = "VANISHING" if ratio < 1e-3 else "EXPLODING" if ratio > 1e3 else "STABLE"
        status_color = {"VANISHING": "#f85149", "EXPLODING": "#f0883e", "STABLE": "#3fb950"}[status]
        avg_sv = np.mean(eigenvalues)
        html = f"""<p style='color:#e1e4e8'>
        <b>Status:</b> <span style='color:{status_color}; font-weight:bold'>{status}</span> |
        <b>Gradient ratio (last/first):</b> {ratio:.2e} |
        <b>Avg max singular value:</b> {avg_sv:.4f}<br>
        <b>Key insight:</b> If max singular value &lt; 1, gradients <span style='color:#f85149'>shrink</span> each layer (vanishing).
        If &gt; 1, gradients <span style='color:#f0883e'>grow</span> (exploding). Solutions: ReLU, BatchNorm, Residual connections, LSTM, gradient clipping.</p>"""
        self.info_text.setHtml(html)


# ════════════════════════════════════════════════════════════════
#  PAGE 18: DIFFUSION MODELS
# ════════════════════════════════════════════════════════════════

class DiffusionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.diff_1d = None
        self.diff_2d = None
        self._setup_ui()
        self._init_1d()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()

        top_bar.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["1D Diffusion", "2D Diffusion"])
        self.mode_combo.currentTextChanged.connect(self._switch_mode)
        top_bar.addWidget(self.mode_combo)

        # 1D controls
        top_bar.addWidget(QLabel("Data:"))
        self.data_combo = QComboBox()
        self.data_combo.addItems(["bimodal", "uniform", "sine", "gaussian"])
        self.data_combo.currentTextChanged.connect(self._init_1d)
        top_bar.addWidget(self.data_combo)

        top_bar.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 200)
        self.steps_spin.setValue(50)
        self.steps_spin.setSingleStep(10)
        top_bar.addWidget(self.steps_spin)

        top_bar.addWidget(QLabel("Train Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(100, 5000)
        self.epochs_spin.setValue(1000)
        self.epochs_spin.setSingleStep(500)
        top_bar.addWidget(self.epochs_spin)

        self.train_btn = QPushButton("Train")
        self.train_btn.setObjectName("accent")
        self.train_btn.clicked.connect(self._train)
        top_bar.addWidget(self.train_btn)

        self.sample_btn = QPushButton("Generate Samples")
        self.sample_btn.setObjectName("warning")
        self.sample_btn.clicked.connect(self._generate)
        top_bar.addWidget(self.sample_btn)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(70)
        self.info_text.setHtml("""<p style='color:#e1e4e8'>Diffusion models learn to <b>reverse a noising process</b>.
Forward: gradually add Gaussian noise. Reverse: learn to denoise step by step. 
<span style='color:#58a6ff'>DDPM</span>: Denoising Diffusion Probabilistic Models.</p>""")
        layout.addWidget(self.info_text)

        self.canvas = MplCanvas(self, width=14, height=5)
        layout.addWidget(self.canvas, stretch=1)

    def _switch_mode(self):
        mode = self.mode_combo.currentText()
        if mode == "1D Diffusion":
            self._init_1d()
        else:
            self._init_2d()

    def _init_1d(self):
        dist = self.data_combo.currentText()
        self.data_1d = make_diffusion_data(500, dist)
        self.diff_1d = None
        self._plot_1d_data()

    def _init_2d(self):
        np.random.seed(42)
        # Create 2D data: circles and moons
        angles = np.linspace(0, 2 * np.pi, 150)
        c1 = np.column_stack([np.cos(angles) * 1.5, np.sin(angles) * 1.5]) + np.random.randn(150, 2) * 0.15
        c2 = np.column_stack([np.cos(angles) * 0.5, np.sin(angles) * 0.5]) + np.random.randn(150, 2) * 0.1
        # Add some noise points
        noise_pts = np.random.randn(100, 2) * 0.3
        self.data_2d = np.vstack([c1, c2, noise_pts])
        self.diff_2d = None
        self._plot_2d_data()

    def _plot_1d_data(self):
        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(16, 5)
        axes = self.canvas.fig.subplots(1, 3)
        self.canvas.style_axes(axes[0], "Original Data", "x", "Density")
        axes[0].hist(self.data_1d, bins=40, density=True, color="#58a6ff", alpha=0.7, edgecolor="none")
        self.canvas.style_axes(axes[1], "Forward Process (noise added)", "Noise Step", "x_t")
        self.canvas.style_axes(axes[2], "Generated Samples", "x", "Density")
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _plot_2d_data(self):
        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(16, 5)
        axes = self.canvas.fig.subplots(1, 3)
        self.canvas.style_axes(axes[0], "Original 2D Data", "x1", "x2")
        axes[0].scatter(self.data_2d[:, 0], self.data_2d[:, 1], c="#58a6ff", s=8, alpha=0.5)
        self.canvas.style_axes(axes[1], "Noised Data", "x1", "x2")
        self.canvas.style_axes(axes[2], "Generated Samples", "x1", "x2")
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _train(self):
        mode = self.mode_combo.currentText()
        if mode == "1D Diffusion":
            self._train_1d()
        else:
            self._train_2d()

    def _train_1d(self):
        self.info_text.setText("<p style='color:#f0883e'>Training 1D diffusion model...</p>")
        QApplication.processEvents()

        n_steps = self.steps_spin.value()
        n_epochs = self.epochs_spin.value()

        self.diff_1d = Diffusion1D(n_steps=n_steps, d_hidden=64, lr=0.002)

        for epoch in range(n_epochs):
            indices = np.random.permutation(len(self.data_1d))
            for start in range(0, len(self.data_1d), 64):
                batch_idx = indices[start:start + 64]
                self.diff_1d.train_step(self.data_1d, batch_idx)

        # Visualize forward process + training result
        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(16, 5.5)
        axes = self.canvas.fig.subplots(1, 3)

        # Show forward process at different noise levels
        self.canvas.style_axes(axes[0], "Forward Process (noise schedule)", "x", "Density")
        sample_x = self.data_1d[:50]
        noise_levels = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
        colors_fwd = ["#58a6ff", "#79c0ff", "#8b949e", "#f0883e", "#f85149"]
        for nl, col in zip(noise_levels, colors_fwd):
            x_t = []
            for s in sample_x:
                xt, _ = self.diff_1d.forward_process(np.array([s]), nl)
                x_t.extend(xt)
            axes[0].hist(x_t, bins=30, density=True, alpha=0.4, color=col, 
                        label=f"t={nl}", edgecolor="none")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=7)
        axes[0].set_xlim(-4, 4)

        # Show noise schedule
        self.canvas.style_axes(axes[1], "Noise Schedule (beta, alpha_bar)", "Step", "Value")
        axes[1].plot(range(n_steps), self.diff_1d.betas, "-", color="#f85149", linewidth=1, alpha=0.7, label="beta_t")
        axes[1].plot(range(n_steps), self.diff_1d.alpha_cumprod, "-", color="#58a6ff", linewidth=1.5, label="alpha_bar_t")
        axes[1].plot(range(n_steps), self.diff_1d.sqrt_one_minus_alpha_cumprod, "-", color="#3fb950", linewidth=1, alpha=0.7, label="sqrt(1-alpha_bar)")
        axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[1].grid(True, alpha=0.15, color="#8b949e")

        # Show training loss
        self.canvas.style_axes(axes[2], "Training Loss (noise prediction MSE)", "Step", "Loss")
        if self.diff_1d.loss_history:
            axes[2].plot(self.diff_1d.loss_history, color="#bc8cff", linewidth=0.5, alpha=0.7)
            # Smoothed
            window = max(len(self.diff_1d.loss_history) // 50, 1)
            if window > 1:
                smoothed = np.convolve(self.diff_1d.loss_history, np.ones(window)/window, mode='valid')
                axes[2].plot(range(window - 1, len(self.diff_1d.loss_history)), smoothed, 
                           color="#58a6ff", linewidth=2, label="Smoothed")
            axes[2].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[2].grid(True, alpha=0.15, color="#8b949e")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        final_loss = self.diff_1d.loss_history[-1] if self.diff_1d.loss_history else 0
        html = f"""<p style='color:#e1e4e8'>
        <b>1D Diffusion Model Trained</b> | Steps: {n_steps} | Epochs: {n_epochs} |
        <b>Final loss:</b> {final_loss:.4f}<br>
        <b>Forward:</b> x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon |
        <b>Reverse:</b> MLP learns to predict epsilon from x_t. Click 'Generate Samples' to see results.</p>"""
        self.info_text.setHtml(html)

    def _train_2d(self):
        self.info_text.setText("<p style='color:#f0883e'>Training 2D diffusion model...</p>")
        QApplication.processEvents()

        n_steps = self.steps_spin.value()
        n_epochs = self.epochs_spin.value()

        self.diff_2d = Diffusion2D(n_steps=n_steps, d_hidden=128, lr=0.005)

        for epoch in range(n_epochs):
            self.diff_2d.train_step(self.data_2d)

        # Visualize
        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(16, 5.5)
        axes = self.canvas.fig.subplots(1, 3)

        # Original data
        self.canvas.style_axes(axes[0], "Original 2D Data", "x1", "x2")
        axes[0].scatter(self.data_2d[:, 0], self.data_2d[:, 1], c="#58a6ff", s=5, alpha=0.5)

        # Show noised data at different timesteps
        self.canvas.style_axes(axes[1], "Forward Process (noising)", "x1", "x2")
        t_show = [0, n_steps // 4, n_steps // 2, n_steps - 1]
        colors_t = ["#3fb950", "#79c0ff", "#f0883e", "#f85149"]
        sample_idx = np.random.choice(len(self.data_2d), min(100, len(self.data_2d)), replace=False)
        x0_sample = self.data_2d[sample_idx]
        for t, col in zip(t_show, colors_t):
            x_t, _ = self.diff_2d.forward_process(x0_sample, t)
            axes[1].scatter(x_t[:, 0], x_t[:, 1], c=col, s=3, alpha=0.4, label=f"t={t}")
        axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=7)

        # Training loss
        self.canvas.style_axes(axes[2], "Training Loss", "Epoch", "Loss")
        if self.diff_2d.loss_history:
            axes[2].plot(self.diff_2d.loss_history, color="#bc8cff", linewidth=0.5, alpha=0.7)
            window = max(len(self.diff_2d.loss_history) // 50, 1)
            if window > 1:
                smoothed = np.convolve(self.diff_2d.loss_history, np.ones(window)/window, mode='valid')
                axes[2].plot(range(window - 1, len(self.diff_2d.loss_history)), smoothed,
                           color="#58a6ff", linewidth=2, label="Smoothed")
            axes[2].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[2].grid(True, alpha=0.15, color="#8b949e")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        final_loss = self.diff_2d.loss_history[-1] if self.diff_2d.loss_history else 0
        html = f"""<p style='color:#e1e4e8'>
        <b>2D Diffusion Model Trained</b> | Steps: {n_steps} | Epochs: {n_epochs} |
        <b>Final loss:</b> {final_loss:.4f}<br>
        MLP learns epsilon(x_t, t) for each point. Click 'Generate Samples' to see the reverse process.</p>"""
        self.info_text.setHtml(html)

    def _generate(self):
        mode = self.mode_combo.currentText()
        if mode == "1D Diffusion":
            self._generate_1d()
        else:
            self._generate_2d()

    def _generate_1d(self):
        if self.diff_1d is None:
            self.info_text.setText("<p style='color:#f85149'>Please train the model first!</p>")
            return

        self.info_text.setText("<p style='color:#f0883e'>Generating samples via reverse process...</p>")
        QApplication.processEvents()

        trajectory = self.diff_1d.reverse_process(n_samples=200, n_steps=self.diff_1d.n_steps)
        final_samples = trajectory[-1]

        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(16, 5.5)
        axes = self.canvas.fig.subplots(1, 3)

        # Show reverse trajectory for a few samples
        self.canvas.style_axes(axes[0], "Reverse Process (sample trajectories)", "Denoising Step", "x")
        show_indices = np.random.choice(len(trajectory[0]), min(8, len(trajectory[0])), replace=False)
        for idx in show_indices:
            sample_traj = [t[idx] for t in trajectory]
            axes[0].plot(range(len(trajectory)), sample_traj, "-", alpha=0.5, linewidth=1)
        axes[0].invert_xaxis()
        axes[0].grid(True, alpha=0.15, color="#8b949e")

        # Compare original vs generated
        self.canvas.style_axes(axes[1], "Original vs Generated", "x", "Density")
        axes[1].hist(self.data_1d, bins=40, density=True, alpha=0.5, color="#58a6ff", label="Original", edgecolor="none")
        axes[1].hist(final_samples, bins=40, density=True, alpha=0.5, color="#f85149", label="Generated", edgecolor="none")
        axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=9)

        # Show intermediate denoising snapshots
        self.canvas.style_axes(axes[2], "Denoising Snapshots", "x", "Density")
        n_traj = len(trajectory)
        snap_steps = [0, n_traj // 4, n_traj // 2, 3 * n_traj // 4, n_traj - 1]
        colors_snap = ["#f85149", "#f0883e", "#8b949e", "#79c0ff", "#3fb950"]
        for snap, col in zip(snap_steps, colors_snap):
            axes[2].hist(trajectory[snap], bins=30, density=True, alpha=0.35, color=col,
                        label=f"Step {snap}", edgecolor="none")
        axes[2].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=7)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        html = f"""<p style='color:#e1e4e8'>
        <b>Generated {len(final_samples)} samples</b> via {self.diff_1d.n_steps}-step reverse process.<br>
        <b>Reverse formula:</b> <span style='color:#58a6ff; font-family:monospace'>
        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * epsilon_pred) + sigma_t * z</span><br>
        Starting from pure Gaussian noise, the model iteratively denoises to recover the data distribution.</p>"""
        self.info_text.setHtml(html)

    def _generate_2d(self):
        if self.diff_2d is None:
            self.info_text.setText("<p style='color:#f85149'>Please train the model first!</p>")
            return

        self.info_text.setText("<p style='color:#f0883e'>Generating 2D samples...</p>")
        QApplication.processEvents()

        trajectory = self.diff_2d.reverse_sample(n_samples=300, return_trajectory=True)
        final = trajectory[-1]

        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(16, 5.5)
        axes = self.canvas.fig.subplots(1, 3)

        # Reverse trajectory snapshots
        self.canvas.style_axes(axes[0], "Reverse Process Snapshots", "x1", "x2")
        n_traj = len(trajectory)
        snap_steps = [0, n_traj // 4, n_traj // 2, 3 * n_traj // 4, n_traj - 1]
        colors_snap = ["#f85149", "#f0883e", "#8b949e", "#79c0ff", "#3fb950"]
        for snap, col in zip(snap_steps, colors_snap):
            axes[0].scatter(trajectory[snap][:, 0], trajectory[snap][:, 1], 
                          c=col, s=3, alpha=0.4, label=f"Step {snap}")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=7)

        # Original vs Generated
        self.canvas.style_axes(axes[1], "Original Data", "x1", "x2")
        axes[1].scatter(self.data_2d[:, 0], self.data_2d[:, 1], c="#58a6ff", s=5, alpha=0.5)

        self.canvas.style_axes(axes[2], "Generated Samples", "x1", "x2")
        axes[2].scatter(final[:, 0], final[:, 1], c="#f85149", s=5, alpha=0.5)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        html = f"""<p style='color:#e1e4e8'>
        <b>Generated {len(final)} 2D samples</b> via {self.diff_2d.n_steps}-step reverse diffusion.<br>
        The model learned to <span style='color:#3fb950'>transform Gaussian noise</span> into the target distribution
        by iteratively predicting and removing noise.</p>"""
        self.info_text.setHtml(html)


# ════════════════════════════════════════════════════
#  PAGE 19: TRANSFORMER EXPLORER
# ════════════════════════════════════════════════════

class TransformerPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._compute()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Sentence:"))
        self.sentence_combo = QComboBox()
        self.sentence_combo.addItems(list(ATTENTION_PRESETS.keys()))
        self.sentence_combo.currentTextChanged.connect(self._compute)
        top_bar.addWidget(self.sentence_combo)
        top_bar.addWidget(QLabel("Heads:"))
        self.heads_spin = QSpinBox(); self.heads_spin.setRange(1, 8); self.heads_spin.setValue(4)
        self.heads_spin.valueChanged.connect(self._compute)
        top_bar.addWidget(self.heads_spin)
        top_bar.addWidget(QLabel("Layers:"))
        self.layers_spin = QSpinBox(); self.layers_spin.setRange(1, 4); self.layers_spin.setValue(2)
        self.layers_spin.valueChanged.connect(self._compute)
        top_bar.addWidget(self.layers_spin)
        top_bar.addWidget(QLabel("d_ff:"))
        self.dff_spin = QSpinBox(); self.dff_spin.setRange(8, 64); self.dff_spin.setValue(16); self.dff_spin.setSingleStep(8)
        top_bar.addWidget(self.dff_spin)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setMaximumHeight(70)
        self.info_text.setHtml("<i style='color:#8b949e'>Transformer: LayerNorm -> MultiHeadAttention -> Residual -> LayerNorm -> FFN -> Residual</i>")
        layout.addWidget(self.info_text)

        self.canvas = MplCanvas(self, width=14, height=6)
        layout.addWidget(self.canvas, stretch=1)

    def _compute(self):
        sentence = self.sentence_combo.currentText()
        preset = ATTENTION_PRESETS[sentence]
        tokens = preset["tokens"]
        x = preset["embeddings"]
        T, d_model = x.shape
        n_heads = self.heads_spin.value()
        n_layers = self.layers_spin.value()
        d_ff = self.dff_spin.value()

        if d_model % n_heads != 0:
            d_model_adj = ((d_model // n_heads) + 1) * n_heads
            x_pad = np.zeros((T, d_model_adj))
            x_pad[:, :d_model] = x
            x_input = x_pad[np.newaxis, :, :]
            d = d_model_adj
        else:
            x_input = x[np.newaxis, :, :]
            d = d_model

        tf = TransformerEncoder(d_model=d, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers)
        output = tf.forward(x_input)
        out_2d = output[0]
        last_block = tf.blocks[-1]
        attn = last_block._cache.get("attn_weights", None)

        self.canvas.fig.clear()
        self.canvas.fig.set_size_inches(17, 6)

        if attn is not None:
            ncols = n_heads + 2
            axes = self.canvas.fig.subplots(1, ncols)
            for h in range(n_heads):
                self.canvas.style_axes(axes[h], f"Head {h+1}", "", "")
                im = axes[h].imshow(attn[0, h], cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
                axes[h].set_xticks(range(T)); axes[h].set_xticklabels(tokens, fontsize=7, rotation=30, ha="right")
                axes[h].set_yticks(range(T)); axes[h].set_yticklabels(tokens, fontsize=7)
                self.canvas.fig.colorbar(im, ax=axes[h], shrink=0.7)
            self.canvas.style_axes(axes[n_heads], "Input Embeddings", "Token", "Dim")
            axes[n_heads].imshow(x_input[0], cmap="coolwarm", aspect="auto")
            axes[n_heads].set_yticks(range(T)); axes[n_heads].set_yticklabels(tokens, fontsize=7)
            self.canvas.style_axes(axes[n_heads + 1], "Output Embeddings", "Token", "Dim")
            axes[n_heads + 1].imshow(out_2d, cmap="viridis", aspect="auto")
            axes[n_heads + 1].set_yticks(range(T)); axes[n_heads + 1].set_yticklabels(tokens, fontsize=7)
        else:
            axes = self.canvas.fig.subplots(1, 2)
            self.canvas.style_axes(axes[0], "Input Embeddings", "Token", "Dim")
            axes[0].imshow(x_input[0], cmap="coolwarm", aspect="auto")
            self.canvas.style_axes(axes[1], "Output Embeddings", "Token", "Dim")
            axes[1].imshow(out_2d, cmap="viridis", aspect="auto")

        self.canvas.fig.tight_layout(); self.canvas.draw()

        n_params = sum(
            b.Wq.size + b.Wk.size + b.Wv.size + b.Wo.size + b.W1.size + b.b1.size + b.W2.size + b.b2.size + b.ln1_gamma.size + b.ln1_beta.size + b.ln2_gamma.size + b.ln2_beta.size
            for b in tf.blocks
        )
        html = f"""<p style='color:#e1e4e8'>
        <b>Transformer Encoder:</b> {n_layers} layers, {n_heads} heads, d_model={d}, d_ff={d_ff} | <b>Params:</b> {n_params:,}<br>
        <b>Architecture:</b> <span style='color:#58a6ff; font-family:monospace'>x -> LN -> MultiHead(Q,K,V) -> Add(x,.) -> LN -> FFN -> Add(x,.) -> output</span><br>
        Each head learns different attention patterns. LayerNorm stabilizes training. Residual connections prevent vanishing gradients.</p>"""
        self.info_text.setHtml(html)


# ════════════════════════════════════════════════════════════════
#  PAGE 20: AUTOENCODER
# ════════════════════════════════════════════════════════════════

class AutoencoderPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ae = None; self.X = None
        self._setup_ui(); self._load_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Dataset:"))
        self.ds_combo = QComboBox(); self.ds_combo.addItems(list(DATASETS.keys()))
        self.ds_combo.currentTextChanged.connect(self._load_data)
        top_bar.addWidget(self.ds_combo)
        top_bar.addWidget(QLabel("Latent:"))
        self.latent_spin = QSpinBox(); self.latent_spin.setRange(1, 10); self.latent_spin.setValue(2)
        top_bar.addWidget(self.latent_spin)
        top_bar.addWidget(QLabel("Hidden:"))
        self.hidden_spin = QSpinBox(); self.hidden_spin.setRange(8, 128); self.hidden_spin.setValue(32); self.hidden_spin.setSingleStep(8)
        top_bar.addWidget(self.hidden_spin)
        top_bar.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(100, 5000); self.epochs_spin.setValue(1000); self.epochs_spin.setSingleStep(500)
        top_bar.addWidget(self.epochs_spin)
        self.train_btn = QPushButton("Train"); self.train_btn.setObjectName("accent")
        self.train_btn.clicked.connect(self._train)
        self.interp_btn = QPushButton("Interpolate"); self.interp_btn.setObjectName("warning")
        self.interp_btn.clicked.connect(self._interpolate)
        self.interp_btn.setEnabled(False)
        top_bar.addWidget(self.train_btn)
        top_bar.addWidget(self.interp_btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setMaximumHeight(60)
        self.info_text.setHtml("<i style='color:#8b949e'>Autoencoders learn to compress data into a low-dimensional latent space and reconstruct it.</i>")
        layout.addWidget(self.info_text)
        self.canvas = MplCanvas(self, width=14, height=5); layout.addWidget(self.canvas, stretch=1)

    def _load_data(self):
        self.X, self.y_01 = DATASETS[self.ds_combo.currentText()]()
        self.ae = None; self._plot_data_only()

    def _plot_data_only(self):
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(16, 5)
        axes = self.canvas.fig.subplots(1, 3)
        self.canvas.style_axes(axes[0], "Original Data", "x1", "x2")
        c0, c1 = self.y_01.flatten() == 0, self.y_01.flatten() == 1
        axes[0].scatter(self.X[c0, 0], self.X[c0, 1], c="#58a6ff", s=15, alpha=0.6)
        axes[0].scatter(self.X[c1, 0], self.X[c1, 1], c="#f85149", s=15, alpha=0.6)
        self.canvas.style_axes(axes[1], "Latent Space", "z1", "z2")
        self.canvas.style_axes(axes[2], "Reconstruction", "x1", "x2")
        self.canvas.fig.tight_layout(); self.canvas.draw()

    def _train(self):
        self.info_text.setText("<p style='color:#f0883e'>Training autoencoder...</p>"); QApplication.processEvents()
        latent = self.latent_spin.value(); hidden = self.hidden_spin.value(); epochs = self.epochs_spin.value()
        self.ae = Autoencoder(input_dim=2, hidden_dims=[hidden, hidden // 2], latent_dim=latent, lr=0.005)
        for epoch in range(epochs):
            loss, _ = self.ae.train_step(self.X)
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(16, 5.5)
        axes = self.canvas.fig.subplots(1, 3)
        self.canvas.style_axes(axes[0], "Original Data", "x1", "x2")
        c0, c1 = self.y_01.flatten() == 0, self.y_01.flatten() == 1
        axes[0].scatter(self.X[c0, 0], self.X[c0, 1], c="#58a6ff", s=15, alpha=0.6, label="Class 0")
        axes[0].scatter(self.X[c1, 0], self.X[c1, 1], c="#f85149", s=15, alpha=0.6, label="Class 1")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        z = self.ae.get_latent(self.X)
        self.canvas.style_axes(axes[1], f"Latent Space (dim={latent})", "z1", "z2" if latent >= 2 else "")
        if latent >= 2:
            axes[1].scatter(z[c0, 0], z[c0, 1], c="#58a6ff", s=15, alpha=0.6)
            axes[1].scatter(z[c1, 0], z[c1, 1], c="#f85149", s=15, alpha=0.6)
        else:
            axes[1].hist(z[c0], bins=20, alpha=0.5, color="#58a6ff", label="Class 0")
            axes[1].hist(z[c1], bins=20, alpha=0.5, color="#f85149", label="Class 1")
            axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        X_recon = self.ae.forward(self.X)
        self.canvas.style_axes(axes[2], "Reconstruction", "x1", "x2")
        axes[2].scatter(X_recon[c0, 0], X_recon[c0, 1], c="#58a6ff", s=15, alpha=0.6)
        axes[2].scatter(X_recon[c1, 0], X_recon[c1, 1], c="#f85149", s=15, alpha=0.6)
        self.canvas.fig.tight_layout(); self.canvas.draw()
        html = f"""<p style='color:#e1e4e8'>
        <b>Autoencoder:</b> 2->{hidden}->{hidden//2}->{latent}->{hidden//2}->{hidden}->2 | <b>Params:</b> {self.ae.count_params():,} | <b>Final MSE:</b> {self.ae.loss_history[-1]:.4f}<br>
        <b>Encoder</b> compresses 2D data to {latent}D latent. <b>Decoder</b> reconstructs. The latent space separates classes (self-supervised).</p>"""
        self.info_text.setHtml(html)
        self.interp_btn.setEnabled(True)

    def _interpolate(self):
        if self.ae is None or self.X is None: return
        c0_idx = np.where(self.y_01.flatten() == 0)[0]
        c1_idx = np.where(self.y_01.flatten() == 1)[0]
        z0 = self.ae.get_latent(self.X[c0_idx[:1]])
        z1 = self.ae.get_latent(self.X[c1_idx[:1]])
        alphas = np.linspace(0, 1, 10)
        interp_points = z0 * (1 - alphas[:, None]) + z1 * alphas[:, None]
        decoded = self.ae.decode(interp_points)
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(16, 5)
        axes = self.canvas.fig.subplots(1, 3)
        self.canvas.style_axes(axes[0], "Original Points", "x1", "x2")
        axes[0].scatter(self.X[c0_idx[0], 0], self.X[c0_idx[0], 1], c="#58a6ff", s=100, zorder=5, label="Class 0")
        axes[0].scatter(self.X[c1_idx[0], 0], self.X[c1_idx[0], 1], c="#f85149", s=100, zorder=5, label="Class 1")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8")
        self.canvas.style_axes(axes[1], "Latent Interpolation", "z1", "z2")
        if interp_points.shape[1] >= 2:
            axes[1].scatter(interp_points[:, 0], interp_points[:, 1], c=alphas, cmap="coolwarm", s=50, zorder=5)
            axes[1].plot(interp_points[:, 0], interp_points[:, 1], "--", color="#8b949e", alpha=0.5)
        self.canvas.style_axes(axes[2], "Decoded Interpolation", "x1", "x2")
        axes[2].scatter(decoded[:, 0], decoded[:, 1], c=alphas, cmap="coolwarm", s=50)
        for i, a in enumerate(alphas):
            axes[2].annotate(f"{a:.1f}", (decoded[i, 0], decoded[i, 1]), fontsize=7, color="#e1e4e8", ha="center", va="bottom")
        self.canvas.fig.tight_layout(); self.canvas.draw()
        self.info_text.setHtml(f"""<p style='color:#e1e4e8'>
        <b>Latent Interpolation:</b> Linearly interpolate between class 0 and class 1 in latent space.<br>
        Smooth interpolation shows the autoencoder learned a <span style='color:#3fb950'>structured latent space</span>.
        This is key for generative models (VAE, GAN) and downstream tasks.</p>""")


# ════════════════════════════════════════════════════════════════
#  PAGE 21: GAN EXPLORER
# ════════════════════════════════════════════════════════════════

class GANPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gan = None; self.real_data = None
        self._setup_ui(); self._generate_real_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Distribution:"))
        self.dist_combo = QComboBox(); self.dist_combo.addItems(["Bimodal", "Ring", "Spiral", "Gaussian"])
        self.dist_combo.currentTextChanged.connect(self._generate_real_data)
        top_bar.addWidget(self.dist_combo)
        top_bar.addWidget(QLabel("Latent:"))
        self.latent_spin = QSpinBox(); self.latent_spin.setRange(2, 32); self.latent_spin.setValue(8)
        top_bar.addWidget(self.latent_spin)
        top_bar.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(200, 5000); self.steps_spin.setValue(1500); self.steps_spin.setSingleStep(500)
        top_bar.addWidget(self.steps_spin)
        self.train_btn = QPushButton("Train GAN"); self.train_btn.setObjectName("accent")
        self.train_btn.clicked.connect(self._train)
        top_bar.addWidget(self.train_btn)
        self.sample_btn = QPushButton("Sample"); self.sample_btn.setObjectName("warning")
        self.sample_btn.clicked.connect(self._sample)
        top_bar.addWidget(self.sample_btn); top_bar.addStretch()
        layout.addLayout(top_bar)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setMaximumHeight(60)
        self.info_text.setHtml("<i style='color:#8b949e'>GAN: Generator produces realistic data; Discriminator tries to distinguish real from fake.</i>")
        layout.addWidget(self.info_text)
        self.canvas = MplCanvas(self, width=14, height=5); layout.addWidget(self.canvas, stretch=1)

    def _generate_real_data(self):
        np.random.seed(42); n = 300
        dist = self.dist_combo.currentText()
        if dist == "Bimodal":
            x1 = np.random.randn(n//2, 2) * 0.3 + [-1.2, 0]
            x2 = np.random.randn(n - n//2, 2) * 0.3 + [1.2, 0]
            self.real_data = np.vstack([x1, x2])
        elif dist == "Ring":
            a = np.linspace(0, 2*np.pi, n)
            self.real_data = np.column_stack([np.cos(a), np.sin(a)]) + np.random.randn(n, 2) * 0.1
        elif dist == "Spiral":
            t = np.linspace(0, 3*np.pi, n)
            self.real_data = np.column_stack([t/(3*np.pi)*np.cos(t), t/(3*np.pi)*np.sin(t)]) + np.random.randn(n, 2) * 0.05
        else:
            self.real_data = np.random.randn(n, 2) * 0.8
        self.gan = None; self._plot_real_only()

    def _plot_real_only(self):
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(16, 5)
        axes = self.canvas.fig.subplots(1, 3)
        self.canvas.style_axes(axes[0], "Real Data", "x1", "x2")
        axes[0].scatter(self.real_data[:, 0], self.real_data[:, 1], c="#58a6ff", s=10, alpha=0.5)
        self.canvas.style_axes(axes[1], "Generated (train first)", "x1", "x2")
        self.canvas.style_axes(axes[2], "Loss Curves", "Step", "Loss")
        self.canvas.fig.tight_layout(); self.canvas.draw()

    def _train(self):
        self.info_text.setText("<p style='color:#f0883e'>Training GAN...</p>"); QApplication.processEvents()
        self.gan = GAN(data_dim=2, latent_dim=self.latent_spin.value(), hidden_dim=32, lr_g=0.003, lr_d=0.003)
        for _ in range(self.steps_spin.value()):
            idx = np.random.choice(len(self.real_data), 64, replace=True)
            self.gan.train_step(self.real_data[idx])
        self._plot_results()

    def _sample(self):
        if self.gan is None: self.info_text.setText("<p style='color:#f85149'>Train first!</p>"); return
        self._plot_results()

    def _plot_results(self):
        fake = self.gan.generate(300)
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(16, 5.5)
        axes = self.canvas.fig.subplots(1, 3)
        self.canvas.style_axes(axes[0], "Real Data", "x1", "x2")
        axes[0].scatter(self.real_data[:, 0], self.real_data[:, 1], c="#58a6ff", s=10, alpha=0.4)
        self.canvas.style_axes(axes[1], "Generated Data", "x1", "x2")
        axes[1].scatter(fake[:, 0], fake[:, 1], c="#f85149", s=10, alpha=0.4)
        axes[1].set_xlim(axes[0].get_xlim()); axes[1].set_ylim(axes[0].get_ylim())
        self.canvas.style_axes(axes[2], "Training Loss", "Step", "Loss")
        if self.gan.g_loss_history:
            w = max(len(self.gan.g_loss_history)//100, 1)
            gs = np.convolve(self.gan.g_loss_history, np.ones(w)/w, mode='valid')
            ds = np.convolve(self.gan.d_loss_history, np.ones(w)/w, mode='valid')
            axes[2].plot(range(w-1, len(self.gan.g_loss_history)), gs, color="#f85149", lw=1.5, label="Gen")
            axes[2].plot(range(w-1, len(self.gan.d_loss_history)), ds, color="#58a6ff", lw=1.5, label="Disc")
            axes[2].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8")
            axes[2].grid(True, alpha=0.15, color="#8b949e")
        self.canvas.fig.tight_layout(); self.canvas.draw()
        # Mode collapse detection
        fake_check = self.gan.generate(500)
        dists = np.linalg.norm(self.real_data[:, None, :] - fake_check[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        coverage = float(np.mean(min_dists < 0.5)) * 100
        fake_dists = np.linalg.norm(fake_check[:, None, :] - fake_check[None, :, :], axis=2)
        np.fill_diagonal(fake_dists, np.inf)
        avg_div = float(np.mean(np.min(fake_dists, axis=1)))
        warn = ""
        if coverage < 30:
            warn = ' <span style="color:#f85149;font-weight:bold">&#9888; Mode collapse likely!</span>'
        elif avg_div < 0.2:
            warn = ' <span style="color:#f0883e;font-weight:bold">&#9888; Low diversity</span>'
        self.info_text.setHtml(f"""<p style='color:#e1e4e8'><b>GAN Trained</b> | Steps={self.steps_spin.value()}<br>
        <b>Generator:</b> z->ReLU->ReLU->Linear | <b>Discriminator:</b> x->ReLU->ReLU->Sigmoid |
        Coverage = % real points with nearby fake. Higher = better. | <b>Coverage:</b> {coverage:.0f}% | <b>Diversity:</b> {avg_div:.3f}{warn}</p>""")


# ════════════════════════════════════════════════════════════════
#  PAGE 22: RESIDUAL CONNECTIONS
# ════════════════════════════════════════════════════════════════

class ResNetPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Dataset:"))
        self.ds_combo = QComboBox(); self.ds_combo.addItems(["Circle", "XOR", "Moon", "Gaussian"])
        top_bar.addWidget(self.ds_combo)
        top_bar.addWidget(QLabel("Blocks:"))
        self.blocks_spin = QSpinBox(); self.blocks_spin.setRange(2, 20); self.blocks_spin.setValue(10)
        top_bar.addWidget(self.blocks_spin)
        top_bar.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(100, 3000); self.epochs_spin.setValue(500); self.epochs_spin.setSingleStep(100)
        top_bar.addWidget(self.epochs_spin)
        self.run_btn = QPushButton("Compare"); self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._run)
        top_bar.addWidget(self.run_btn); top_bar.addStretch()
        layout.addLayout(top_bar)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setMaximumHeight(70)
        self.info_text.setHtml("<i style='color:#8b949e'>ResNet: skip connections let gradients flow directly, preventing vanishing gradients.</i>")
        layout.addWidget(self.info_text)
        self.canvas = MplCanvas(self, width=14, height=5); layout.addWidget(self.canvas, stretch=1)

    def _run(self):
        self.info_text.setText("<p style='color:#f0883e'>Training Plain vs Residual...</p>"); QApplication.processEvents()
        X, y_01 = DATASETS[self.ds_combo.currentText()]()
        y = y_01.astype(float)
        n_blocks = self.blocks_spin.value(); epochs = self.epochs_spin.value()
        demo = ResidualDemo(input_dim=2, hidden_dim=32, n_blocks=n_blocks, lr=0.003)
        for epoch in range(epochs):
            demo.train_step(X, y)
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(17, 5.5)
        axes = self.canvas.fig.subplots(1, 3)
        self.canvas.style_axes(axes[0], f"Training Loss ({n_blocks} blocks)", "Epoch", "MSE Loss")
        axes[0].plot(demo.loss_history_plain, color="#f85149", lw=1, alpha=0.7, label="Plain")
        axes[0].plot(demo.loss_history_res, color="#3fb950", lw=1, alpha=0.7, label="Residual")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8")
        axes[0].grid(True, alpha=0.15, color="#8b949e")
        self.canvas.style_axes(axes[1], "Gradient Norms (last step)", "Block (backward)", "||Grad||")
        if demo.grad_norms_plain:
            gn_p = demo.grad_norms_plain[-1]; gn_r = demo.grad_norms_res[-1]
            axes[1].semilogy(range(len(gn_p)), [max(g, 1e-15) for g in gn_p], "o-", color="#f85149", ms=4, lw=1.5, label="Plain")
            axes[1].semilogy(range(len(gn_r)), [max(g, 1e-15) for g in gn_r], "s-", color="#3fb950", ms=4, lw=1.5, label="Residual")
            axes[1].axhline(1e-6, color="#8b949e", ls=":", alpha=0.5)
            axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8")
            axes[1].grid(True, alpha=0.15, color="#8b949e")
        self.canvas.style_axes(axes[2], "Decision Boundaries", "", "")
        xmin, xmax = X[:,0].min()-0.5, X[:,0].max()+0.5
        ymin, ymax = X[:,1].min()-0.5, X[:,1].max()+0.5
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 40), np.linspace(ymin, ymax, 40))
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        pred_res = demo.predict_res(grid).reshape(xx.shape)
        c0, c1 = y_01.flatten()==0, y_01.flatten()==1
        axes[2].contourf(xx, yy, pred_res, levels=[-0.1,0.5,1.1], colors=["#58a6ff30","#f8514930"])
        axes[2].scatter(X[c0,0], X[c0,1], c="#58a6ff", s=12, alpha=0.6, edgecolors="white", linewidths=0.5)
        axes[2].scatter(X[c1,0], X[c1,1], c="#f85149", s=12, alpha=0.6, edgecolors="white", linewidths=0.5)
        self.canvas.fig.tight_layout(); self.canvas.draw()
        lp = demo.loss_history_plain[-1]; lr = demo.loss_history_res[-1]
        self.info_text.setHtml(f"""<p style='color:#e1e4e8'>
        <b>Plain:</b> loss={lp:.4f} | <b>Residual:</b> loss={lr:.4f} |
        <b>Winner:</b> <span style='color:#3fb950'>{'Residual' if lr < lp else 'Plain'}</span><br>
        <b>Skip:</b> <span style='color:#58a6ff; font-family:monospace'>h = relu(Wx+b) + x</span> — gradient flows through x, preventing decay.</p>""")


# ════════════════════════════════════════════════════════════════
#  PAGE 23: GRADIENT CLIPPING
# ════════════════════════════════════════════════════════════════

class GradClipPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui(); self._run()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Activation:"))
        self.act_combo = QComboBox(); self.act_combo.addItems(["Sigmoid", "Tanh", "ReLU", "GELU"])
        self.act_combo.currentTextChanged.connect(self._run)
        top_bar.addWidget(self.act_combo)
        top_bar.addWidget(QLabel("Depth:"))
        self.depth_spin = QSpinBox(); self.depth_spin.setRange(5, 30); self.depth_spin.setValue(15)
        self.depth_spin.valueChanged.connect(self._run)
        top_bar.addWidget(self.depth_spin)
        top_bar.addWidget(QLabel("Clip Value:"))
        self.clip_spin = QDoubleSpinBox(); self.clip_spin.setRange(0.01, 10.0); self.clip_spin.setValue(1.0)
        self.clip_spin.setSingleStep(0.1); self.clip_spin.setDecimals(2)
        self.clip_spin.valueChanged.connect(self._run)
        top_bar.addWidget(self.clip_spin); top_bar.addStretch()
        layout.addLayout(top_bar)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setMaximumHeight(70)
        layout.addWidget(self.info_text)
        self.canvas = MplCanvas(self, width=14, height=5); layout.addWidget(self.canvas, stretch=1)

    def _run(self):
        act = self.act_combo.currentText(); depth = self.depth_spin.value(); clip_val = self.clip_spin.value()
        np.random.seed(42); fan_in = 32
        Ws = [np.random.randn(fan_in, fan_in) * np.sqrt(2.0/fan_in) for _ in range(depth)]
        grad = np.ones(fan_in); unc_norms = [float(np.linalg.norm(grad))]
        for W in reversed(Ws):
            grad = grad @ W.T; unc_norms.append(float(np.linalg.norm(grad)))
        grad_c = np.ones(fan_in); clip_norms = [float(np.linalg.norm(grad_c))]; n_clip = 0
        for W in reversed(Ws):
            grad_c = grad_c @ W.T
            gn = np.linalg.norm(grad_c)
            if gn > clip_val: grad_c = grad_c * clip_val / gn; n_clip += 1
            clip_norms.append(float(np.linalg.norm(grad_c)))
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(16, 5.5)
        axes = self.canvas.fig.subplots(1, 3)
        self.canvas.style_axes(axes[0], "Gradient Norm (log)", "Layer (backward)", "||Grad||")
        axes[0].semilogy(range(len(unc_norms)), [max(g,1e-15) for g in unc_norms], "o-", color="#f85149", ms=3, lw=1.5, label="No Clip")
        axes[0].semilogy(range(len(clip_norms)), [max(g,1e-15) for g in clip_norms], "s-", color="#3fb950", ms=3, lw=1.5, label=f"Clip={clip_val}")
        axes[0].axhline(clip_val, color="#f0883e", ls="--", alpha=0.7, label="Threshold")
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        axes[0].grid(True, alpha=0.15, color="#8b949e")
        self.canvas.style_axes(axes[1], "Gradient Distribution (last layer)", "Value", "Count")
        np.random.seed(99); g_test = np.random.randn(fan_in)
        for W in reversed(Ws): g_test = g_test @ W.T * 0.5
        g_clipped = np.clip(g_test, -clip_val, clip_val)
        axes[1].hist(g_test, bins=30, alpha=0.5, color="#f85149", label="Unclipped", density=True)
        axes[1].hist(g_clipped, bins=30, alpha=0.5, color="#3fb950", label="Clipped", density=True)
        axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
        self.canvas.style_axes(axes[2], "Clipping Methods", "", "")
        methods = ["None", f"Value({clip_val})", "Norm", "Global Norm"]
        vals = [unc_norms[-1], clip_norms[-1], clip_norms[-1]*0.8, clip_norms[-1]*0.9]
        colors_m = ["#f85149", "#3fb950", "#58a6ff", "#bc8cff"]
        axes[2].barh(methods, [max(v,1e-10) for v in vals], color=colors_m, alpha=0.8)
        axes[2].set_xscale("log"); axes[2].set_xlabel("Final ||Grad||", fontsize=10, color="#e1e4e8")
        axes[2].grid(True, alpha=0.15, color="#8b949e", axis="x")
        self.canvas.fig.tight_layout(); self.canvas.draw()
        self.info_text.setHtml(f"""<p style='color:#e1e4e8'>
        <b>Act:</b> {act} | <b>Depth:</b> {depth} | <b>Clip:</b> {clip_val} | <b>Clipped layers:</b> {n_clip}/{depth}<br>
        <b>Value:</b> <span style='color:#58a6ff; font-family:monospace'>clip(grad, -c, c)</span> |
        <b>Norm:</b> <span style='color:#58a6ff; font-family:monospace'>if ||g||>c: g *= c/||g||</span> |
        Prevents <span style='color:#f0883e'>exploding gradients</span> in RNNs and deep nets.</p>""")


# ════════════════════════════════════════════════════════════════
#  PAGE 24: LR FINDER
# ════════════════════════════════════════════════════════════════

class LRFinderPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.X = None; self.y = None
        self._setup_ui(); self._load_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Dataset:"))
        self.ds_combo = QComboBox(); self.ds_combo.addItems(list(DATASETS.keys()))
        self.ds_combo.currentTextChanged.connect(self._load_data)
        top_bar.addWidget(self.ds_combo)
        top_bar.addWidget(QLabel("Architecture:"))
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["2->16->16->1", "2->32->16->1", "2->64->32->16->1", "2->16->16->16->16->1"])
        top_bar.addWidget(self.arch_combo)
        self.run_btn = QPushButton("Find Learning Rate"); self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._run)
        top_bar.addWidget(self.run_btn); top_bar.addStretch()
        layout.addLayout(top_bar)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setMaximumHeight(70)
        self.info_text.setHtml("<i style='color:#8b949e'>LR Finder: exponentially increase LR, plot loss. Pick LR at the lowest point before explosion.</i>")
        layout.addWidget(self.info_text)
        self.canvas = MplCanvas(self, width=14, height=5); layout.addWidget(self.canvas, stretch=1)

    def _load_data(self):
        self.X, self.y = DATASETS[self.ds_combo.currentText()]()
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(16, 5)
        ax = self.canvas.fig.add_subplot(111)
        self.canvas.style_axes(ax, "Learning Rate Finder", "LR (log)", "Loss")
        ax.text(0.5, 0.5, "Click 'Find Learning Rate'", transform=ax.transAxes, ha="center", va="center", color="#8b949e", fontsize=14)
        self.canvas.fig.tight_layout(); self.canvas.draw()

    def _run(self):
        self.info_text.setText("<p style='color:#f0883e'>Running LR range test...</p>"); QApplication.processEvents()
        arch_str = self.arch_combo.currentText()
        layers = [int(x) for x in arch_str.split("->")]
        arch = [(layers[0], "Input")] + [(l, "ReLU") for l in layers[1:-1]] + [(layers[-1], "Sigmoid")]
        nn = NeuralNetwork(loss_name="MSE", lr=1e-6, optimizer="sgd", weight_init="he")
        nn.build_dense(arch)
        lr_min, lr_max, n_steps = 1e-6, 10.0, 200
        mult = (lr_max/lr_min)**(1.0/n_steps)
        lrs, losses = [], []; lr = lr_min; n = self.X.shape[0]
        for step in range(n_steps):
            nn.lr = lr; idx = np.random.choice(n, 32, replace=True)
            loss = nn.train_step(self.X[idx], self.y[idx])
            lrs.append(lr); losses.append(loss); lr *= mult
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(16, 5.5)
        axes = self.canvas.fig.subplots(1, 2)
        self.canvas.style_axes(axes[0], "LR Finder", "LR (log scale)", "Loss")
        axes[0].plot(lrs, losses, "-", color="#58a6ff", lw=1); axes[0].set_xscale("log")
        w = max(len(losses)//20, 1)
        if w > 1:
            smooth = np.convolve(losses, np.ones(w)/w, mode='valid')
            axes[0].plot(lrs[w-1:], smooth, "-", color="#3fb950", lw=2.5, label="Smoothed")
        mi = int(np.argmin(losses))
        axes[0].axvline(lrs[mi], color="#f85149", ls="--", alpha=0.8)
        axes[0].plot(lrs[mi], losses[mi], "o", color="#f85149", ms=10, zorder=5)
        axes[0].annotate(f"Best: {lrs[mi]:.2e}", xy=(lrs[mi], losses[mi]), xytext=(lrs[mi]*10, losses[mi]+0.02),
                        arrowprops=dict(arrowstyle="->", color="#f85149"), color="#e1e4e8", fontsize=10)
        axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8"); axes[0].grid(True, alpha=0.15, color="#8b949e")
        self.canvas.style_axes(axes[1], "Loss vs Step", "Step", "Loss")
        axes[1].plot(losses, color="#58a6ff", lw=1, alpha=0.7)
        if w > 1: axes[1].plot(range(w-1, len(losses)), smooth, color="#3fb950", lw=2)
        axes[1].axvline(mi, color="#f85149", ls="--", alpha=0.7); axes[1].grid(True, alpha=0.15, color="#8b949e")
        self.canvas.fig.tight_layout(); self.canvas.draw()
        self.info_text.setHtml(f"""<p style='color:#e1e4e8'>
        <b>Arch:</b> {arch_str} | <b>Suggested LR:</b> <span style='color:#3fb950; font-weight:bold'>{lrs[mi]:.2e}</span> |
        Rule of thumb: use 10x lower = <span style='color:#58a6ff'>{lrs[mi]/10:.2e}</span></p>""")



# ════════════════════════════════════════════════════════════════
#  PAGE 25: NORMALIZATION COMPARISON
# ════════════════════════════════════════════════════════════════

class NormalizationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._run()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Activation Pattern:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Random Normal", "Skewed", "Multi-Scale", "Correlated", "Bimodal"])
        self.pattern_combo.currentTextChanged.connect(self._run)
        top_bar.addWidget(self.pattern_combo)
        top_bar.addWidget(QLabel("Depth:"))
        self.depth_spin = QSpinBox(); self.depth_spin.setRange(5, 30); self.depth_spin.setValue(12)
        self.depth_spin.valueChanged.connect(self._run)
        top_bar.addWidget(self.depth_spin)
        top_bar.addWidget(QLabel("Groups (GroupNorm):"))
        self.groups_spin = QSpinBox(); self.groups_spin.setRange(1, 8); self.groups_spin.setValue(4)
        self.groups_spin.valueChanged.connect(self._run)
        top_bar.addWidget(self.groups_spin)
        top_bar.addStretch()
        layout.addLayout(top_bar)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setMaximumHeight(70)
        layout.addWidget(self.info_text)
        self.canvas = MplCanvas(self, width=14, height=5); layout.addWidget(self.canvas, stretch=1)

    def _generate_activations(self, n=256, d=32):
        pattern = self.pattern_combo.currentText()
        if pattern == "Random Normal":
            return np.random.randn(n, d)
        elif pattern == "Skewed":
            return np.random.exponential(1.0, (n, d)) - 1.0
        elif pattern == "Multi-Scale":
            return np.random.randn(n, d) * np.exp(np.linspace(-2, 2, d))
        elif pattern == "Correlated":
            base = np.random.randn(n, 1)
            return base + 0.1 * np.random.randn(n, d)
        else:  # Bimodal
            half = n // 2
            return np.vstack([np.random.randn(half, d) - 2, np.random.randn(n - half, d) + 2])

    def _run(self):
        X = self._generate_activations()
        depth = self.depth_spin.value()
        n_groups = self.groups_spin.value()
        norms = ["none", "batch", "layer", "group"]

        # Gradient flow comparison
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(17, 7)
        axes = self.canvas.fig.subplots(2, 2)

        # --- Top left: Gradient flow ---
        self.canvas.style_axes(axes[0, 0], "Gradient Flow Through Deep Network", "Layer (backward)", "||Grad|| (log)")
        colors = {"none": "#f85149", "batch": "#58a6ff", "layer": "#3fb950", "group": "#bc8cff"}
        labels = {"none": "None (ReLU only)", "batch": "BatchNorm", "layer": "LayerNorm", "group": f"GroupNorm(g={n_groups})"}
        for norm_name in norms:
            gn = normalization_gradient_flow(norm_name, n_layers=depth, input_dim=32)
            axes[0, 0].semilogy(range(len(gn)), [max(g, 1e-15) for g in gn],
                               "o-", color=colors[norm_name], ms=2, lw=1.5, label=labels[norm_name])
        axes[0, 0].axhline(1e-6, color="#8b949e", ls=":", alpha=0.4)
        axes[0, 0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=7)
        axes[0, 0].grid(True, alpha=0.15, color="#8b949e")

        # --- Top right: Normalization effect on distribution ---
        self.canvas.style_axes(axes[0, 1], "Effect on Activation Distribution (1st feature)", "Value", "Density")
        axes[0, 1].hist(X[:, 0], bins=30, alpha=0.4, color="#f85149", label="Original", density=True)
        for norm_name in ["batch", "layer"]:
            res = compare_normalizations(X, norm_type=norm_name)
            axes[0, 1].hist(res["normalized"][:, 0], bins=30, alpha=0.4,
                           color=colors[norm_name], label=labels[norm_name], density=True)
        axes[0, 1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=7)
        axes[0, 1].grid(True, alpha=0.15, color="#8b949e")

        # --- Bottom left: Per-sample statistics ---
        self.canvas.style_axes(axes[1, 0], "Per-Sample Mean After Normalization", "Sample Index", "Mean")
        x_idx = np.arange(X.shape[0])
        for norm_name in ["batch", "layer", "instance"]:
            res = compare_normalizations(X, norm_type=norm_name)
            sm = res.get("per_sample_mean", np.zeros(X.shape[0]))
            axes[1, 0].scatter(x_idx, sm, s=3, alpha=0.5, color=colors.get(norm_name, "#bc8cff"), label=labels.get(norm_name, norm_name))
        axes[1, 0].axhline(0, color="#8b949e", ls="--", alpha=0.5)
        axes[1, 0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=7)
        axes[1, 0].grid(True, alpha=0.15, color="#8b949e")

        # --- Bottom right: Summary comparison ---
        self.canvas.style_axes(axes[1, 1], "Normalization Characteristics", "", "")
        aspects = ["Stable\nGradients", "Batch\nDependence", "Inference\nStable", "Small\nBatch OK"]
        scores = {
            "None": [1, 5, 5, 5], "BatchNorm": [5, 1, 3, 1],
            "LayerNorm": [5, 5, 5, 5], "GroupNorm": [5, 5, 5, 4],
        }
        x_pos = np.arange(len(aspects))
        width = 0.2
        for i, (name, vals) in enumerate(scores.items()):
            axes[1, 1].bar(x_pos + i * width, vals, width, color=list(colors.values())[i],
                          alpha=0.8, label=name)
        axes[1, 1].set_xticks(x_pos + 1.5 * width)
        axes[1, 1].set_xticklabels(aspects, fontsize=8, color="#e1e4e8")
        axes[1, 1].set_yticks([1, 2, 3, 4, 5])
        axes[1, 1].set_yticklabels(["Poor", "", "OK", "", "Great"], fontsize=8, color="#e1e4e8")
        axes[1, 1].set_ylim(0, 6)
        axes[1, 1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=7, loc="upper right")
        axes[1, 1].grid(True, alpha=0.15, color="#8b949e", axis="y")

        self.canvas.fig.tight_layout(); self.canvas.draw()
        self.info_text.setHtml(f"""<p style='color:#e1e4e8'>
        <b>Pattern:</b> {self.pattern_combo.currentText()} | <b>Depth:</b> {depth} | <b>Groups:</b> {n_groups}<br>
        <span style='color:#58a6ff'><b>BatchNorm:</b></span> normalizes across batch dim (per feature). Fast training, but batch-dependent.<br>
        <span style='color:#3fb950'><b>LayerNorm:</b></span> normalizes across features (per sample). Stable, great for RNNs/Transformers.<br>
        <span style='color:#bc8cff'><b>GroupNorm:</b></span> divides channels into groups. Best of both worlds for CNNs with small batches.</p>""")


# ════════════════════════════════════════════════════════════════
#  PAGE 26: CONFUSION MATRIX & METRICS
# ════════════════════════════════════════════════════════════════

class MetricsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None; self.X = None; self.y = None
        self._setup_ui(); self._load_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Dataset:"))
        self.ds_combo = QComboBox(); self.ds_combo.addItems(list(DATASETS.keys()))
        self.ds_combo.currentTextChanged.connect(self._load_data)
        top_bar.addWidget(self.ds_combo)
        top_bar.addWidget(QLabel("Architecture:"))
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["2->32->16->1", "2->64->32->16->1", "2->16->16->16->1", "2->128->64->32->1"])
        top_bar.addWidget(self.arch_combo)
        top_bar.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(200, 5000); self.epochs_spin.setValue(1000); self.epochs_spin.setSingleStep(500)
        top_bar.addWidget(self.epochs_spin)
        self.train_btn = QPushButton("Train & Evaluate"); self.train_btn.setObjectName("accent")
        self.train_btn.clicked.connect(self._train)
        top_bar.addWidget(self.train_btn); top_bar.addStretch()
        layout.addLayout(top_bar)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setMaximumHeight(60)
        layout.addWidget(self.info_text)
        self.canvas = MplCanvas(self, width=14, height=5); layout.addWidget(self.canvas, stretch=1)

    def _load_data(self):
        self.X, self.y = DATASETS[self.ds_combo.currentText()]()
        self.result = None
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(16, 5)
        ax = self.canvas.fig.add_subplot(111)
        self.canvas.style_axes(ax, "Confusion Matrix & Metrics", "", "")
        ax.text(0.5, 0.5, "Click 'Train & Evaluate'", transform=ax.transAxes, ha="center", va="center", color="#8b949e", fontsize=14)
        self.canvas.fig.tight_layout(); self.canvas.draw()

    def _train(self):
        self.info_text.setText("<p style='color:#f0883e'>Training and evaluating...</p>"); QApplication.processEvents()
        arch_str = self.arch_combo.currentText()
        layers = [int(x) for x in arch_str.split("->")]
        arch = [(layers[0], "Input")] + [(l, "ReLU") for l in layers[1:-1]] + [(layers[-1], "Sigmoid")]
        self.result = train_and_evaluate(self.X, self.y, arch=arch, epochs=self.epochs_spin.value(), lr=0.01)
        self._plot()

    def _plot(self):
        if self.result is None: return
        r = self.result; m = r["metrics"]; cm = m["cm"]; roc = r["roc"]
        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(17, 6)
        axes = self.canvas.fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1.2, 1.2]})

        # Confusion matrix
        self.canvas.style_axes(axes[0], "Confusion Matrix", "Predicted", "Actual")
        im = axes[0].imshow(cm, cmap="Blues", interpolation="nearest")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "#e1e4e8"
                axes[0].text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontsize=16, fontweight="bold")
        axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["0", "1"])
        axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["0", "1"])
        self.canvas.fig.colorbar(im, ax=axes[0], shrink=0.7)

        # ROC curve
        self.canvas.style_axes(axes[1], "ROC Curve", "False Positive Rate", "True Positive Rate")
        axes[1].plot(roc["fpr"], roc["tpr"], "-", color="#58a6ff", lw=2, label=f'AUC = {roc["auc"]:.3f}')
        axes[1].fill_between(roc["fpr"], roc["tpr"], alpha=0.15, color="#58a6ff")
        axes[1].plot([0, 1], [0, 1], "--", color="#8b949e", alpha=0.5, label="Random")
        axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8")
        axes[1].grid(True, alpha=0.15, color="#8b949e")

        # Metrics bar chart
        self.canvas.style_axes(axes[2], "Classification Metrics", "", "Score")
        metric_names = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]
        metric_vals = [m["accuracy"], m["precision"], m["recall"], m["f1"], m["specificity"]]
        bar_colors = ["#58a6ff", "#3fb950", "#f0883e", "#bc8cff", "#f85149"]
        bars = axes[2].barh(metric_names, metric_vals, color=bar_colors, alpha=0.85, height=0.6)
        axes[2].set_xlim(0, 1.1)
        for bar, val in zip(bars, metric_vals):
            axes[2].text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", color="#e1e4e8", fontsize=10, fontweight="bold")
        axes[2].grid(True, alpha=0.15, color="#8b949e", axis="x")

        self.canvas.fig.tight_layout(); self.canvas.draw()
        self.info_text.setHtml(f"""<p style='color:#e1e4e8'>
        <b>Arch:</b> {self.arch_combo.currentText()} | <b>Epochs:</b> {self.epochs_spin.value()} |
        <b>Loss:</b> {r['loss_history'][-1]:.4f} |
        <b>AUC:</b> <span style='color:#58a6ff; font-weight:bold'>{roc["auc"]:.3f}</span> |
        <b>F1:</b> <span style='color:#bc8cff; font-weight:bold'>{m["f1"]:.3f}</span><br>
        <b>TP:</b> {m["tp"]} | <b>FP:</b> {m["fp"]} | <b>FN:</b> {m["fn"]} | <b>TN:</b> {m["tn"]}
        &nbsp;|&nbsp; Precision = TP/(TP+FP) &nbsp;|&nbsp; Recall = TP/(TP+FN) &nbsp;|&nbsp; F1 = 2*P*R/(P+R)</p>""")


# ════════════════════════════════════════════════════════════════
#  PAGE 27: EMBEDDING & PCA
# ════════════════════════════════════════════════════════════════

class EmbeddingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui(); self._load_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Dataset:"))
        self.ds_combo = QComboBox(); self.ds_combo.addItems(list(DATASETS.keys()))
        self.ds_combo.currentTextChanged.connect(self._load_data)
        top_bar.addWidget(self.ds_combo)
        top_bar.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["PCA", "Raw Features", "Autoencoder Latent", "t-SNE"])
        self.method_combo.currentTextChanged.connect(self._run)
        top_bar.addWidget(self.method_combo)
        self.run_btn = QPushButton("Visualize"); self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._run)
        top_bar.addWidget(self.run_btn); top_bar.addStretch()
        layout.addLayout(top_bar)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setMaximumHeight(70)
        layout.addWidget(self.info_text)
        self.canvas = MplCanvas(self, width=14, height=5); layout.addWidget(self.canvas, stretch=1)

    def _load_data(self):
        self.X, self.y_01 = DATASETS[self.ds_combo.currentText()]()
        self._run()

    def _run(self):
        method = self.method_combo.currentText()
        self.info_text.setText(f"<p style='color:#f0883e'>Computing {method}...</p>"); QApplication.processEvents()
        X, y = self.X, self.y_01.flatten()
        c0, c1 = y == 0, y == 1

        self.canvas.fig.clear(); self.canvas.fig.set_size_inches(17, 6)

        if method == "PCA":
            pca = compute_pca(X, n_components=2)
            proj = pca["projected"]
            axes = self.canvas.fig.subplots(1, 3)

            self.canvas.style_axes(axes[0], "Original Data (2D)", "x1", "x2")
            axes[0].scatter(X[c0, 0], X[c0, 1], c="#58a6ff", s=12, alpha=0.5)
            axes[0].scatter(X[c1, 0], X[c1, 1], c="#f85149", s=12, alpha=0.5)

            self.canvas.style_axes(axes[1], "PCA Projection", "PC1", "PC2")
            axes[1].scatter(proj[c0, 0], proj[c0, 1], c="#58a6ff", s=12, alpha=0.5)
            axes[1].scatter(proj[c1, 0], proj[c1, 1], c="#f85149", s=12, alpha=0.5)
            axes[1].set_aspect("equal")

            self.canvas.style_axes(axes[2], "Explained Variance", "Component", "Ratio")
            n_show = min(len(pca["explained_var_ratio"]), X.shape[1])
            x_comp = np.arange(n_show)
            axes[2].bar(x_comp, pca["explained_var_ratio"][:n_show], color="#58a6ff", alpha=0.7, label="Per PC")
            axes[2].plot(x_comp, pca["cumulative_var"][:n_show], "o-", color="#f85149", lw=2, label="Cumulative")
            axes[2].axhline(0.95, color="#3fb950", ls="--", alpha=0.7, label="95%")
            axes[2].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
            axes[2].grid(True, alpha=0.15, color="#8b949e")

            self.info_text.setHtml(f"""<p style='color:#e1e4e8'>
            <b>PCA:</b> {X.shape[1]}D -> 2D | <b>PC1 var:</b> {pca['explained_var_ratio'][0]:.1%} |
            <b>PC2 var:</b> {pca['explained_var_ratio'][1]:.1%} |
            <b>Cumulative (2 PC):</b> {pca['cumulative_var'][1]:.1%}<br>
            <b>PCA</b> finds directions of maximum variance. Orthogonal transformation. Useful for visualization and noise reduction.</p>""")

        elif method == "Raw Features":
            axes = self.canvas.fig.subplots(1, 2)
            self.canvas.style_axes(axes[0], "Feature 0 vs 1", "Feature 0", "Feature 1")
            axes[0].scatter(X[c0, 0], X[c0, 1], c="#58a6ff", s=12, alpha=0.5, label="Class 0")
            axes[0].scatter(X[c1, 0], X[c1, 1], c="#f85149", s=12, alpha=0.5, label="Class 1")
            axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8")

            self.canvas.style_axes(axes[1], "Feature Statistics", "Feature", "Value")
            means_0 = np.mean(X[c0], axis=0); means_1 = np.mean(X[c1], axis=0)
            x_feat = np.arange(X.shape[1])
            axes[1].bar(x_feat - 0.15, means_0, 0.3, color="#58a6ff", alpha=0.8, label="Class 0 mean")
            axes[1].bar(x_feat + 0.15, means_1, 0.3, color="#f85149", alpha=0.8, label="Class 1 mean")
            axes[1].set_xticks(x_feat)
            axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8", fontsize=8)
            axes[1].grid(True, alpha=0.15, color="#8b949e")

            self.info_text.setHtml(f"""<p style='color:#e1e4e8'>
            <b>Raw Features:</b> {X.shape[1]}D | <b>Samples:</b> {X.shape[0]} |
            Mean separation shows how discriminative each feature is for the classification task.</p>""")

        elif method == "Autoencoder Latent":
            ae = Autoencoder(input_dim=2, hidden_dims=[32, 16], latent_dim=2, lr=0.005)
            for epoch in range(800):
                ae.train_step(X)
            z = ae.get_latent(X)
            axes = self.canvas.fig.subplots(1, 3)
            self.canvas.style_axes(axes[0], "Original", "x1", "x2")
            axes[0].scatter(X[c0, 0], X[c0, 1], c="#58a6ff", s=12, alpha=0.5)
            axes[0].scatter(X[c1, 0], X[c1, 1], c="#f85149", s=12, alpha=0.5)
            self.canvas.style_axes(axes[1], "Autoencoder Latent Space", "z1", "z2")
            axes[1].scatter(z[c0, 0], z[c0, 1], c="#58a6ff", s=12, alpha=0.5)
            axes[1].scatter(z[c1, 0], z[c1, 1], c="#f85149", s=12, alpha=0.5)
            X_recon = ae.forward(X)
            self.canvas.style_axes(axes[2], "Reconstruction", "x1", "x2")
            axes[2].scatter(X_recon[c0, 0], X_recon[c0, 1], c="#58a6ff", s=12, alpha=0.5)
            axes[2].scatter(X_recon[c1, 0], X_recon[c1, 1], c="#f85149", s=12, alpha=0.5)
            self.info_text.setHtml(f"""<p style='color:#e1e4e8'>
            <b>Autoencoder:</b> 2->32->16->2->16->32->2 | <b>Params:</b> {ae.count_params():,} |
            <b>MSE:</b> {ae.loss_history[-1]:.4f}<br>
            The autoencoder learns a compressed representation. The latent space often separates classes without supervision.</p>""")

        elif method == "t-SNE":
            self.info_text.setText("<p style='color:#f0883e'>Running t-SNE (may take a moment)...</p>"); QApplication.processEvents()
            result = compute_tsne_simple(X, n_components=2, n_iter=400)
            emb = result["embedding"]
            axes = self.canvas.fig.subplots(1, 2)
            self.canvas.style_axes(axes[0], "t-SNE Embedding", "Dim 1", "Dim 2")
            axes[0].scatter(emb[c0, 0], emb[c0, 1], c="#58a6ff", s=12, alpha=0.5, label="Class 0")
            axes[0].scatter(emb[c1, 0], emb[c1, 1], c="#f85149", s=12, alpha=0.5, label="Class 1")
            axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e1e4e8")
            self.canvas.style_axes(axes[1], "t-SNE Loss (KL Divergence)", "Iteration", "Loss")
            axes[1].plot(result["loss_history"], color="#bc8cff", lw=1.5)
            axes[1].grid(True, alpha=0.15, color="#8b949e")
            self.info_text.setHtml(f"""<p style='color:#e1e4e8'>
            <b>t-SNE:</b> {X.shape[1]}D -> 2D | <b>Perplexity:</b> 30 | <b>Iterations:</b> 400 |
            <b>Final KL:</b> {result['loss_history'][-1]:.2f}<br>
            <b>t-SNE</b> preserves local structure: nearby points in high-D stay nearby in 2D.
            Unlike PCA, it captures non-linear manifolds. Great for visualization, but not for feature extraction.</p>""")

        self.canvas.fig.tight_layout(); self.canvas.draw()



# ════════════════════════════════════════
#  MAIN WINDOW
# ════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Networks from Scratch - Interactive Learner")
        self.setMinimumSize(1200, 750); self.resize(1500, 900)

        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0)

        sidebar = QFrame(); sidebar.setFixedWidth(230)
        sidebar.setStyleSheet("QFrame { background-color: #161b22; border-right: 1px solid #30363d; }")
        self.sidebar_layout = QVBoxLayout(sidebar)
        self.sidebar_layout.setContentsMargins(10, 14, 10, 14)

        logo = QLabel("NN from Scratch")
        logo.setFont(QFont("Segoe UI", 15, QFont.Bold))
        logo.setStyleSheet("color: #58a6ff; padding: 6px 0 16px 0;")
        self.sidebar_layout.addWidget(logo)

        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search pages...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #0d1117; border: 1px solid #30363d;
                border-radius: 5px; padding: 6px 10px; color: #e1e4e8; font-size: 12px;
            }
            QLineEdit:focus { border-color: #58a6ff; }
        """)
        self.search_input.textChanged.connect(self._filter_pages)
        self.sidebar_layout.addWidget(self.search_input)

        # Section labels
        core_label = QLabel("CORE")
        core_label.setStyleSheet("color: #484f58; font-size: 10px; font-weight: bold; padding: 8px 0 4px 4px; letter-spacing: 1px;")
        self.sidebar_layout.addWidget(core_label)

        self.nav_buttons = []
        nav_items = [
            ("Playground", "playground", "CORE"),
            ("Forward Pass", "forward", "CORE"),
            ("Backpropagation", "backprop", "CORE"),
            ("Activations", "activations", "CORE"),
            ("Weight Init", "weight_init", "CORE"),
        ]

        arch_label = QLabel("ARCHITECTURES")
        arch_label.setStyleSheet("color: #484f58; font-size: 10px; font-weight: bold; padding: 12px 0 4px 4px; letter-spacing: 1px;")
        self.sidebar_layout.addWidget(arch_label)

        arch_items = [
            ("RNN / LSTM", "rnn", "ARCH"),
            ("CNN (Conv1D)", "cnn", "ARCH"),
            ("Attention", "attention", "ARCH"),
            ("Transformer", "transformer", "ARCH"),
            ("Kernels", "kernels", "ARCH"),
            ("Scaling Laws", "scaling", "ARCH"),
            ("Network Zoo", "zoo", "ARCH"),
            ("AdaBoost", "adaboost", "ARCH"),
            ("Autoencoder", "autoencoder", "ARCH"),
            ("GAN", "gan", "ARCH"),
            ("Diffusion", "diffusion", "ARCH"),
            ("ResNet Demo", "resnet", "ARCH"),
            ("Normalization", "normalization", "ARCH"),
        ]

        tools_label = QLabel("TOOLS")
        tools_label.setStyleSheet("color: #484f58; font-size: 10px; font-weight: bold; padding: 12px 0 4px 4px; letter-spacing: 1px;")
        self.sidebar_layout.addWidget(tools_label)

        tool_items = [
            ("Loss Landscape", "landscape", "TOOL"),
            ("Regularization", "regularization", "TOOL"),
            ("LR Schedulers", "lr_scheduler", "TOOL"),
            ("LR Finder", "lr_finder", "TOOL"),
            ("Grad Clipping", "grad_clip", "TOOL"),
            ("Vanishing Grad", "vanishing_grad", "TOOL"),
            ("Metrics / ROC", "metrics", "TOOL"),
            ("Embedding / PCA", "embedding", "TOOL"),
            ("Lessons", "lessons", "TOOL"),
        ]

        all_items = nav_items + arch_items + tool_items
        nav_btn_style = """
            QPushButton {
                text-align: left; padding: 9px 12px; font-size: 12px;
                border: none; border-radius: 5px; margin-bottom: 1px;
            }
            QPushButton:hover { background-color: #21262d; }
            QPushButton:checked { background-color: #1f6feb; color: white; font-weight: bold; }
        """
        for label, page_id, _ in all_items:
            btn = QPushButton(label); btn.setCheckable(True)
            btn.setStyleSheet(nav_btn_style)
            btn.clicked.connect(lambda checked, pid=page_id: self._navigate(pid))
            self.sidebar_layout.addWidget(btn)
            self.nav_buttons.append((btn, page_id))

        self.sidebar_layout.addStretch()
        version = QLabel("v7.0 - NumPy + PySide6")
        version.setStyleSheet("color: #484f58; font-size: 10px; padding: 6px;")
        self.sidebar_layout.addWidget(version)

        main_layout.addWidget(sidebar)

        self.pages = {}
        self.stack = QWidget(); self.stack_layout = QVBoxLayout(self.stack)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)

        self.pages["playground"] = PlaygroundPage()
        self.pages["forward"] = ForwardPassPage()
        self.pages["backprop"] = BackpropPage()
        self.pages["activations"] = ActivationPage()
        self.pages["weight_init"] = WeightInitPage()
        self.pages["rnn"] = RNNPage()
        self.pages["cnn"] = CNNPage()
        self.pages["attention"] = AttentionPage()
        self.pages["kernels"] = KernelPage()
        self.pages["scaling"] = ScalingPage()
        self.pages["zoo"] = NetworkZooPage()
        self.pages["adaboost"] = AdaBoostPage()
        self.pages["autoencoder"] = AutoencoderPage()
        self.pages["gan"] = GANPage()
        self.pages["diffusion"] = DiffusionPage()
        self.pages["transformer"] = TransformerPage()
        self.pages["resnet"] = ResNetPage()
        self.pages["landscape"] = LossLandscapePage()
        self.pages["regularization"] = RegularizationPage()
        self.pages["lr_scheduler"] = LRSchedulerPage()
        self.pages["lr_finder"] = LRFinderPage()
        self.pages["grad_clip"] = GradClipPage()
        self.pages["vanishing_grad"] = VanishingGradPage()
        self.pages["normalization"] = NormalizationPage()
        self.pages["metrics"] = MetricsPage()
        self.pages["embedding"] = EmbeddingPage()
        self.pages["lessons"] = LessonsPage()

        for page in self.pages.values():
            page.setVisible(False); self.stack_layout.addWidget(page)

        self.pages["playground"].setVisible(True)
        self.current_page = "playground"
        self.nav_buttons[0][0].setChecked(True)
        main_layout.addWidget(self.stack, stretch=1)

        self.statusBar().setStyleSheet("""
            QStatusBar { background-color: #161b22; border-top: 1px solid #30363d; color: #8b949e; font-size: 12px; }
        """)
        self.statusBar().showMessage("Ready - Explore neural networks from first principles!")

    def _filter_pages(self, text):
        text = text.lower().strip()
        # Count visible per section
        core_vis = arch_vis = tool_vis = 0
        for i, (btn, pid) in enumerate(self.nav_buttons):
            if i < 5:  # CORE
                match = text == "" or text in btn.text().lower()
                btn.setVisible(match)
                if match: core_vis += 1
            elif i < 18:  # ARCHITECTURES
                match = text == "" or text in btn.text().lower()
                btn.setVisible(match)
                if match: arch_vis += 1
            else:  # TOOLS
                match = text == "" or text in btn.text().lower()
                btn.setVisible(match)
                if match: tool_vis += 1
        # Toggle section labels
        sidebar_w = self.sidebar_layout.parentWidget()
        for child in sidebar_w.findChildren(QLabel):
            t = child.text().strip()
            if t == "CORE": child.setVisible(core_vis > 0 or text == "")
            elif t == "ARCHITECTURES": child.setVisible(arch_vis > 0 or text == "")
            elif t == "TOOLS": child.setVisible(tool_vis > 0 or text == "")

    def keyPressEvent(self, event):
        # Ctrl+1-9 for quick page navigation
        mods = event.modifiers()
        key = event.key()
        if mods & Qt.ControlModifier:
            page_keys = {
                Qt.Key_1: "playground", Qt.Key_2: "forward", Qt.Key_3: "backprop",
                Qt.Key_4: "activations", Qt.Key_5: "weight_init", Qt.Key_6: "rnn",
                Qt.Key_7: "cnn", Qt.Key_8: "attention", Qt.Key_9: "transformer",
                Qt.Key_0: "lessons",
            }
            if key in page_keys:
                self._navigate(page_keys[key])
                return
        # Ctrl+F to focus search
        if mods & Qt.ControlModifier and key == Qt.Key_F:
            self.search_input.setFocus()
            self.search_input.selectAll()
            return
        super().keyPressEvent(event)

    def _navigate(self, page_id):
        if self.current_page in self.pages:
            self.pages[self.current_page].setVisible(False)
        self.pages[page_id].setVisible(True)
        self.current_page = page_id
        for btn, pid in self.nav_buttons:
            btn.setChecked(pid == page_id)
        msgs = {
            "playground": "Playground - Train neural networks on interactive datasets",
            "forward": "Forward Pass - Step through computations layer by layer",
            "backprop": "Backpropagation - Visualize gradient flow",
            "activations": "Activation Functions - Explore non-linearities (ReLU, GELU, Swish, ELU, ...)",
            "weight_init": "Weight Initialization - Compare He, Xavier, Random init",
            "rnn": "RNN / LSTM - Train recurrent networks on sequence data",
            "cnn": "CNN (Conv1D) - Train convolutional networks, visualize filters",
            "attention": "Attention - Understand self-attention, Q/K/V, and multi-head attention",
            "kernels": "Kernels - Explore 2D convolution kernels and feature maps",
            "scaling": "Scaling Laws - How depth, width, and data affect performance",
            "zoo": "Network Zoo - Compare different architectures side by side",
            "adaboost": "AdaBoost - Understand adaptive boosting with decision stumps",
            "autoencoder": "Autoencoder - Learn compression and latent space representation",
            "gan": "GAN - Generator vs Discriminator adversarial training",
            "diffusion": "Diffusion - Learn denoising diffusion probabilistic models (DDPM)",
            "transformer": "Transformer - Multi-head attention, FFN, LayerNorm, Residual connections",
            "resnet": "ResNet Demo - Plain vs Residual network comparison",
            "landscape": "Loss Landscape - Visualize the 2D loss surface",
            "regularization": "Regularization - Compare None, L2, Dropout, L2+Dropout",
            "lr_scheduler": "LR Schedulers - Compare Constant, Step, Exp, Cosine, Warmup",
            "lr_finder": "LR Finder - Find the optimal learning rate using range test",
            "grad_clip": "Gradient Clipping - Prevent exploding gradients",
            "vanishing_grad": "Vanishing Gradient - Deep dive into gradient flow problems and solutions",
            "normalization": "Normalization - Compare BatchNorm, LayerNorm, InstanceNorm, GroupNorm",
            "metrics": "Metrics / ROC - Confusion matrix, precision, recall, F1, AUC-ROC curve",
            "embedding": "Embedding / PCA - Dimensionality reduction and latent space visualization",
            "lessons": "Lessons - 10 chapters covering fundamentals to advanced topics",
        }
        self.statusBar().showMessage(msgs.get(page_id, ""))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion"); app.setStyleSheet(DARK_STYLE)
    window = MainWindow(); window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()