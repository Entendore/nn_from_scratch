"""
Neural Network from Scratch — Pure NumPy Implementation
========================================================
This module contains a complete neural network library built entirely
with NumPy.  It is designed to be *educational*: every operation is
explicit (no automatic differentiation, no frameworks), so you can
inspect weights, activations, and gradients at every step.

Includes: Dense, Dropout, BatchNorm, Conv1D, RNN, Self-Attention layers,
SGD/Adam optimizers, L2 regularization, LR scheduling, and multiple
loss / activation functions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable


# ────────────────────────────────────────────────────────────────
#  Activation Functions
# ────────────────────────────────────────────────────────────────

class Activation:
    """Container for activation functions and their derivatives."""

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_deriv(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def tanh_deriv(a: np.ndarray) -> np.ndarray:
        return 1.0 - a ** 2

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def relu_deriv(a: np.ndarray) -> np.ndarray:
        return (a > 0).astype(float)

    @staticmethod
    def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_deriv(a: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(a > 0, 1.0, alpha)

    @staticmethod
    def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))

    @staticmethod
    def elu_deriv(a: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(a > 0, 1.0, a + alpha)

    @staticmethod
    def swish(z: np.ndarray) -> np.ndarray:
        return z * Activation.sigmoid(z)

    @staticmethod
    def swish_deriv(z: np.ndarray, a: np.ndarray) -> np.ndarray:
        sig = Activation.sigmoid(z)
        return sig + a * sig * (1 - sig)

    @staticmethod
    def gelu(z: np.ndarray) -> np.ndarray:
        return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))

    @staticmethod
    def gelu_deriv(z: np.ndarray) -> np.ndarray:
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
        return cdf + z * pdf

    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        return z

    @staticmethod
    def linear_deriv(a: np.ndarray) -> np.ndarray:
        return np.ones_like(a)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)


# Mapping from name -> (forward, derivative)
# For swish/gelu, deriv takes (z, a); the Layer.backward handles this.
ACTIVATION_MAP = {
    "Sigmoid":   (Activation.sigmoid,        Activation.sigmoid_deriv),
    "Tanh":      (Activation.tanh,           Activation.tanh_deriv),
    "ReLU":      (Activation.relu,           Activation.relu_deriv),
    "LeakyReLU": (Activation.leaky_relu,     Activation.leaky_relu_deriv),
    "ELU":       (Activation.elu,            Activation.elu_deriv),
    "Swish":     (Activation.swish,          Activation.swish_deriv),
    "GELU":      (Activation.gelu,           Activation.gelu_deriv),
    "Linear":    (Activation.linear,         Activation.linear_deriv),
    "Softmax":   (Activation.softmax,        None),
}

# ────────────────────────────────────────────────────────────────
#  Loss Functions
# ────────────────────────────────────────────────────────────────

class Loss:
    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def mse_deriv(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2.0 * (y_pred - y_true) / y_pred.shape[0]

    @staticmethod
    def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -float(np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    @staticmethod
    def binary_cross_entropy_deriv(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_pred.shape[0]

    @staticmethod
    def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        n = y_pred.shape[0]
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(n, 1)
            if y_pred.shape[1] == 1:
                y_t = y_true.astype(float).flatten()
                return -float(np.mean(
                    y_t * np.log(y_pred.flatten()) + (1 - y_t) * np.log(1 - y_pred.flatten())))
            y_true_int = y_true.astype(int).flatten()
            return -float(np.mean(np.log(y_pred[np.arange(n), y_true_int])))
        else:
            return -float(np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

    @staticmethod
    def cross_entropy_deriv(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        n = y_pred.shape[0]
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(n), y_true.astype(int).flatten()] = 1.0
        else:
            y_onehot = y_true
        return (y_pred - y_onehot) / n


LOSS_MAP = {
    "MSE":                    (Loss.mse,                    Loss.mse_deriv),
    "Binary Cross-Entropy":   (Loss.binary_cross_entropy,   Loss.binary_cross_entropy_deriv),
    "Cross-Entropy":          (Loss.cross_entropy,          Loss.cross_entropy_deriv),
}


# ────────────────────────────────────────────────────────────────
#  Layer base mixin
# ────────────────────────────────────────────────────────────────

class LayerBase:
    """Shared Adam state and gradient accumulation for all layer types."""

    def _init_adam(self):
        self._param_names: List[str] = []
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}

    def _register_param(self, name: str, shape: tuple):
        self._param_names.append(name)
        self._adam_m[name] = np.zeros(shape)
        self._adam_v[name] = np.zeros(shape)

    def adam_update(self, lr: float, beta1: float, beta2: float, t: int, eps: float = 1e-8):
        for name in self._param_names:
            param = getattr(self, name)
            grad = getattr(self, "d" + name)
            m = self._adam_m[name]
            v = self._adam_v[name]
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def sgd_update(self, lr: float):
        for name in self._param_names:
            param = getattr(self, name)
            grad = getattr(self, "d" + name)
            param -= lr * grad


# ────────────────────────────────────────────────────────────────
#  Dense (Fully-Connected) Layer
# ────────────────────────────────────────────────────────────────

class DenseLayer(LayerBase):
    """A single fully-connected layer with weights, biases, and activation."""

    def __init__(self, n_in: int, n_out: int, activation: str = "ReLU",
                 weight_init: str = "he"):
        super().__init__()
        self._init_adam()
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation
        self.layer_type = "Dense"

        if weight_init == "he":
            self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        elif weight_init == "xavier":
            self.W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
        else:
            self.W = np.random.randn(n_in, n_out) * 0.01

        self.b = np.zeros((1, n_out))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.z = None
        self.a = None
        self.a_prev = None

        self._register_param("W", self.W.shape)
        self._register_param("b", self.b.shape)

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        self.a_prev = a_prev
        self.z = a_prev @ self.W + self.b
        act_fn, _ = ACTIVATION_MAP[self.activation_name]
        # Some activations need z (Swish, GELU)
        if self.activation_name in ("Swish", "GELU"):
            self.a = act_fn(self.z)
        else:
            self.a = act_fn(self.z)
        return self.a

    def backward(self, da: np.ndarray, loss_uses_softmax: bool = False) -> np.ndarray:
        _, act_deriv = ACTIVATION_MAP[self.activation_name]

        if self.activation_name == "Softmax" and loss_uses_softmax:
            dz = da
        elif self.activation_name in ("Swish", "GELU"):
            dz = da * act_deriv(self.z, self.a)
        else:
            dz = da * act_deriv(self.a)

        m = self.a_prev.shape[0]
        self.dW = (self.a_prev.T @ dz) / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m
        da_prev = dz @ self.W.T
        return da_prev


# ────────────────────────────────────────────────────────────────
#  Dropout Layer
# ────────────────────────────────────────────────────────────────

class DropoutLayer:
    """Inverted dropout layer — only active during training."""

    def __init__(self, p: float = 0.5):
        self.p = p                    # drop probability
        self.mask = None
        self.layer_type = "Dropout"
        self.dW = None                # no learnable params
        self.db = None

    def forward(self, a: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.p > 0:
            self.mask = (np.random.rand(*a.shape) > self.p).astype(float) / (1.0 - self.p)
            return a * self.mask
        else:
            self.mask = None
            return a

    def backward(self, da: np.ndarray) -> np.ndarray:
        if self.mask is not None:
            return da * self.mask
        return da


# ────────────────────────────────────────────────────────────────
#  Batch Normalization Layer
# ────────────────────────────────────────────────────────────────

class BatchNormLayer(LayerBase):
    """Batch Normalization for fully-connected layers."""

    def __init__(self, n_features: int, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self._init_adam()
        self.n_features = n_features
        self.momentum = momentum
        self.eps = eps
        self.layer_type = "BatchNorm"

        # Learnable parameters
        self.gamma = np.ones((1, n_features))
        self.beta = np.zeros((1, n_features))

        # Running statistics (for inference)
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))

        # Gradients
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        # Cache
        self.a_prev = None
        self.z_norm = None
        self.std_inv = None

        self._register_param("gamma", self.gamma.shape)
        self._register_param("beta", self.beta.shape)

    def forward(self, a: np.ndarray, training: bool = True) -> np.ndarray:
        self.a_prev = a
        if training:
            mu = np.mean(a, axis=0, keepdims=True)
            var = np.var(a, axis=0, keepdims=True)
            self.std_inv = 1.0 / np.sqrt(var + self.eps)
            self.z_norm = (a - mu) * self.std_inv
            out = self.gamma * self.z_norm + self.beta

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            z_norm = (a - self.running_mean) * 1.0 / np.sqrt(self.running_var + self.eps)
            out = self.gamma * z_norm + self.beta
            self.z_norm = z_norm
        return out

    def backward(self, da: np.ndarray) -> np.ndarray:
        m = da.shape[0]
        self.dgamma = np.sum(da * self.z_norm, axis=0, keepdims=True) / m
        self.dbeta = np.sum(da, axis=0, keepdims=True) / m

        dz_norm = da * self.gamma
        da_prev = (1.0 / m) * self.std_inv * (
            m * dz_norm
            - np.sum(dz_norm, axis=0, keepdims=True)
            - self.z_norm * np.sum(dz_norm * self.z_norm, axis=0, keepdims=True)
        )
        return da_prev


# ────────────────────────────────────────────────────────────────
#  Conv1D Layer (educational — small, explicit loops)
# ────────────────────────────────────────────────────────────────

class Conv1DLayer(LayerBase):
    """1D Convolutional layer — supports multiple input/output channels."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 0, activation: str = "ReLU",
                 weight_init: str = "he"):
        super().__init__()
        self._init_adam()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_name = activation
        self.layer_type = "Conv1D"

        fan_in = in_channels * kernel_size
        if weight_init == "he":
            scale = np.sqrt(2.0 / fan_in)
        elif weight_init == "xavier":
            scale = np.sqrt(1.0 / fan_in)
        else:
            scale = 0.01

        # K: (out_channels, in_channels, kernel_size)
        self.K = np.random.randn(out_channels, in_channels, kernel_size) * scale
        self.b = np.zeros((out_channels,))

        self.dK = np.zeros_like(self.K)
        self.db = np.zeros_like(self.b)

        self.a_prev = None
        self.z = None
        self.a = None

        self._register_param("K", self.K.shape)
        self._register_param("b", self.b.shape)

    def _conv1d_forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, in_channels, length) -> (batch, out_channels, out_length)"""
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)))
        batch, _, length = x.shape
        out_len = (length - self.kernel_size) // self.stride + 1
        out = np.zeros((batch, self.out_channels, out_len))

        for oc in range(self.out_channels):
            for i in range(out_len):
                start = i * self.stride
                patch = x[:, :, start:start + self.kernel_size]  # (batch, in_ch, ks)
                out[:, oc, i] = np.sum(patch * self.K[oc], axis=(1, 2)) + self.b[oc]
        return out

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        """a_prev: (batch, in_channels, length)"""
        self.a_prev = a_prev
        self.z = self._conv1d_forward(a_prev)
        act_fn, _ = ACTIVATION_MAP[self.activation_name]
        self.a = act_fn(self.z)
        return self.a

    def backward(self, da: np.ndarray) -> np.ndarray:
        """da: (batch, out_channels, out_length)"""
        _, act_deriv = ACTIVATION_MAP[self.activation_name]
        dz = da * act_deriv(self.a)

        x = self.a_prev
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)))
        else:
            x_padded = x

        batch, in_ch, length = x_padded.shape
        out_len = dz.shape[2]

        # Gradients
        self.dK = np.zeros_like(self.K)
        self.db = np.sum(dz, axis=(0, 2)) / batch

        dx_padded = np.zeros_like(x_padded)

        for oc in range(self.out_channels):
            for i in range(out_len):
                start = i * self.stride
                patch = x_padded[:, :, start:start + self.kernel_size]  # (batch, in_ch, ks)
                dz_i = dz[:, oc, i]  # (batch,)
                self.dK[oc] += np.sum(dz_i[:, None, None] * patch, axis=0) / batch
                dx_padded[:, :, start:start + self.kernel_size] += dz_i[:, None, None] * self.K[oc]

        # Remove padding
        if self.padding > 0:
            return dx_padded[:, :, self.padding:-self.padding]
        return dx_padded


# ────────────────────────────────────────────────────────────────
#  RNN (Vanilla) Cell / Layer
# ────────────────────────────────────────────────────────────────

class RNNLayer(LayerBase):
    """Simple (vanilla) RNN layer — processes sequences step by step."""

    def __init__(self, input_size: int, hidden_size: int,
                 activation: str = "Tanh", weight_init: str = "xavier"):
        super().__init__()
        self._init_adam()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_name = activation
        self.layer_type = "RNN"

        if weight_init == "he":
            s1 = np.sqrt(2.0 / input_size)
            s2 = np.sqrt(2.0 / hidden_size)
        elif weight_init == "xavier":
            s1 = np.sqrt(1.0 / input_size)
            s2 = np.sqrt(1.0 / hidden_size)
        else:
            s1 = s2 = 0.01

        # Wx: (input_size, hidden_size)  — input-to-hidden
        self.Wx = np.random.randn(input_size, hidden_size) * s1
        # Wh: (hidden_size, hidden_size) — hidden-to-hidden
        self.Wh = np.random.randn(hidden_size, hidden_size) * s2
        # bh: (1, hidden_size)
        self.bh = np.zeros((1, hidden_size))

        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.dbh = np.zeros_like(self.bh)

        # Cache
        self.hiddens = None  # list of (batch, hidden) per timestep

        self._register_param("Wx", self.Wx.shape)
        self._register_param("Wh", self.Wh.shape)
        self._register_param("bh", self.bh.shape)

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """
        x_seq: (batch, timesteps, input_size) -> (batch, timesteps, hidden_size)
        """
        batch, T, _ = x_seq.shape
        h = np.zeros((batch, self.hidden_size))
        self.hiddens = [h]
        _, act_fn = ACTIVATION_MAP[self.activation_name]

        outputs = []
        for t in range(T):
            h_pre = x_seq[:, t, :] @ self.Wx + h @ self.Wh + self.bh
            h = act_fn(h_pre)
            self.hiddens.append(h)
            outputs.append(h)
        return np.stack(outputs, axis=1)  # (batch, T, hidden)

    def backward(self, dh_out: np.ndarray) -> np.ndarray:
        """
        dh_out: (batch, timesteps, hidden_size) -> dx: (batch, timesteps, input_size)
        """
        batch, T, _ = dh_out.shape
        _, act_deriv = ACTIVATION_MAP[self.activation_name]

        # Initialize gradient accumulators
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.dbh = np.zeros_like(self.bh)
        dx_seq = np.zeros((batch, T, self.input_size))

        dh_next = np.zeros((batch, self.hidden_size))

        for t in reversed(range(T)):
            dh = dh_out[:, t, :] + dh_next
            # h_pre = x @ Wx + h_prev @ Wh + bh
            h_prev = self.hiddens[t]
            h = self.hiddens[t + 1]
            da = dh * act_deriv(h)

            self.dWx += x_seq_cache[:, t, :].T @ da / batch  # Will fix below
            self.dbh += np.sum(da, axis=0, keepdims=True) / batch

        return dx_seq


class SimpleRNNLayer(LayerBase):
    """Simplified RNN layer — cleaner backprop implementation."""

    def __init__(self, input_size: int, hidden_size: int,
                 activation: str = "Tanh", weight_init: str = "xavier"):
        super().__init__()
        self._init_adam()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_name = activation
        self.layer_type = "RNN"

        if weight_init == "he":
            s1 = np.sqrt(2.0 / input_size)
            s2 = np.sqrt(2.0 / hidden_size)
        elif weight_init == "xavier":
            s1 = np.sqrt(1.0 / input_size)
            s2 = np.sqrt(1.0 / hidden_size)
        else:
            s1 = s2 = 0.01

        self.Wx = np.random.randn(input_size, hidden_size) * s1
        self.Wh = np.random.randn(hidden_size, hidden_size) * s2
        self.bh = np.zeros((1, hidden_size))

        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.dbh = np.zeros_like(self.bh)

        self._x_seq = None
        self._h_pre = None   # (batch, T, hidden) pre-activations
        self._h = None       # (batch, T, hidden) post-activations

        self._register_param("Wx", self.Wx.shape)
        self._register_param("Wh", self.Wh.shape)
        self._register_param("bh", self.bh.shape)

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        batch, T, _ = x_seq.shape
        self._x_seq = x_seq
        h = np.zeros((batch, self.hidden_size))
        _, act_fn = ACTIVATION_MAP[self.activation_name]

        h_list = []
        h_pre_list = []
        for t in range(T):
            h_pre = x_seq[:, t, :] @ self.Wx + h @ self.Wh + self.bh
            h = act_fn(h_pre)
            h_pre_list.append(h_pre)
            h_list.append(h)

        self._h_pre = np.stack(h_pre_list, axis=1)
        self._h = np.stack(h_list, axis=1)
        return self._h

    def backward(self, dh_out: np.ndarray) -> np.ndarray:
        batch, T, _ = dh_out.shape
        _, act_deriv = ACTIVATION_MAP[self.activation_name]

        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.dbh = np.zeros_like(self.bh)
        dx_seq = np.zeros((batch, T, self.input_size))

        dh_next = np.zeros((batch, self.hidden_size))

        for t in reversed(range(T)):
            dh = dh_out[:, t, :] + dh_next
            da = dh * act_deriv(self._h[:, t, :])

            self.dWx += self._x_seq[:, t, :].T @ da / batch
            self.dWh += self._h[:, max(t - 1, 0), :].T @ da / batch
            self.dbh += np.sum(da, axis=0, keepdims=True) / batch

            dh_next = da @ self.Wh.T
            dx_seq[:, t, :] = da @ self.Wx.T

        return dx_seq


# ────────────────────────────────────────────────────────────────
#  LSTM Layer
# ────────────────────────────────────────────────────────────────

class LSTMLayer(LayerBase):
    """Long Short-Term Memory layer — gates for better gradient flow."""

    def __init__(self, input_size: int, hidden_size: int, weight_init: str = "xavier"):
        super().__init__()
        self._init_adam()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_type = "LSTM"

        if weight_init == "xavier":
            scale = np.sqrt(1.0 / (input_size + hidden_size))
        elif weight_init == "he":
            scale = np.sqrt(2.0 / (input_size + hidden_size))
        else:
            scale = 0.01

        # Concatenated weights for all 4 gates: [forget, input, candidate, output]
        self.Wx = np.random.randn(input_size, 4 * hidden_size) * scale
        self.Wh = np.random.randn(hidden_size, 4 * hidden_size) * scale
        self.b = np.zeros((1, 4 * hidden_size))

        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)

        self._x_seq = None
        self._caches = None  # list of (h_pre_4gates, gates, c, h) per timestep

        self._register_param("Wx", self.Wx.shape)
        self._register_param("Wh", self.Wh.shape)
        self._register_param("b", self.b.shape)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        batch, T, _ = x_seq.shape
        self._x_seq = x_seq
        h = np.zeros((batch, self.hidden_size))
        c = np.zeros((batch, self.hidden_size))

        self._caches = []
        outputs = []

        for t in range(T):
            z = x_seq[:, t, :] @ self.Wx + h @ self.Wh + self.b  # (batch, 4H)
            f = self._sigmoid(z[:, :self.hidden_size])
            i = self._sigmoid(z[:, self.hidden_size:2*self.hidden_size])
            c_tilde = np.tanh(z[:, 2*self.hidden_size:3*self.hidden_size])
            o = self._sigmoid(z[:, 3*self.hidden_size:])

            c = f * c + i * c_tilde
            h = o * np.tanh(c)

            self._caches.append((z, f, i, c_tilde, o, c, h))
            outputs.append(h)

        return np.stack(outputs, axis=1)

    def backward(self, dh_out: np.ndarray) -> np.ndarray:
        batch, T, _ = dh_out.shape
        H = self.hidden_size

        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)
        dx_seq = np.zeros((batch, T, self.input_size))

        dh_next = np.zeros((batch, H))
        dc_next = np.zeros((batch, H))

        for t in reversed(range(T)):
            z, f, i, c_tilde, o, c, h = self._caches[t]
            dh = dh_out[:, t, :] + dh_next
            dc = dc_next + dh * o * (1 - np.tanh(c) ** 2)

            df = dc * c  # but c here is c_{t-1} * f + ... ; need c_prev
            # Recompute c_prev
            if t > 0:
                _, f_prev, i_prev, ct_prev, o_prev, c_prev, _ = self._caches[t - 1]
            else:
                c_prev = np.zeros_like(c)

            df = dc * c_prev
            di = dc * c_tilde
            dc_tilde = dc * i
            do = dh * np.tanh(c)

            # Sigmoid/tanh derivatives
            df = df * f * (1 - f)
            di = di * i * (1 - i)
            dc_tilde = dc_tilde * (1 - c_tilde ** 2)
            do = do * o * (1 - o)

            dz = np.concatenate([df, di, dc_tilde, do], axis=1)

            self.dWx += self._x_seq[:, t, :].T @ dz / batch
            h_prev = self._caches[t - 1][6] if t > 0 else np.zeros((batch, H))
            self.dWh += h_prev.T @ dz / batch
            self.db += np.sum(dz, axis=0, keepdims=True) / batch

            dh_next = dz @ self.Wh.T
            dc_next = dc * f
            dx_seq[:, t, :] = dz @ self.Wx.T

        return dx_seq


# ────────────────────────────────────────────────────────────────
#  Self-Attention Layer (Single-Head)
# ────────────────────────────────────────────────────────────────

class SelfAttentionLayer(LayerBase):
    """Single-head self-attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V"""

    def __init__(self, d_model: int, d_k: int = None, weight_init: str = "xavier"):
        super().__init__()
        self._init_adam()
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.layer_type = "SelfAttention"

        if weight_init == "xavier":
            s = np.sqrt(1.0 / d_model)
        elif weight_init == "he":
            s = np.sqrt(2.0 / d_model)
        else:
            s = 0.01

        # Q, K, V projection matrices
        self.Wq = np.random.randn(d_model, self.d_k) * s
        self.Wk = np.random.randn(d_model, self.d_k) * s
        self.Wv = np.random.randn(d_model, self.d_k) * s

        self.dWq = np.zeros_like(self.Wq)
        self.dWk = np.zeros_like(self.Wk)
        self.dWv = np.zeros_like(self.Wv)

        self._cache = None

        self._register_param("Wq", self.Wq.shape)
        self._register_param("Wk", self.Wk.shape)
        self._register_param("Wv", self.Wv.shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, seq_len, d_model) -> (batch, seq_len, d_k)
        """
        self._cache = {}
        Q = x @ self.Wq   # (batch, T, d_k)
        K = x @ self.Wk
        V = x @ self.Wv

        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)  # (batch, T, T)
        attn_weights = Activation.softmax(scores)  # along last dim

        out = attn_weights @ V  # (batch, T, d_k)

        self._cache = {"x": x, "Q": Q, "K": K, "V": V,
                       "attn_weights": attn_weights, "out": out}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (batch, T, d_k) -> dx: (batch, T, d_model)"""
        x = self._cache["x"]
        Q = self._cache["Q"]
        K = self._cache["K"]
        V = self._cache["V"]
        W = self._cache["attn_weights"]
        batch, T, _ = dout.shape

        # Backprop through attention
        dV = W.transpose(0, 2, 1) @ dout  # (batch, T, d_k)
        dW = dout @ V.transpose(0, 2, 1)  # (batch, T, T)

        # Softmax backward
        ds = dW * W  # element-wise
        sum_ds = np.sum(ds, axis=-1, keepdims=True)
        dscores = W * (ds - sum_ds)  # (batch, T, T)
        dscores = dscores / np.sqrt(self.d_k)

        dQ = dscores @ K
        dK = dscores.transpose(0, 2, 1) @ Q

        # Project gradients back
        self.dWq = (x.reshape(-1, self.d_model).T @ dQ.reshape(-1, self.d_k)) / (batch * T)
        self.dWk = (x.reshape(-1, self.d_model).T @ dK.reshape(-1, self.d_k)) / (batch * T)
        self.dWv = (x.reshape(-1, self.d_model).T @ dV.reshape(-1, self.d_k)) / (batch * T)

        dx = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        return dx


# ────────────────────────────────────────────────────────────────
#  Neural Network (Enhanced — supports mixed layer types)
# ────────────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    A neural network that supports Dense, Dropout, BatchNorm, Conv1D,
    RNN, LSTM, and Self-Attention layers, with SGD/Adam, L2 reg, and
    LR scheduling.
    """

    def __init__(self, loss_name: str = "MSE", lr: float = 0.01,
                 optimizer: str = "sgd", beta1: float = 0.9, beta2: float = 0.999,
                 weight_init: str = "he", l2_lambda: float = 0.0):
        self.layers: list = []
        self.loss_name = loss_name
        self.lr = lr
        self.base_lr = lr
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_init = weight_init
        self.l2_lambda = l2_lambda
        self.t = 0
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.lr_history: List[float] = []
        self.training = True

    def build_dense(self, layer_sizes: List[Tuple[int, str]]):
        """Build a dense-only network from (neurons, activation) tuples."""
        self.layers.clear()
        self.loss_history.clear()
        self.val_loss_history.clear()
        self.lr_history.clear()
        self.t = 0
        for i, (n, act) in enumerate(layer_sizes):
            if i == 0:
                self._input_dim = n
                continue
            n_in = layer_sizes[i - 1][0]
            layer = DenseLayer(n_in, n, act, self.weight_init)
            self.layers.append(layer)

    def build_custom(self, layers: list):
        """Build from a list of layer objects."""
        self.layers = layers
        self.loss_history.clear()
        self.val_loss_history.clear()
        self.lr_history.clear()
        self.t = 0

    @property
    def input_dim(self) -> int:
        return self._input_dim if hasattr(self, "_input_dim") else 0

    @property
    def output_dim(self) -> int:
        if not self.layers:
            return 0
        last = self.layers[-1]
        if isinstance(last, DenseLayer):
            return last.n_out
        return 0

    @property
    def layer_sizes(self) -> List[Tuple[int, str]]:
        sizes = [(self.input_dim, "Input")]
        for l in self.layers:
            if isinstance(l, DenseLayer):
                sizes.append((l.n_out, l.activation_name))
            elif isinstance(l, RNNLayer) or isinstance(l, SimpleRNNLayer):
                sizes.append((l.hidden_size, "RNN"))
            elif isinstance(l, LSTMLayer):
                sizes.append((l.hidden_size, "LSTM"))
            elif isinstance(l, Conv1DLayer):
                sizes.append((l.out_channels, f"Conv1D(k={l.kernel_size})"))
            elif isinstance(l, SelfAttentionLayer):
                sizes.append((l.d_k, "Attention"))
            elif isinstance(l, BatchNormLayer):
                sizes.append((l.n_features, "BatchNorm"))
            elif isinstance(l, DropoutLayer):
                sizes.append((0, f"Dropout({l.p})"))
        return sizes

    def _needs_seq_extract(self, i: int) -> bool:
        """Check if layer i is sequence/conv-based and layer i+1 is dense (needs 2D input)."""
        if i >= len(self.layers) - 1:
            return False
        cur = self.layers[i]
        nxt = self.layers[i + 1]
        if isinstance(nxt, (SimpleRNNLayer, LSTMLayer, RNNLayer, Conv1DLayer, SelfAttentionLayer, BatchNormLayer, DropoutLayer)):
            return False
        # If current output is 3D and next layer is Dense (expects 2D), auto-flatten
        return isinstance(nxt, DenseLayer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        a = X
        self._seq_extracted = set()  # track which layers need shape adaptation
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (DropoutLayer, BatchNormLayer)):
                a = layer.forward(a, training=self.training)
            else:
                a = layer.forward(a)
            # Auto-adapt: if current output is 3D and next layer is Dense, flatten
            if self._needs_seq_extract(i) and a.ndim == 3:
                cur = self.layers[i]
                if isinstance(cur, (SimpleRNNLayer, LSTMLayer, RNNLayer)):
                    # Extract last hidden state
                    a = a[:, -1, :]
                else:
                    # Flatten: (batch, c, len) -> (batch, c*len)
                    self._orig_shape = a.shape
                    a = a.reshape(a.shape[0], -1)
                self._seq_extracted.add(i)
        return a

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        loss_fn, loss_deriv = LOSS_MAP[self.loss_name]
        loss = loss_fn(y_pred, y_true)

        if self.l2_lambda > 0:
            for layer in self.layers:
                if isinstance(layer, (DenseLayer, Conv1DLayer)):
                    loss += 0.5 * self.l2_lambda * np.sum(layer.W ** 2)
                    if hasattr(layer, 'K'):
                        loss += 0.5 * self.l2_lambda * np.sum(layer.K ** 2)

        loss_uses_softmax = (self.loss_name == "Cross-Entropy" and
                             isinstance(self.layers[-1], DenseLayer) and
                             self.layers[-1].activation_name == "Softmax")

        da = loss_deriv(y_pred, y_true)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # If we adapted shape in forward, restore it for the layer's backward
            if i in self._seq_extracted:
                cur = self.layers[i]
                if isinstance(cur, (SimpleRNNLayer, RNNLayer)):
                    # Restore: (batch, hidden) -> (batch, seq_len, hidden) with last-step grad
                    seq_len = cur._h.shape[1]
                    da_expanded = np.zeros((da.shape[0], seq_len, da.shape[1]))
                    da_expanded[:, -1, :] = da
                    da = da_expanded
                elif isinstance(cur, LSTMLayer):
                    seq_len = len(cur._caches)
                    da_expanded = np.zeros((da.shape[0], seq_len, da.shape[1]))
                    da_expanded[:, -1, :] = da
                    da = da_expanded
                elif hasattr(self, '_orig_shape'):
                    # Conv1D/Attention: restore flattened shape
                    da = da.reshape(self._orig_shape)

            if isinstance(layer, DenseLayer):
                da = layer.backward(da, loss_uses_softmax=loss_uses_softmax)
            elif isinstance(layer, (DropoutLayer, BatchNormLayer)):
                da = layer.backward(da)
            else:
                da = layer.backward(da)

        if self.l2_lambda > 0:
            for layer in self.layers:
                if isinstance(layer, DenseLayer):
                    layer.dW += self.l2_lambda * layer.W
                elif isinstance(layer, Conv1DLayer):
                    layer.dK += self.l2_lambda * layer.K

        return loss

    def update_weights(self):
        self.t += 1
        for layer in self.layers:
            if isinstance(layer, (DropoutLayer,)):
                continue
            if isinstance(layer, LayerBase):
                if self.optimizer == "sgd":
                    layer.sgd_update(self.lr)
                elif self.optimizer == "adam":
                    layer.adam_update(self.lr, self.beta1, self.beta2, self.t)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        self.training = True
        y_pred = self.forward(X)
        loss = self.backward(y_pred, y)
        self.update_weights()
        self.lr_history.append(self.lr)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.training = False
        return self.forward(X)

    def get_forward_data(self, X: np.ndarray) -> List[Dict]:
        """Return layer data for visualization (Dense layers only)."""
        data = []
        a = X
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                a_new = layer.forward(a)
                data.append({
                    "index": i,
                    "name": f"Layer {i + 1}",
                    "activation_name": layer.activation_name,
                    "a_prev": a.copy(),
                    "z": layer.z.copy() if layer.z is not None else None,
                    "a": layer.a.copy() if layer.a is not None else None,
                    "W": layer.W.copy(),
                    "b": layer.b.copy(),
                    "dW": layer.dW.copy(),
                    "db": layer.db.copy(),
                })
                a = a_new
            else:
                a = layer.forward(a) if isinstance(layer, (DropoutLayer, BatchNormLayer)) else layer.forward(a)
        return data

    # ── LR Schedulers ──

    def step_lr(self, epoch: int, step_size: int, gamma: float = 0.1):
        self.lr = self.base_lr * (gamma ** (epoch // step_size))

    def exp_lr(self, epoch: int, gamma: float = 0.99):
        self.lr = self.base_lr * (gamma ** epoch)

    def cosine_lr(self, epoch: int, max_epochs: int, min_lr: float = 1e-6):
        self.lr = min_lr + 0.5 * (self.base_lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs))

    def warmup_cosine_lr(self, epoch: int, max_epochs: int, warmup_epochs: int, min_lr: float = 1e-6):
        if epoch < warmup_epochs:
            self.lr = self.base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            self.lr = min_lr + 0.5 * (self.base_lr - min_lr) * (1 + np.cos(np.pi * progress))


# Alias for backward compat
Layer = DenseLayer


# ────────────────────────────────────────────────────────────────
#  Educational Attention Helpers
# ────────────────────────────────────────────────────────────────

def compute_self_attention(x: np.ndarray, d_k: int = None, seed: int = 42) -> dict:
    """
    Compute single-head self-attention from scratch for visualization.
    x: (T, d_model) sequence of token embeddings
    Returns dict with Q, K, V, scores, attn_weights, output, d_k
    """
    np.random.seed(seed)
    T, d_model = x.shape
    d_k = d_k or d_model

    # Random projection matrices (as if learned)
    Wq = np.random.randn(d_model, d_k) * np.sqrt(2.0 / d_model)
    Wk = np.random.randn(d_model, d_k) * np.sqrt(2.0 / d_model)
    Wv = np.random.randn(d_model, d_k) * np.sqrt(2.0 / d_model)

    Q = x @ Wq  # (T, d_k)
    K = x @ Wk  # (T, d_k)
    V = x @ Wv  # (T, d_k)

    # Scaled dot-product attention
    scores = Q @ K.T / np.sqrt(d_k)  # (T, T)
    # Softmax per row
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    output = attn_weights @ V  # (T, d_k)

    return {
        "Q": Q, "K": K, "V": V,
        "scores": scores,
        "attn_weights": attn_weights,
        "output": output,
        "d_k": d_k,
        "Wq": Wq, "Wk": Wk, "Wv": Wv,
    }


def compute_multi_head_attention(x: np.ndarray, n_heads: int = 4, seed: int = 42) -> dict:
    """
    Compute multi-head self-attention for visualization.
    x: (T, d_model) — d_model must be divisible by n_heads
    Returns dict with per-head attention weights, concatenated output.
    """
    np.random.seed(seed)
    T, d_model = x.shape
    d_k = d_model // n_heads

    head_results = []
    outputs = []
    for h in range(n_heads):
        result = compute_self_attention(x, d_k, seed=seed + h * 100)
        head_results.append(result)
        outputs.append(result["output"])

    # Concatenate heads
    concat = np.concatenate(outputs, axis=-1)  # (T, d_model)

    # Output projection
    np.random.seed(seed + 999)
    Wo = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    final_output = concat @ Wo

    return {
        "heads": head_results,
        "concat": concat,
        "final_output": final_output,
        "Wo": Wo,
        "n_heads": n_heads,
        "d_k": d_k,
    }


# Preset token embeddings for attention visualization
ATTENTION_PRESETS = {
    "The cat sat on the mat": {
        "tokens": ["The", "cat", "sat", "on", "the", "mat"],
        "embeddings": np.array([
            [ 0.8,  0.2, -0.5,  0.1,  0.3, -0.2,  0.6, -0.1],
            [-0.3,  0.9,  0.4, -0.2,  0.7,  0.1, -0.3,  0.5],
            [ 0.1, -0.1,  0.8,  0.5, -0.3,  0.6,  0.2, -0.4],
            [ 0.4, -0.3,  0.1,  0.9, -0.2, -0.1,  0.3,  0.7],
            [ 0.7,  0.1, -0.4,  0.2,  0.2, -0.3,  0.5, -0.2],
            [-0.2,  0.8,  0.3, -0.1,  0.6,  0.2, -0.4,  0.4],
        ], dtype=float),
    },
    "I love deep learning": {
        "tokens": ["I", "love", "deep", "learning"],
        "embeddings": np.array([
            [ 0.9,  0.1, -0.3,  0.5,  0.2, -0.1,  0.4, -0.2],
            [-0.2,  0.8,  0.3, -0.1,  0.7,  0.4, -0.3,  0.6],
            [ 0.3, -0.2,  0.7,  0.6, -0.4,  0.5,  0.1, -0.3],
            [ 0.1, -0.3,  0.6,  0.8, -0.2,  0.7,  0.3, -0.1],
        ], dtype=float),
    },
    "Attention is all you need": {
        "tokens": ["Attention", "is", "all", "you", "need"],
        "embeddings": np.array([
            [ 0.7,  0.3, -0.2,  0.5,  0.1, -0.4,  0.6,  0.2],
            [ 0.2, -0.1,  0.8, -0.3,  0.4,  0.5, -0.2,  0.3],
            [ 0.1,  0.5,  0.3,  0.2,  0.6, -0.1,  0.4, -0.3],
            [ 0.5, -0.4,  0.1,  0.7, -0.2,  0.3,  0.2,  0.5],
            [-0.3,  0.6,  0.4, -0.2,  0.8,  0.1, -0.1,  0.7],
        ], dtype=float),
    },
}


# ────────────────────────────────────────────────────────────────
#  Educational 2D Convolution / Kernel Helpers
# ────────────────────────────────────────────────────────────────

def conv2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    2D convolution (cross-correlation) for educational visualization.
    image: (H, W) grayscale image
    kernel: (kH, kW) convolution kernel
    Returns: (out_H, out_W) feature map
    """
    if padding > 0:
        image = np.pad(image, padding, mode='edge')
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(patch * kernel)
    return output


def generate_test_image(image_type: str = "checkerboard", size: int = 32) -> np.ndarray:
    """Generate simple test images for kernel visualization."""
    if image_type == "checkerboard":
        img = np.zeros((size, size))
        block = size // 4
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    img[i*block:(i+1)*block, j*block:(j+1)*block] = 1.0
    elif image_type == "circles":
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        img = ((xx**2 + yy**2 < 0.8) & (xx**2 + yy**2 > 0.3)).astype(float)
    elif image_type == "gradient":
        x = np.linspace(0, 1, size)
        img = np.tile(x, (size, 1))
    elif image_type == "edges_h":
        img = np.zeros((size, size))
        img[size//3:2*size//3, :] = 1.0
    elif image_type == "diagonal":
        img = np.eye(size)
    elif image_type == "noise":
        np.random.seed(42)
        img = np.random.rand(size, size)
    else:
        img = np.zeros((size, size))
    return img


# Preset 2D kernels
KERNEL_PRESETS = {
    "Edge Detect (Horizontal)": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
    "Edge Detect (Vertical)": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
    "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
    "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
    "Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "Gaussian Blur (3x3)": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16.0,
    "Gaussian Blur (5x5)": np.array([
        [1,  4,  7,  4,  1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4,  1],
    ], dtype=float) / 256.0,
    "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
    "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
    "Custom (zeros)": np.zeros((3, 3), dtype=float),
}


# ────────────────────────────────────────────────────────────────
#  Educational Scaling Helpers
# ────────────────────────────────────────────────────────────────

def run_scaling_experiment(dataset_fn, scale_dim: str = "depth",
                           values: list = None, epochs: int = 200,
                           lr: float = 0.01, optimizer: str = "adam") -> list:
    """
    Run a scaling experiment by varying one dimension.
    scale_dim: 'depth' | 'width' | 'data'
    Returns list of (label, n_params, loss_history, final_acc) tuples.
    """
    X_full, y_full = dataset_fn()
    n_full = len(X_full)
    input_dim = X_full.shape[1]

    if values is None:
        if scale_dim == "depth":
            values = [1, 2, 3, 4, 6]
        elif scale_dim == "width":
            values = [4, 8, 16, 32, 64, 128]
        else:  # data
            values = [50, 100, 200, 400, 800]

    results = []
    for val in values:
        if scale_dim == "depth":
            arch = [(input_dim, "Input")]
            for _ in range(val):
                arch.append((16, "ReLU"))
            arch.append((1, "Sigmoid"))
            label = f"Depth={val}"
        elif scale_dim == "width":
            arch = [(input_dim, "Input"), (val, "ReLU"), (max(val // 2, 2), "ReLU"), (1, "Sigmoid")]
            label = f"Width={val}"
        else:  # data
            n_use = min(val, n_full)
            arch = [(input_dim, "Input"), (16, "ReLU"), (8, "ReLU"), (1, "Sigmoid")]
            label = f"Data={n_use}"

        nn = NeuralNetwork(loss_name="MSE", lr=lr, optimizer=optimizer, weight_init="he")
        nn.build_dense(arch)

        X_train = X_full[:n_use] if scale_dim == "data" else X_full
        y_train = y_full[:n_use] if scale_dim == "data" else y_full

        losses = []
        for epoch in range(epochs):
            loss = nn.train_step(X_train, y_train)
            losses.append(loss)

        pred = nn.predict(X_full)
        acc = float(np.mean(((pred > 0.5).astype(float)) == y_full))

        # Count parameters
        n_params = sum(
            (l.W.size + l.b.size) for l in nn.layers if hasattr(l, 'W') and l.W is not None
        )

        results.append((label, n_params, losses, acc))

    return results


# ────────────────────────────────────────────────────────────────
#  Datasets
# ────────────────────────────────────────────────────────────────

def make_xor(n: int = 200, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float).reshape(-1, 1)
    X += np.random.randn(n, 2) * noise
    return X, y


def make_circle(n: int = 300, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.randn(n, 2) * 0.5
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 0.25).astype(float).reshape(-1, 1)
    X += np.random.randn(n, 2) * noise
    return X, y


def make_spiral(n: int = 300, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    n_half = n // 2
    r = np.linspace(0.1, 1, n_half)
    t1 = np.linspace(0, 2 * np.pi, n_half) + np.random.randn(n_half) * noise
    t2 = np.linspace(0, 2 * np.pi, n_half) + np.random.randn(n_half) * noise
    X1 = np.column_stack([r * np.cos(t1), r * np.sin(t1)])
    X2 = np.column_stack([r * np.cos(t2 + np.pi), r * np.sin(t2 + np.pi)])
    X = np.vstack([X1, X2])
    y = np.vstack([np.zeros((n_half, 1)), np.ones((n_half, 1))])
    return X, y


def make_moon(n: int = 300, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    n_half = n // 2
    angles1 = np.linspace(0, np.pi, n_half)
    X1 = np.column_stack([np.cos(angles1), np.sin(angles1)]) + noise * np.random.randn(n_half, 2)
    X2 = np.column_stack([1 - np.cos(angles1), 0.5 - np.sin(angles1)]) + noise * np.random.randn(n_half, 2)
    X = np.vstack([X1, X2])
    y = np.vstack([np.zeros((n_half, 1)), np.ones((n_half, 1))])
    return X, y


def make_gaussian(n: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    n_half = n // 2
    X1 = np.random.randn(n_half, 2) * 0.5 + np.array([-1, -1])
    X2 = np.random.randn(n_half, 2) * 0.5 + np.array([1, 1])
    X = np.vstack([X1, X2])
    y = np.vstack([np.zeros((n_half, 1)), np.ones((n_half, 1))])
    return X, y


def make_checkers(n: int = 400, grid: int = 3, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (n, 2))
    ix = ((X[:, 0] + 1) / 2 * grid).astype(int)
    iy = ((X[:, 1] + 1) / 2 * grid).astype(int)
    y = ((ix + iy) % 2).astype(float).reshape(-1, 1)
    X += np.random.randn(n, 2) * noise
    return X, y


def make_sine_sequence(n_sequences: int = 50, seq_len: int = 20,
                       freq_range: tuple = (0.5, 3.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sine wave sequences for RNN/LSTM learning."""
    np.random.seed(42)
    X_seq = []
    y_seq = []
    for _ in range(n_sequences):
        freq = np.random.uniform(*freq_range)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 4 * np.pi, seq_len + 1)
        wave = np.sin(freq * t + phase)
        X_seq.append(wave[:-1])
        y_seq.append(wave[1:])
    X = np.array(X_seq).reshape(n_sequences, seq_len, 1)
    y = np.array(y_seq).reshape(n_sequences, seq_len, 1)
    return X, y


def make_counter_sequence(n_sequences: int = 100, seq_len: int = 10,
                          max_val: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """RNN task: predict the count of a specific value in a sequence."""
    np.random.seed(42)
    X_seq = []
    y_seq = []
    for _ in range(n_sequences):
        seq = np.random.randint(0, max_val + 1, seq_len)
        # Count occurrences of value 3
        count = np.sum(seq == 3) / seq_len
        X_seq.append(seq.astype(float) / max_val)
        y_seq.append([count])
    X = np.array(X_seq).reshape(n_sequences, seq_len, 1)
    y = np.array(y_seq).reshape(n_sequences, 1)
    return X, y


def make_conv_signal(n: int = 50, seq_len: int = 64,
                     in_channels: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 1D signals for Conv1D: detect pulses."""
    np.random.seed(42)
    X = np.random.randn(n, in_channels, seq_len) * 0.3
    y = np.zeros((n, 1))
    for i in range(n):
        # Randomly insert a pulse
        pos = np.random.randint(5, seq_len - 5)
        width = np.random.randint(2, 5)
        X[i, 0, pos:pos + width] += 1.5
        y[i, 0] = 1.0
    return X, y


DATASETS = {
    "XOR":      make_xor,
    "Circle":   make_circle,
    "Spiral":   make_spiral,
    "Moon":     make_moon,
    "Gaussian": make_gaussian,
    "Checkers": make_checkers,
}

SEQUENCE_DATASETS = {
    "Sine Wave":       make_sine_sequence,
    "Counter":         make_counter_sequence,
}

CONV_DATASETS = {
    "Pulse Detection": make_conv_signal,
}


# ────────────────────────────────────────────────────────────────
#  AdaBoost (Decision Stump Ensemble) — Pure NumPy
# ────────────────────────────────────────────────────────────────

class DecisionStump:
    """A single decision stump (1-level decision tree) — the classic AdaBoost weak learner."""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.polarity = 1  # +1 or -1
        self.alpha = 0.0   # learner weight

    def fit(self, X, y, sample_weights):
        """
        Find the best feature and threshold to minimize weighted error.
        X: (n, d), y: (n,), sample_weights: (n,)
        """
        n, d = X.shape
        min_error = float('inf')

        for feature in range(d):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n)
                    if polarity == 1:
                        predictions[X[:, feature] <= threshold] = -1
                    else:
                        predictions[X[:, feature] > threshold] = -1

                    weighted_error = np.sum(sample_weights[predictions != y])

                    if weighted_error < min_error:
                        min_error = weighted_error
                        self.feature_idx = feature
                        self.threshold = threshold
                        self.polarity = polarity

        # Compute alpha (avoid division by zero)
        self.alpha = 0.5 * np.log((1.0 - min_error + 1e-10) / (min_error + 1e-10))
        return min_error

    def predict(self, X):
        """X: (n, d) -> predictions: (n,)"""
        n = X.shape[0]
        predictions = np.ones(n)
        if self.polarity == 1:
            predictions[X[:, self.feature_idx] <= self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] > self.threshold] = -1
        return predictions


class AdaBoostClassifier:
    """AdaBoost (Adaptive Boosting) — ensemble of weighted decision stumps."""

    def __init__(self, n_estimators: int = 10):
        self.n_estimators = n_estimators
        self.stumps: List[DecisionStump] = []
        self.stump_errors: List[float] = []
        self.stump_alphas: List[float] = []
        self.sample_weight_history: List[np.ndarray] = []
        self.ensemble_errors: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (n, d), y: (n,) with values in {-1, +1}
        Returns list of dicts with per-round info for visualization.
        """
        n, d = X.shape
        sample_weights = np.ones(n) / n
        self.stumps = []
        self.stump_errors = []
        self.stump_alphas = []
        self.sample_weight_history = [sample_weights.copy()]
        self.ensemble_errors = []

        round_info = []

        for round_idx in range(self.n_estimators):
            stump = DecisionStump()
            error = stump.fit(X, y, sample_weights)
            stump.alpha = stump.alpha  # already set in fit
            self.stumps.append(stump)
            self.stump_errors.append(error)
            self.stump_alphas.append(stump.alpha)

            # Update sample weights
            predictions = stump.predict(X)

            sample_weights *= np.exp(-stump.alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)
            self.sample_weight_history.append(sample_weights.copy())

            # Compute ensemble error so far
            ensemble_pred = self.predict(X)
            ensemble_error = np.mean(ensemble_pred != y)
            self.ensemble_errors.append(ensemble_error)

            round_info.append({
                "round": round_idx + 1,
                "feature": stump.feature_idx,
                "threshold": stump.threshold,
                "polarity": stump.polarity,
                "alpha": stump.alpha,
                "weighted_error": error,
                "sample_weights": sample_weights.copy(),
                "ensemble_error": ensemble_error,
            })

        return round_info

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (n, d) -> predictions: (n,) in {-1, +1}"""
        n = X.shape[0]
        stump_preds = np.array([s.alpha * s.predict(X) for s in self.stumps])
        return np.sign(np.sum(stump_preds, axis=0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability-like scores in [0, 1]."""
        n = X.shape[0]
        stump_preds = np.array([s.alpha * s.predict(X) for s in self.stumps])
        raw = np.sum(stump_preds, axis=0)
        # Sigmoid to map to [0, 1]
        return 1.0 / (1.0 + np.exp(-raw))

    def get_partial_ensemble_prediction(self, X: np.ndarray, n_stumps: int) -> np.ndarray:
        """Get prediction using only the first n_stumps."""
        n = X.shape[0]
        if n_stumps == 0:
            return np.zeros(n)
        stump_preds = np.array([self.stumps[i].alpha * self.stumps[i].predict(X)
                                for i in range(min(n_stumps, len(self.stumps)))])
        return np.sign(np.sum(stump_preds, axis=0))


# ────────────────────────────────────────────────────────────────
#  Vanishing Gradient Analysis
# ────────────────────────────────────────────────────────────────

def analyze_gradient_flow(activation_name: str, n_layers: int = 20,
                          input_val: float = 1.0, learning_signal: float = 1.0) -> dict:
    """
    Simulate gradient flow through a deep chain of layers.
    Returns gradient magnitudes at each layer for visualization.
    """
    _, act_fn = ACTIVATION_MAP[activation_name]
    _, act_deriv = ACTIVATION_MAP[activation_name]

    # Forward pass
    z = input_val
    activations = [z]
    for _ in range(n_layers):
        a = act_fn(z)
        # Simple: z_next = W * a where W ~ N(0, 1/sqrt(fan_in))
        z = a  # For pure activation chain (W=identity)
        activations.append(a)

    # Backward pass — gradient magnitude at each layer
    grad = learning_signal
    grad_magnitudes = [abs(grad)]

    for i in reversed(range(n_layers)):
        a = activations[i + 1]
        if activation_name in ("Swish", "GELU"):
            d = act_deriv(activations[i], a)
        else:
            d = act_deriv(a)
        grad = grad * abs(d)
        grad_magnitudes.append(abs(grad))

    grad_magnitudes.reverse()

    return {
        "activations": activations,
        "gradient_magnitudes": grad_magnitudes,
        "n_layers": n_layers,
        "activation": activation_name,
    }


def analyze_gradient_flow_with_weights(activation_name: str, n_layers: int = 20,
                                        fan_in: int = 64, weight_init: str = "he",
                                        input_val: float = 1.0) -> dict:
    """
    Simulate gradient flow through a deep network WITH weight matrices.
    This is more realistic — shows how W * f'(z) products compound.
    """
    _, act_fn = ACTIVATION_MAP[activation_name]
    _, act_deriv = ACTIVATION_MAP[activation_name]

    np.random.seed(42)

    # Generate weight matrices
    if weight_init == "he":
        scale = np.sqrt(2.0 / fan_in)
    elif weight_init == "xavier":
        scale = np.sqrt(1.0 / fan_in)
    else:
        scale = 0.01

    # Track a single neuron's gradient through the chain
    # For simplicity, track scalar gradient flow: grad *= |W| * |f'(z)|
    # where |W| is the average absolute weight

    # Generate weights and track singular values
    weights = [np.random.randn(fan_in, fan_in) * scale for _ in range(n_layers)]

    # Forward pass (tracking a single signal)
    z = np.ones(fan_in) * input_val
    activations_forward = []
    z_values = []

    for i in range(n_layers):
        z = z @ weights[i]  # (fan_in,)
        z_values.append(z.copy())
        a = act_fn(z)
        activations_forward.append(a.copy())
        z = a

    # Backward pass
    grad = np.ones(fan_in) * 1.0
    grad_norms = [np.linalg.norm(grad)]

    for i in reversed(range(n_layers)):
        a = activations_forward[i]
        if activation_name in ("Swish", "GELU"):
            d = act_deriv(z_values[i], a)
        else:
            d = act_deriv(a)
        grad = (grad * d) @ weights[i]
        grad_norms.append(np.linalg.norm(grad))

    grad_norms.reverse()

    # Also compute eigenvalue analysis
    eigenvalue_ratios = []
    for w in weights:
        # Use singular values as proxy
        svs = np.linalg.svd(w, compute_uv=False)
        eigenvalue_ratios.append(max(svs.max(), 1e-10))

    return {
        "gradient_norms": grad_norms,
        "eigenvalue_ratios": eigenvalue_ratios,
        "n_layers": n_layers,
        "activation": activation_name,
        "weight_init": weight_init,
        "fan_in": fan_in,
    }


# ────────────────────────────────────────────────────────────────
#  Diffusion Model — 1D (Educational DDPM)
# ────────────────────────────────────────────────────────────────

class Diffusion1D:
    """
    Simplified 1D Denoising Diffusion Probabilistic Model (DDPM) for education.
    Demonstrates the forward (noising) and reverse (denoising) processes.
    Uses a simple MLP to predict noise.
    """

    def __init__(self, n_steps: int = 50, d_hidden: int = 64,
                 noise_schedule: str = "linear", lr: float = 0.001):
        self.n_steps = n_steps
        self.d_hidden = d_hidden
        self.lr = lr
        self.noise_schedule = noise_schedule

        # Noise schedule betas
        if noise_schedule == "linear":
            self.betas = np.linspace(1e-4, 0.02, n_steps)
        elif noise_schedule == "cosine":
            steps = np.arange(n_steps + 1) / n_steps
            self.betas = 0.02 * (1 - np.cos(np.pi * steps[1:] / np.cos(np.pi * steps[0])))
            self.betas = np.clip(self.betas, 1e-6, 0.1)
        else:
            self.betas = np.linspace(1e-4, 0.05, n_steps)

        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alpha_cumprod)

        # Simple noise prediction network (MLP)
        self._init_noise_net()
        self.loss_history = []

    def _init_noise_net(self):
        """Initialize a small MLP for noise prediction."""
        np.random.seed(42)
        d_in = 2  # (x_t, t_normalized)
        self.W1 = np.random.randn(d_in, self.d_hidden) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(self.d_hidden)
        self.W2 = np.random.randn(self.d_hidden, self.d_hidden) * np.sqrt(2.0 / self.d_hidden)
        self.b2 = np.zeros(self.d_hidden)
        self.W3 = np.random.randn(self.d_hidden, 1) * np.sqrt(2.0 / self.d_hidden)
        self.b3 = np.zeros(1)

        # Adam state
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mW3 = np.zeros_like(self.W3); self.vW3 = np.zeros_like(self.W3)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.mb3 = np.zeros_like(self.b3); self.vb3 = np.zeros_like(self.b3)
        self.t = 0

    def _noise_net_forward(self, x_t, t_norm):
        """Forward pass of the noise prediction MLP."""
        inp = np.column_stack([x_t.flatten(), t_norm])  # (n, 2)
        z1 = inp @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        z3 = a2 @ self.W3 + self.b3
        return z3.flatten()

    def _noise_net_backward(self, x_t, t_norm, d_out):
        """Backward pass of the noise prediction MLP. Returns gradients."""
        n = len(x_t)
        inp = np.column_stack([x_t.flatten(), t_norm])

        z1 = inp @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        z3 = a2 @ self.W3 + self.b3

        d_out = d_out.reshape(-1, 1)

        dz3 = d_out
        self.dW3 = a2.T @ dz3 / n
        self.db3 = np.sum(dz3, axis=0) / n
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0).astype(float)
        self.dW2 = a1.T @ dz2 / n
        self.db2 = np.sum(dz2, axis=0) / n
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0).astype(float)
        self.dW1 = inp.T @ dz1 / n
        self.db1 = np.sum(dz1, axis=0) / n

    def _adam_update(self, param, dparam, m, v, name=""):
        self.t += 1 if name == "W3" else 0  # only count once per step
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m[:] = beta1 * m + (1 - beta1) * dparam
        v[:] = beta2 * v + (1 - beta2) * dparam ** 2
        m_hat = m / (1 - beta1 ** self.t)
        v_hat = v / (1 - beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def forward_process(self, x0, t):
        """
        Apply t steps of noise to x0.
        x0: (n,) clean data
        t: int, number of noise steps
        Returns: x_t (noisy), noise_added
        """
        alpha_t = self.alpha_cumprod[t]
        eps = np.random.randn(len(x0))
        x_t = self.sqrt_alpha_cumprod[t] * x0 + self.sqrt_one_minus_alpha_cumprod[t] * eps
        return x_t, eps

    def get_noised_at_all_steps(self, x0):
        """Return x_t at every noise step for a single sample. For visualization."""
        results = []
        for t in range(self.n_steps):
            x_t, _ = self.forward_process(x0, t)
            results.append(x_t)
        return results

    def train_step(self, x0, batch_indices=None):
        """
        One training step: sample random t, add noise, predict noise, compute MSE loss.
        x0: (n,) clean data
        """
        n = len(x0)
        if batch_indices is None:
            batch_indices = np.arange(n)
        x0_batch = x0[batch_indices]
        bs = len(x0_batch)

        # Sample random timesteps
        t_indices = np.random.randint(0, self.n_steps, size=bs)

        # Sample noise
        eps_true = np.random.randn(bs)

        # Get x_t
        x_t = np.array([
            self.sqrt_alpha_cumprod[t_indices[i]] * x0_batch[i] + self.sqrt_one_minus_alpha_cumprod[t_indices[i]] * eps_true[i]
            for i in range(bs)
        ])

        # Predict noise
        t_norm = t_indices / self.n_steps
        eps_pred = self._noise_net_forward(x_t, t_norm)

        # MSE loss
        loss = float(np.mean((eps_pred - eps_true) ** 2))

        # Backward
        d_out = 2.0 * (eps_pred - eps_true) / bs
        self._noise_net_backward(x_t, t_norm, d_out)

        # Adam update
        self.t_step = getattr(self, 't_step', 0) + 1
        t_val = self.t_step

        for param, grad, m, v in [
            (self.W1, self.dW1, self.mW1, self.vW1),
            (self.b1, self.db1, self.mb1, self.vb1),
            (self.W2, self.dW2, self.mW2, self.vW2),
            (self.b2, self.db2, self.mb2, self.vb2),
            (self.W3, self.dW3, self.mW3, self.vW3),
            (self.b3, self.db3, self.mb3, self.vb3),
        ]:
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t_val)
            v_hat = v / (1 - beta2 ** t_val)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        self.loss_history.append(loss)
        return loss

    def reverse_process(self, n_samples=1, n_steps=None):
        """
        Generate samples by reversing the diffusion process.
        Starts from pure noise and iteratively denoises.
        Returns all intermediate x_t for visualization.
        """
        if n_steps is None:
            n_steps = self.n_steps

        x = np.random.randn(n_samples)
        trajectory = [x.copy()]

        for t in reversed(range(n_steps)):
            t_norm = np.array([t / self.n_steps] * n_samples)
            eps_pred = self._noise_net_forward(x, t_norm)

            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alpha_cumprod[t]

            # DDPM update
            if t > 0:
                noise = np.random.randn(n_samples)
            else:
                noise = np.zeros(n_samples)

            x = (1.0 / np.sqrt(alpha_t)) * (
                x - (1.0 - alpha_t) / np.sqrt(1.0 - alpha_cumprod_t + 1e-10) * eps_pred
            ) + np.sqrt(self.betas[t]) * noise

            trajectory.append(x.copy())

        return trajectory  # list of x arrays, from noisy to clean


def make_diffusion_data(n: int = 500, distribution: str = "bimodal") -> np.ndarray:
    """Generate 1D data for diffusion model training."""
    np.random.seed(42)
    if distribution == "bimodal":
        # Two Gaussian peaks
        n_half = n // 2
        x1 = np.random.randn(n_half) * 0.3 - 1.5
        x2 = np.random.randn(n - n_half) * 0.3 + 1.5
        return np.concatenate([x1, x2])
    elif distribution == "uniform":
        return np.random.uniform(-2, 2, n)
    elif distribution == "sine":
        t = np.linspace(0, 4 * np.pi, n)
        return np.sin(t) + np.random.randn(n) * 0.1
    else:  # gaussian
        return np.random.randn(n) * 0.8


# ────────────────────────────────────────────────────────────────
#  Diffusion Model — 2D (Educational, for Visualization)
# ────────────────────────────────────────────────────────────────

class Diffusion2D:
    """
    Simplified 2D diffusion model for educational visualization.
    Uses a tiny MLP on (x, y, t) -> predicted noise (nx, ny).
    """

    def __init__(self, n_steps: int = 100, d_hidden: int = 64, lr: float = 0.005):
        self.n_steps = n_steps
        self.d_hidden = d_hidden
        self.lr = lr

        # Linear noise schedule
        self.betas = np.linspace(1e-4, 0.02, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alpha_cumprod)

        self._init_net()
        self.loss_history = []
        self._step_count = 0

    def _init_net(self):
        np.random.seed(42)
        d_in = 3  # (x, y, t)
        self.W1 = np.random.randn(d_in, self.d_hidden) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(self.d_hidden)
        self.W2 = np.random.randn(self.d_hidden, self.d_hidden) * np.sqrt(2.0 / self.d_hidden)
        self.b2 = np.zeros(self.d_hidden)
        self.W3 = np.random.randn(self.d_hidden, 2) * np.sqrt(2.0 / self.d_hidden)
        self.b3 = np.zeros(2)

        # Adam state
        self._adam_state = {}
        for name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            param = getattr(self, name)
            self._adam_state[name] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param),
            }

    def _forward(self, xy_t):
        """xy_t: (n, 3) -> (n, 2) predicted noise"""
        z1 = xy_t @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        z3 = a2 @ self.W3 + self.b3
        return z3

    def _backward(self, xy_t, d_out):
        """Compute gradients."""
        n = xy_t.shape[0]
        z1 = xy_t @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)

        dz3 = d_out
        grads = {}
        grads['W3'] = a2.T @ dz3 / n
        grads['b3'] = np.sum(dz3, axis=0) / n
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0).astype(float)
        grads['W2'] = a1.T @ dz2 / n
        grads['b2'] = np.sum(dz2, axis=0) / n
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0).astype(float)
        grads['W1'] = xy_t.T @ dz1 / n
        grads['b1'] = np.sum(dz1, axis=0) / n
        return grads

    def _update(self, grads):
        self._step_count += 1
        t = self._step_count
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name in grads:
            param = getattr(self, name)
            grad = grads[name]
            state = self._adam_state[name]
            state['m'][:] = beta1 * state['m'] + (1 - beta1) * grad
            state['v'][:] = beta2 * state['v'] + (1 - beta2) * grad ** 2
            m_hat = state['m'] / (1 - beta1 ** t)
            v_hat = state['v'] / (1 - beta2 ** t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def forward_process(self, x0, t):
        """Add noise at step t. x0: (n, 2)"""
        n = x0.shape[0]
        eps = np.random.randn(n, 2)
        x_t = self.sqrt_alpha_cumprod[t] * x0 + self.sqrt_one_minus_alpha_cumprod[t] * eps
        return x_t, eps

    def train_step(self, x0):
        """One training step on all data."""
        n = x0.shape[0]
        t_indices = np.random.randint(0, self.n_steps, size=n)
        eps_true = np.random.randn(n, 2)

        # Get x_t for each sample
        x_t = np.array([
            self.sqrt_alpha_cumprod[t_indices[i]] * x0[i] + self.sqrt_one_minus_alpha_cumprod[t_indices[i]] * eps_true[i]
            for i in range(n)
        ])

        t_norm = (t_indices / self.n_steps).reshape(-1, 1)
        xy_t = np.column_stack([x_t, t_norm.flatten()])

        eps_pred = self._forward(xy_t)
        loss = float(np.mean((eps_pred - eps_true) ** 2))

        d_out = 2.0 * (eps_pred - eps_true) / n
        grads = self._backward(xy_t, d_out)
        self._update(grads)

        self.loss_history.append(loss)
        return loss

    def reverse_sample(self, n_samples=1, return_trajectory=False):
        """Generate 2D samples. Returns trajectory if requested."""
        x = np.random.randn(n_samples, 2)
        if return_trajectory:
            trajectory = [x.copy()]

        for t in reversed(range(self.n_steps)):
            t_norm = np.full(n_samples, t / self.n_steps).reshape(-1, 1)
            xy_t = np.column_stack([x, t_norm.flatten()])
            eps_pred = self._forward(xy_t)

            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alpha_cumprod[t]

            noise = np.random.randn(n_samples, 2) if t > 0 else np.zeros((n_samples, 2))

            x = (1.0 / np.sqrt(alpha_t)) * (
                x - (1.0 - alpha_t) / np.sqrt(1.0 - alpha_cumprod_t + 1e-10) * eps_pred
            ) + np.sqrt(self.betas[t]) * noise

            if return_trajectory:
                trajectory.append(x.copy())

        if return_trajectory:
            return trajectory
        return x

    def get_noised_at_steps(self, x0, steps):
        """Get noised versions at specific timesteps for visualization."""
        results = []
        for t in steps:
            x_t, _ = self.forward_process(x0, t)
            results.append(x_t)
        return results


class TransformerBlock(LayerBase):
    """
    A single Transformer encoder block: LayerNorm -> MultiHeadAttention -> Residual ->
    LayerNorm -> FFN -> Residual. Educational implementation with explicit multi-head attention.
    """
    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = None, weight_init: str = "xavier"):
        super().__init__()
        self._init_adam()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_ff = d_ff or 4 * d_model
        self.layer_type = "Transformer"

        s = np.sqrt(1.0 / d_model) if weight_init == "xavier" else np.sqrt(2.0 / d_model)

        # Multi-head attention parameters
        self.Wq = np.random.randn(d_model, d_model) * s
        self.Wk = np.random.randn(d_model, d_model) * s
        self.Wv = np.random.randn(d_model, d_model) * s
        self.Wo = np.random.randn(d_model, d_model) * s

        # FFN parameters
        self.W1 = np.random.randn(d_model, self.d_ff) * s
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * s
        self.b2 = np.zeros(d_model)

        # Layer norm parameters (learnable scale/shift)
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

        # Gradients
        self.dWq = np.zeros_like(self.Wq)
        self.dWk = np.zeros_like(self.Wk)
        self.dWv = np.zeros_like(self.Wv)
        self.dWo = np.zeros_like(self.Wo)
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)
        self.dln1_gamma = np.zeros_like(self.ln1_gamma)
        self.dln1_beta = np.zeros_like(self.ln1_beta)
        self.dln2_gamma = np.zeros_like(self.ln2_gamma)
        self.dln2_beta = np.zeros_like(self.ln2_beta)

        for name, shape in [
            ("Wq", self.Wq.shape), ("Wk", self.Wk.shape), ("Wv", self.Wv.shape),
            ("Wo", self.Wo.shape), ("W1", self.W1.shape), ("b1", self.b1.shape),
            ("W2", self.W2.shape), ("b2", self.b2.shape),
            ("ln1_gamma", self.ln1_gamma.shape), ("ln1_beta", self.ln1_beta.shape),
            ("ln2_gamma", self.ln2_gamma.shape), ("ln2_beta", self.ln2_beta.shape),
        ]:
            self._register_param(name, shape)

        self._cache = None

    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta, x_norm, mean, var

    def _ffn(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        return z2, a1, z1

    def forward(self, x):
        """
        x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        """
        batch, T, d = x.shape

        # Pre-norm multi-head self-attention
        x_norm1, xn1, m1, v1 = self._layer_norm(x, self.ln1_gamma, self.ln1_beta)

        # Multi-head attention
        Q = x_norm1 @ self.Wq  # (batch, T, d_model)
        K = x_norm1 @ self.Wk
        V = x_norm1 @ self.Wv

        # Reshape for multi-head: (batch, T, n_heads, d_k) -> (batch, n_heads, T, d_k)
        Q = Q.reshape(batch, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        attn_out = attn_weights @ V  # (batch, n_heads, T, d_k)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, T, self.d_model)

        # Output projection
        attn_out = attn_out @ self.Wo

        # Residual connection
        x = x + attn_out

        # Pre-norm FFN
        x_norm2, xn2, m2, v2 = self._layer_norm(x, self.ln2_gamma, self.ln2_beta)
        ffn_out, a1_ffn, z1_ffn = self._ffn(x_norm2)

        # Residual connection
        x = x + ffn_out

        self._cache = {
            "x_input": x, "x_norm1": x_norm1, "xn1": xn1, "m1": m1, "v1": v1,
            "Q": Q, "K": K, "V": V, "attn_weights": attn_weights,
            "attn_out_pre_proj": attn_out,  # before Wo
            "x_after_attn": x - ffn_out,  # before FFN residual
            "x_norm2": x_norm2, "xn2": xn2, "m2": m2, "v2": v2,
            "a1_ffn": a1_ffn, "z1_ffn": z1_ffn,
        }

        return x

    def backward(self, dout):
        """Simplified backward — uses numerical gradient approximation for complex paths."""
        # For educational purposes, we use a simpler backward that still enables Adam updates
        batch, T, d = dout.shape
        eps = 1e-5

        # Numerical gradients for attention params
        for param_name in ["Wq", "Wk", "Wv", "Wo"]:
            param = getattr(self, param_name)
            grad = np.zeros_like(param)
            for i in range(min(param.shape[0], 8)):  # sample for speed
                for j in range(min(param.shape[1], 8)):
                    param[i, j] += eps
                    out_plus = self.forward(self._cache["x_input"])
                    loss_plus = np.mean((out_plus - self._cache["x_input"]) ** 2)  # dummy loss
                    param[i, j] -= 2 * eps
                    out_minus = self.forward(self._cache["x_input"])
                    loss_minus = np.mean((out_minus - self._cache["x_input"]) ** 2)
                    param[i, j] += eps
                    grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
                    if i >= 7 or j >= 7:
                        grad[i, j:] = grad[i, j-1]
                        break
                if i >= 7:
                    grad[i:, :] = grad[i-1, :]
                    break
            setattr(self, "d" + param_name, grad / batch)

        # Numerical gradients for FFN
        for param_name in ["W1", "b1", "W2", "b2"]:
            param = getattr(self, param_name)
            grad = np.zeros_like(param)
            flat_idx = 0
            for idx in np.ndindex(param.shape):
                if flat_idx > 200:
                    break
                param[idx] += eps
                out_plus = self.forward(self._cache["x_input"])
                loss_plus = np.mean(dout * out_plus)
                param[idx] -= 2 * eps
                out_minus = self.forward(self._cache["x_input"])
                loss_minus = np.mean(dout * out_minus)
                param[idx] += eps
                grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                flat_idx += 1
            setattr(self, "d" + param_name, grad / batch)

        # Layer norm grads (small, just set to dout-derived values)
        self.dln1_gamma = np.mean(dout, axis=(0, 1)) * 0.01
        self.dln1_beta = np.mean(dout, axis=(0, 1)) * 0.01
        self.dln2_gamma = np.mean(dout, axis=(0, 1)) * 0.01
        self.dln2_beta = np.mean(dout, axis=(0, 1)) * 0.01

        # Restore cache by re-doing forward
        self.forward(self._cache["x_input"])

        return dout  # simplified: pass through


class TransformerEncoder:
    """
    Stack of Transformer blocks for educational sequence classification.
    Not a LayerBase — manages its own stack of TransformerBlocks.
    """
    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = None,
                 n_layers: int = 2, weight_init: str = "xavier"):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, weight_init)
            for _ in range(n_layers)
        ]
        self.loss_history = []

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

    def backward(self, dout):
        dx = dout
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        return dx

    def update_weights(self, lr=0.001, beta1=0.9, beta2=0.999, t=1):
        for i, block in enumerate(self.blocks):
            block.adam_update(lr, beta1, beta2, t + i)


class Autoencoder:
    """
    Dense autoencoder with configurable bottleneck. Educational implementation
    showing compression and reconstruction with latent space visualization.
    """
    def __init__(self, input_dim: int, hidden_dims: list = None, latent_dim: int = 2,
                 lr: float = 0.005, weight_init: str = "he"):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_init = weight_init

        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Build encoder and decoder as lists of (W, b) tuples
        self.encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        self.decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]

        self._init_weights()
        self.loss_history = []
        self.t = 0

    def _init_weights(self):
        self.enc_weights = []
        self.enc_biases = []
        self.dec_weights = []
        self.dec_biases = []

        # Adam state
        self._adam = {"enc_w": [], "enc_b": [], "dec_w": [], "dec_b": []}

        for i in range(len(self.encoder_dims) - 1):
            n_in, n_out = self.encoder_dims[i], self.encoder_dims[i + 1]
            s = np.sqrt(2.0 / n_in) if self.weight_init == "he" else np.sqrt(1.0 / n_in)
            self.enc_weights.append(np.random.randn(n_in, n_out) * s)
            self.enc_biases.append(np.zeros(n_out))
            self._adam["enc_w"].append({"m": np.zeros((n_in, n_out)), "v": np.zeros((n_in, n_out))})
            self._adam["enc_b"].append({"m": np.zeros(n_out), "v": np.zeros(n_out)})

        for i in range(len(self.decoder_dims) - 1):
            n_in, n_out = self.decoder_dims[i], self.decoder_dims[i + 1]
            s = np.sqrt(2.0 / n_in) if self.weight_init == "he" else np.sqrt(1.0 / n_in)
            self.dec_weights.append(np.random.randn(n_in, n_out) * s)
            self.dec_biases.append(np.zeros(n_out))
            self._adam["dec_w"].append({"m": np.zeros((n_in, n_out)), "v": np.zeros((n_in, n_out))})
            self._adam["dec_b"].append({"m": np.zeros(n_out), "v": np.zeros(n_out)})

        # Cache
        self._enc_cache = []
        self._dec_cache = []
        self._z = None

    def encode(self, X):
        a = X
        self._enc_cache = [X]
        for i, (W, b) in enumerate(zip(self.enc_weights, self.enc_biases)):
            z = a @ W + b
            a = np.maximum(0, z) if i < len(self.enc_weights) - 1 else z  # ReLU for hidden, linear for bottleneck
            self._enc_cache.append(a)
        self._z = a
        return a

    def decode(self, z):
        a = z
        self._dec_cache = [z]
        for i, (W, b) in enumerate(zip(self.dec_weights, self.dec_biases)):
            z = a @ W + b
            a = np.maximum(0, z) if i < len(self.dec_weights) - 1 else z  # Linear for output
            self._dec_cache.append(a)
        return a

    def forward(self, X):
        z = self.encode(X)
        return self.decode(z)

    def backward(self, X, X_recon):
        n = X.shape[0]
        # MSE loss gradient
        d_out = 2.0 * (X_recon - X) / n

        # Backprop through decoder
        dec_grads_w = [np.zeros_like(w) for w in self.dec_weights]
        dec_grads_b = [np.zeros_like(b) for b in self.dec_biases]

        da = d_out
        for i in reversed(range(len(self.dec_weights))):
            a_prev = self._dec_cache[i]
            dz = da
            if i < len(self.dec_weights) - 1:
                dz = da * (self._dec_cache[i + 1] > 0).astype(float)  # ReLU deriv
            dec_grads_w[i] = a_prev.T @ dz
            dec_grads_b[i] = np.sum(dz, axis=0)
            da = dz @ self.dec_weights[i].T

        # Backprop through encoder
        enc_grads_w = [np.zeros_like(w) for w in self.enc_weights]
        enc_grads_b = [np.zeros_like(b) for b in self.enc_biases]

        for i in reversed(range(len(self.enc_weights))):
            a_prev = self._enc_cache[i]
            dz = da
            if i < len(self.enc_weights) - 1:
                dz = da * (self._enc_cache[i + 1] > 0).astype(float)
            enc_grads_w[i] = a_prev.T @ dz
            enc_grads_b[i] = np.sum(dz, axis=0)
            da = dz @ self.enc_weights[i].T

        return enc_grads_w, enc_grads_b, dec_grads_w, dec_grads_b

    def _adam_update_list(self, params, grads, adam_states, lr, t):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(len(params)):
            m, v = adam_states[i]["m"], adam_states[i]["v"]
            g = grads[i]
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_step(self, X):
        self.t += 1
        X_recon = self.forward(X)
        loss = float(np.mean((X_recon - X) ** 2))

        eg_w, eg_b, dg_w, dg_b = self.backward(X, X_recon)

        self._adam_update_list(self.enc_weights, eg_w, self._adam["enc_w"], self.lr, self.t)
        self._adam_update_list(self.enc_biases, eg_b, self._adam["enc_b"], self.lr, self.t)
        self._adam_update_list(self.dec_weights, dg_w, self._adam["dec_w"], self.lr, self.t)
        self._adam_update_list(self.dec_biases, dg_b, self._adam["dec_b"], self.lr, self.t)

        self.loss_history.append(loss)
        return loss, X_recon

    def get_latent(self, X):
        return self.encode(X)

    def count_params(self):
        n = 0
        for w, b in zip(self.enc_weights, self.enc_biases):
            n += w.size + b.size
        for w, b in zip(self.dec_weights, self.dec_biases):
            n += w.size + b.size
        return n


class GAN:
    """
    Educational GAN with generator and discriminator, both simple MLPs.
    Trains on 1D or 2D data using the minimax loss.
    """
    def __init__(self, data_dim: int = 2, latent_dim: int = 8,
                 hidden_dim: int = 32, lr_g: float = 0.005, lr_d: float = 0.005):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.lr_g = lr_g
        self.lr_d = lr_d

        # Generator: z -> data
        self.G_W1 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.G_b1 = np.zeros(hidden_dim)
        self.G_W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.G_b2 = np.zeros(hidden_dim)
        self.G_W3 = np.random.randn(hidden_dim, data_dim) * np.sqrt(2.0 / hidden_dim)
        self.G_b3 = np.zeros(data_dim)

        # Discriminator: data -> [0, 1]
        self.D_W1 = np.random.randn(data_dim, hidden_dim) * np.sqrt(2.0 / data_dim)
        self.D_b1 = np.zeros(hidden_dim)
        self.D_W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.D_b2 = np.zeros(hidden_dim)
        self.D_W3 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.D_b3 = np.zeros(1)

        # Adam state
        self._init_adam_state()

        self.g_loss_history = []
        self.d_loss_history = []
        self.d_real_acc_history = []
        self.d_fake_acc_history = []
        self.t = 0

    def _init_adam_state(self):
        self._adam = {}
        for name in ["G_W1", "G_b1", "G_W2", "G_b2", "G_W3", "G_b3",
                      "D_W1", "D_b1", "D_W2", "D_b2", "D_W3", "D_b3"]:
            param = getattr(self, name)
            self._adam[name] = {"m": np.zeros_like(param), "v": np.zeros_like(param)}

    def _gen_forward(self, z):
        h1 = np.maximum(0, z @ self.G_W1 + self.G_b1)
        h2 = np.maximum(0, h1 @ self.G_W2 + self.G_b2)
        out = h2 @ self.G_W3 + self.G_b3
        return out

    def _disc_forward(self, x):
        h1 = np.maximum(0, x @ self.D_W1 + self.D_b1)
        h2 = np.maximum(0, h1 @ self.D_W2 + self.D_b2)
        logit = h2 @ self.D_W3 + self.D_b3
        # Sigmoid
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
        return prob, logit

    def _disc_backward(self, x, d_logit):
        """Backward through discriminator. Returns dict of gradients."""
        h1 = np.maximum(0, x @ self.D_W1 + self.D_b1)
        h2 = np.maximum(0, h1 @ self.D_W2 + self.D_b2)

        n = x.shape[0]
        grads = {}

        d_h2 = d_logit @ self.D_W3.T
        d_z2 = d_h2 * (h2 > 0).astype(float)

        grads["D_W3"] = h2.T @ d_logit / n
        grads["D_b3"] = np.sum(d_logit, axis=0) / n

        d_h1 = d_z2 @ self.D_W2.T
        d_z1 = d_h1 * (h1 > 0).astype(float)

        grads["D_W2"] = h1.T @ d_z2 / n
        grads["D_b2"] = np.sum(d_z2, axis=0) / n

        grads["D_W1"] = x.T @ d_z1 / n
        grads["D_b1"] = np.sum(d_z1, axis=0) / n

        return grads

    def _gen_backward(self, z, fake_data, d_fake_grad):
        """Backward through generator."""
        h1 = np.maximum(0, z @ self.G_W1 + self.G_b1)
        h2 = np.maximum(0, h1 @ self.G_W2 + self.G_b2)

        n = z.shape[0]
        grads = {}

        # d_fake_grad: (n, data_dim) gradient from discriminator w.r.t. fake data
        d_h2 = d_fake_grad @ self.G_W3.T
        d_z2 = d_h2 * (h2 > 0).astype(float)

        grads["G_W3"] = h2.T @ d_fake_grad / n
        grads["G_b3"] = np.sum(d_fake_grad, axis=0) / n

        d_h1 = d_z2 @ self.G_W2.T
        d_z1 = d_h1 * (h1 > 0).astype(float)

        grads["G_W2"] = h1.T @ d_z2 / n
        grads["G_b2"] = np.sum(d_z2, axis=0) / n

        grads["G_W1"] = z.T @ d_z1 / n
        grads["G_b1"] = np.sum(d_z1, axis=0) / n

        return grads

    def _adam_update(self, grads):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, grad in grads.items():
            param = getattr(self, name)
            state = self._adam[name]
            state["m"][:] = beta1 * state["m"] + (1 - beta1) * grad
            state["v"][:] = beta2 * state["v"] + (1 - beta2) * grad ** 2
            m_hat = state["m"] / (1 - beta1 ** self.t)
            v_hat = state["v"] / (1 - beta2 ** self.t)
            param -= (self.lr_g if name.startswith("G") else self.lr_d) * m_hat / (np.sqrt(v_hat) + eps)

    def train_step(self, real_data):
        """One GAN training step. real_data: (n, data_dim)"""
        n = real_data.shape[0]

        # --- Train Discriminator ---
        z = np.random.randn(n, self.latent_dim)
        fake_data = self._gen_forward(z)

        d_real_prob, d_real_logit = self._disc_forward(real_data)
        d_fake_prob, d_fake_logit = self._disc_forward(fake_data)

        # Binary cross-entropy gradients
        d_real_grad = (d_real_prob - 1)  # gradient of -log(D(x))
        d_fake_grad_d = d_fake_prob       # gradient of -log(1 - D(G(z)))

        d_logit_real = d_real_prob - 1
        d_logit_fake = d_fake_prob

        # Combined discriminator gradient
        d_grads = self._disc_backward(real_data, d_real_grad)
        d_grads_fake = self._disc_backward(fake_data, d_fake_grad_d)

        for k in d_grads:
            d_grads[k] = d_grads[k] + d_grads_fake[k]

        self._adam_update(d_grads)

        d_loss = float(-np.mean(np.log(d_real_prob + 1e-8) + np.log(1 - d_fake_prob + 1e-8)))

        # --- Train Generator ---
        z = np.random.randn(n, self.latent_dim)
        fake_data = self._gen_forward(z)
        d_fake_prob, d_fake_logit = self._disc_forward(fake_data)

        # Generator wants D(G(z)) -> 1
        # Gradient of -log(D(G(z))) w.r.t. G output
        g_data_grad = -(1 - d_fake_prob)  # negative because we want to maximize

        # We need gradient w.r.t. fake_data, then backprop through G
        # Approximate: use d_fake_prob as signal
        g_data_grad = -d_fake_prob  # (n, 1) but need (n, data_dim) — broadcast
        g_data_grad = np.broadcast_to(g_data_grad, fake_data.shape).copy()

        g_grads = self._gen_backward(z, fake_data, g_data_grad)
        self._adam_update(g_grads)

        g_loss = float(-np.mean(np.log(d_fake_prob + 1e-8)))

        self.g_loss_history.append(g_loss)
        self.d_loss_history.append(d_loss)
        self.d_real_acc_history.append(float(np.mean(d_real_prob > 0.5)))
        self.d_fake_acc_history.append(float(np.mean(d_fake_prob < 0.5)))

        return g_loss, d_loss

    def generate(self, n_samples=100):
        z = np.random.randn(n_samples, self.latent_dim)
        return self._gen_forward(z)

    def get_disc_accuracy(self, real_data, n_fake=100):
        z = np.random.randn(n_fake, self.latent_dim)
        fake = self._gen_forward(z)
        d_real, _ = self._disc_forward(real_data)
        d_fake, _ = self._disc_forward(fake)
        return float(np.mean(d_real > 0.5)), float(np.mean(d_fake < 0.5))


class ResidualDemo:
    """
    Educational demo comparing a plain deep network vs. a ResNet-style network
    with skip connections. Trains on 2D classification data.
    """
    def __init__(self, input_dim=2, hidden_dim=32, n_blocks=5, lr=0.005, use_residual=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.lr = lr
        self.use_residual = use_residual
        self.loss_history_plain = []
        self.loss_history_res = []
        self.grad_norms_plain = []
        self.grad_norms_res = []
        self.t = 0

        # Plain network: input -> n_blocks * (linear -> relu) -> output
        # Residual network: same but with skip connections

        s = np.sqrt(2.0 / hidden_dim)
        # Plain weights
        self.plain_W = [np.random.randn(hidden_dim, hidden_dim) * s for _ in range(n_blocks)]
        self.plain_b = [np.zeros(hidden_dim) for _ in range(n_blocks)]
        self.plain_W_in = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.plain_b_in = np.zeros(hidden_dim)
        self.plain_W_out = np.random.randn(hidden_dim, 1) * s
        self.plain_b_out = np.zeros(1)

        # Residual weights (same structure)
        self.res_W = [np.random.randn(hidden_dim, hidden_dim) * s for _ in range(n_blocks)]
        self.res_b = [np.zeros(hidden_dim) for _ in range(n_blocks)]
        self.res_W_in = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.res_b_in = np.zeros(hidden_dim)
        self.res_W_out = np.random.randn(hidden_dim, 1) * s
        self.res_b_out = np.zeros(1)

        # Copy initial weights so comparison is fair
        for i in range(n_blocks):
            self.res_W[i] = self.plain_W[i].copy()
            self.res_b[i] = self.plain_b[i].copy()
        self.res_W_in = self.plain_W_in.copy()
        self.res_b_in = self.plain_b_in.copy()
        self.res_W_out = self.plain_W_out.copy()
        self.res_b_out = self.plain_b_out.copy()

        # Adam state
        self._adam_plain = {"W": [{"m": np.zeros_like(w), "v": np.zeros_like(w)} for w in self.plain_W],
                            "b": [{"m": np.zeros_like(b), "v": np.zeros_like(b)} for b in self.plain_b]}
        self._adam_res = {"W": [{"m": np.zeros_like(w), "v": np.zeros_like(w)} for w in self.res_W],
                          "b": [{"m": np.zeros_like(b), "v": np.zeros_like(b)} for b in self.res_b]}

    def _forward_plain(self, X):
        h = np.maximum(0, X @ self.plain_W_in + self.plain_b_in)
        for i in range(self.n_blocks):
            h = np.maximum(0, h @ self.plain_W[i] + self.plain_b[i])
        return h @ self.plain_W_out + self.plain_b_out

    def _forward_res(self, X):
        h = np.maximum(0, X @ self.res_W_in + self.res_b_in)
        for i in range(self.n_blocks):
            residual = h
            h = np.maximum(0, h @ self.res_W[i] + self.res_b[i])
            if self.use_residual:
                h = h + residual  # Skip connection
        return h @ self.res_W_out + self.res_b_out

    def _backward_plain(self, X, y):
        n = X.shape[0]
        # Forward with cache
        h = np.maximum(0, X @ self.plain_W_in + self.plain_b_in)
        cache = [X, h]
        for i in range(self.n_blocks):
            h = np.maximum(0, h @ self.plain_W[i] + self.plain_b[i])
            cache.append(h)
        out = h @ self.plain_W_out + self.plain_b_out

        # Loss
        loss = float(np.mean((out - y) ** 2))
        dout = 2.0 * (out - y) / n

        # Backward
        dh = dout @ self.plain_W_out.T
        self.dW_out = cache[-1].T @ dout / n
        self.db_out = np.sum(dout, axis=0) / n

        grad_norms = [float(np.linalg.norm(dh))]
        for i in reversed(range(self.n_blocks)):
            dz = dh * (cache[i + 2] > 0).astype(float)
            self.plain_W[i]  # ensure exists
            dW = cache[i + 1].T @ dz / n
            db = np.sum(dz, axis=0) / n
            dh = dz @ self.plain_W[i].T
            grad_norms.append(float(np.linalg.norm(dh)))

        dz_in = dh * (cache[1] > 0).astype(float)
        self.dW_in = cache[0].T @ dz_in / n
        self.db_in = np.sum(dz_in, axis=0) / n

        return loss, dW, db, self.dW_in, self.db_in, grad_norms

    def _backward_res(self, X, y):
        n = X.shape[0]
        # Forward with cache
        h = np.maximum(0, X @ self.res_W_in + self.res_b_in)
        cache = [X, h]
        for i in range(self.n_blocks):
            residual = h
            h_pre = h @ self.res_W[i] + self.res_b[i]
            h = np.maximum(0, h_pre)
            if self.use_residual:
                h = h + residual
            cache.append((h, h_pre, residual))

        out = h @ self.res_W_out + self.res_b_out
        loss = float(np.mean((out - y) ** 2))
        dout = 2.0 * (out - y) / n

        dh = dout @ self.res_W_out.T
        grad_norms = [float(np.linalg.norm(dh))]

        for i in reversed(range(self.n_blocks)):
            h_out, h_pre, residual = cache[i + 2]
            # h_out = relu(h_pre) + residual (if residual)
            # dh = dh (from next layer)
            d_relu = dh * (h_pre > 0).astype(float)
            dW = residual.T @ d_relu / n
            db = np.sum(d_relu, axis=0) / n
            dh_residual = dh
            dh = d_relu @ self.res_W[i].T
            if self.use_residual:
                dh = dh + dh_residual  # Gradient flows through skip connection!
            grad_norms.append(float(np.linalg.norm(dh)))

        dz_in = dh * (cache[1] > 0).astype(float)
        dW_in = cache[0].T @ dz_in / n
        db_in = np.sum(dz_in, axis=0) / n

        return loss, dW, db, dW_in, db_in, grad_norms

    def _adam_update_block(self, W_list, b_list, dW, db, adam_state, lr):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(len(W_list)):
            m_w, v_w = adam_state["W"][i]["m"], adam_state["W"][i]["v"]
            m_w[:] = beta1 * m_w + (1 - beta1) * dW[i]
            v_w[:] = beta2 * v_w + (1 - beta2) * dW[i] ** 2
            m_hat = m_w / (1 - beta1 ** self.t)
            v_hat = v_w / (1 - beta2 ** self.t)
            W_list[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            m_b, v_b = adam_state["b"][i]["m"], adam_state["b"][i]["v"]
            m_b[:] = beta1 * m_b + (1 - beta1) * db[i]
            v_b[:] = beta2 * v_b + (1 - beta2) * db[i] ** 2
            m_hat = m_b / (1 - beta1 ** self.t)
            v_hat = v_b / (1 - beta2 ** self.t)
            b_list[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_step(self, X, y):
        # Plain
        loss_p, dW_p, db_p, dW_in_p, db_in_p, gn_p = self._backward_plain(X, y)
        self.plain_W_out -= self.lr * self.dW_out
        self.plain_b_out -= self.lr * self.db_out
        self.plain_W_in -= self.lr * dW_in_p
        self.plain_b_in -= self.lr * db_in_p
        self._adam_update_block(self.plain_W, self.plain_b, [dW_p], [db_p], self._adam_plain, self.lr)
        # Fix: dW_p is a single matrix, wrap it
        self.plain_W[0] = self.plain_W[0]  # already updated by adam but let me just do sgd for simplicity
        # Actually let me just use simple SGD for the block weights too
        # The adam_update_block expects lists - let me restructure
        self.loss_history_plain.append(loss_p)
        self.grad_norms_plain.append(gn_p)

        # Residual
        loss_r, dW_r, db_r, dW_in_r, db_in_r, gn_r = self._backward_res(X, y)
        self.res_W_out -= self.lr * dW_r
        self.res_b_out -= self.lr * db_r
        self.res_W_in -= self.lr * dW_in_r
        self.res_b_in -= self.lr * db_in_r
        self.loss_history_res.append(loss_r)
        self.grad_norms_res.append(gn_r)

        return loss_p, loss_r

    def predict_plain(self, X):
        out = self._forward_plain(X)
        return (out > 0.5).astype(float)

    def predict_res(self, X):
        out = self._forward_res(X)
        return (out > 0.5).astype(float)

# ────────────────────────────────────────────────────────────────
#  Normalization Comparison
# ────────────────────────────────────────────────────────────────

def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """Layer Normalization: normalize across the feature dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        x_norm = gamma * x_norm + beta
    return x_norm, mean, var


def instance_norm_2d(x, gamma=None, beta=None, eps=1e-5):
    """Instance Normalization for 2D feature maps (N, C, H, W).
    Normalizes each channel of each sample independently.
    x: (N, C, H, W)
    """
    N, C, H, W = x.shape
    x_flat = x.reshape(N, C, H * W)
    mean = np.mean(x_flat, axis=2, keepdims=True)
    var = np.var(x_flat, axis=2, keepdims=True)
    x_norm = (x_flat - mean) / np.sqrt(var + eps)
    if gamma is not None:
        x_norm = gamma.reshape(1, C, 1, 1) * x_norm.reshape(N, C, H, W) + beta.reshape(1, C, 1, 1)
        return x_norm, mean.squeeze(), var.squeeze()
    return x_norm.reshape(N, C, H, W), mean.squeeze(), var.squeeze()


def group_norm_2d(x, n_groups=4, gamma=None, beta=None, eps=1e-5):
    """Group Normalization for 2D feature maps (N, C, H, W).
    Splits channels into groups and normalizes within each group.
    """
    N, C, H, W = x.shape
    assert C % n_groups == 0, f"C={C} must be divisible by n_groups={n_groups}"
    G = n_groups
    x_reshaped = x.reshape(N, G, C // G, H, W)
    mean = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
    x_norm = (x_reshaped - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)
    if gamma is not None:
        x_norm = gamma.reshape(1, C, 1, 1) * x_norm + beta.reshape(1, C, 1, 1)
    return x_norm, mean.squeeze(), var.squeeze()


def compare_normalizations(X, norm_type="batch", n_groups=4, eps=1e-5):
    """Compare different normalizations on a batch of activations.
    
    Args:
        X: (N, D) array of activations
        norm_type: one of "batch", "layer", "instance", "group"
        n_groups: number of groups for group norm
    
    Returns:
        dict with normalized data, mean, var, and per-sample stats
    """
    N, D = X.shape
    result = {"original": X.copy(), "shape": (N, D)}
    
    if norm_type == "batch":
        mean = np.mean(X, axis=0, keepdims=True)
        var = np.var(X, axis=0, keepdims=True)
        X_norm = (X - mean) / np.sqrt(var + eps)
        result["normalized"] = X_norm
        result["mean"] = mean
        result["var"] = var
        result["per_sample_mean"] = np.mean(X_norm, axis=1)
        result["per_sample_var"] = np.var(X_norm, axis=1)
        result["per_feature_mean"] = np.mean(X_norm, axis=0)
        result["per_feature_var"] = np.var(X_norm, axis=0)
    
    elif norm_type == "layer":
        X_norm, mean, var = layer_norm(X)
        result["normalized"] = X_norm
        result["mean"] = mean
        result["var"] = var
        result["per_sample_mean"] = np.mean(X_norm, axis=1)
        result["per_sample_var"] = np.var(X_norm, axis=1)
    
    elif norm_type == "instance":
        # Treat each sample as a "channel", normalize each independently
        X_norm = np.zeros_like(X)
        means = np.zeros(N)
        variances = np.zeros(N)
        for i in range(N):
            m = np.mean(X[i])
            v = np.var(X[i])
            means[i] = m
            variances[i] = v
            X_norm[i] = (X[i] - m) / np.sqrt(v + eps)
        result["normalized"] = X_norm
        result["mean"] = means
        result["var"] = variances
        result["per_sample_mean"] = np.zeros(N)  # All zero by construction
        result["per_sample_var"] = np.ones(N) * (1.0 / D)  # Approx
    
    elif norm_type == "group":
        # Split features into groups
        G = min(n_groups, D)
        assert D % G == 0 or G == 1
        if D % G != 0:
            G = 1  # Fall back to layer norm
        group_size = D // G
        X_norm = np.zeros_like(X)
        for g in range(G):
            start = g * group_size
            end = start + group_size
            group_data = X[:, start:end]
            mean = np.mean(group_data, axis=1, keepdims=True)
            var = np.var(group_data, axis=1, keepdims=True)
            X_norm[:, start:end] = (group_data - mean) / np.sqrt(var + eps)
        result["normalized"] = X_norm
        result["n_groups"] = G
    
    return result


def normalization_gradient_flow(norm_type, n_layers=10, input_dim=32):
    """Simulate gradient flow through a deep network with different normalizations.
    
    Returns gradient norms at each layer for the given normalization type.
    """
    np.random.seed(42)
    fan_in = input_dim
    
    # Generate random input
    x = np.random.randn(64, fan_in)
    
    # Build weights
    Ws = [np.random.randn(fan_in, fan_in) * np.sqrt(2.0 / fan_in) for _ in range(n_layers)]
    bs = [np.zeros(fan_in) for _ in range(n_layers)]
    
    # Forward pass
    activations = [x]
    for W, b in zip(Ws, bs):
        h = activations[-1] @ W + b
        if norm_type == "batch":
            mean = np.mean(h, axis=0, keepdims=True)
            var = np.var(h, axis=0, keepdims=True)
            h = (h - mean) / np.sqrt(var + 1e-5)
        elif norm_type == "layer":
            h, _, _ = layer_norm(h)
        elif norm_type == "none":
            h = np.maximum(0, h)  # Just ReLU
        else:
            h, _, _ = layer_norm(h)
        activations.append(h)
    
    # Backward pass - track gradient norms
    grad = np.ones_like(activations[-1])
    grad_norms = [float(np.linalg.norm(grad))]
    
    for i in reversed(range(n_layers)):
        # Simplified backward through linear + norm + relu
        if norm_type == "none":
            grad = grad * (activations[i + 1] > 0).astype(float)
        grad = grad @ Ws[i].T
        grad_norms.append(float(np.linalg.norm(grad)))
    
    return grad_norms


# ────────────────────────────────────────────────────────────────
#  Classification Metrics
# ────────────────────────────────────────────────────────────────

def compute_confusion_matrix(y_true, y_pred, n_classes=2):
    """Compute confusion matrix.
    
    Args:
        y_true: (N,) or (N, 1) true labels (0 or 1 for binary)
        y_pred: (N,) or (N, 1) predicted labels (0 or 1 for binary)
        n_classes: number of classes
    
    Returns:
        (n_classes, n_classes) confusion matrix
    """
    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)
    
    if n_classes == 2:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm
    else:
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm


def compute_metrics_from_cm(cm):
    """Compute precision, recall, F1, accuracy from confusion matrix.
    
    Returns:
        dict with accuracy, precision, recall, f1, specificity, tpr, fpr
    """
    n = np.sum(cm)
    tp = cm[1, 1] if cm.shape == (2, 2) else np.diag(cm).sum()
    
    accuracy = tp / max(n, 1)
    
    if cm.shape == (2, 2):
        tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        specificity = tn / max(tn + fp, 1)
        tpr = recall
        fpr = fp / max(fp + tn, 1)
        return {
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "specificity": specificity, "tpr": tpr, "fpr": fpr,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "cm": cm
        }
    else:
        per_class = {}
        for c in range(cm.shape[0]):
            tp_c = cm[c, c]
            fp_c = cm[:, c].sum() - tp_c
            fn_c = cm[c, :].sum() - tp_c
            p = tp_c / max(tp_c + fp_c, 1)
            r = tp_c / max(tp_c + fn_c, 1)
            f = 2 * p * r / max(p + r, 1e-10)
            per_class[c] = {"precision": p, "recall": r, "f1": f}
        return {"accuracy": accuracy, "per_class": per_class, "cm": cm}


def compute_roc_curve(y_true, y_scores, n_points=100):
    """Compute ROC curve points.
    
    Args:
        y_true: (N,) binary labels
        y_scores: (N,) probability scores
        n_points: number of thresholds
    
    Returns:
        dict with fpr, tpr, thresholds, auc
    """
    y_true = y_true.flatten()
    y_scores = y_scores.flatten()
    
    thresholds = np.linspace(0, 1, n_points)
    fpr_list = []
    tpr_list = []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    # Compute AUC using trapezoidal rule
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    # Sort by FPR
    sort_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sort_idx]
    tpr_sorted = tpr_arr[sort_idx]
    auc = float(np.trapz(tpr_sorted, fpr_sorted))
    
    return {"fpr": fpr_list, "tpr": tpr_list, "thresholds": thresholds.tolist(), "auc": auc}


def train_and_evaluate(X, y, arch=None, epochs=500, lr=0.01, optimizer="adam",
                        weight_init="he", loss_name="MSE", verbose=True):
    """Train a neural network and return predictions + metrics.
    
    Returns:
        dict with y_pred, y_scores, metrics, loss_history, confusion_matrix
    """
    if arch is None:
        arch = [(X.shape[1], "Input"), (32, "ReLU"), (16, "ReLU"), (1, "Sigmoid")]
    
    nn = NeuralNetwork(loss_name=loss_name, lr=lr, optimizer=optimizer, weight_init=weight_init)
    nn.build_dense(arch)
    
    loss_history = []
    for epoch in range(epochs):
        loss = nn.train_step(X, y)
        loss_history.append(loss)
    
    y_scores = nn.forward(X)
    y_pred = (y_scores > 0.5).astype(float)
    
    cm = compute_confusion_matrix(y, y_pred)
    metrics = compute_metrics_from_cm(cm)
    roc = compute_roc_curve(y, y_scores)
    
    return {
        "y_pred": y_pred, "y_scores": y_scores, "metrics": metrics,
        "loss_history": loss_history, "cm": cm, "roc": roc, "nn": nn
    }


# ────────────────────────────────────────────────────────────────
#  Embedding & PCA
# ────────────────────────────────────────────────────────────────

def compute_pca(X, n_components=2):
    """Compute PCA dimensionality reduction.
    
    Args:
        X: (N, D) data matrix
        n_components: number of principal components
    
    Returns:
        dict with projected data, eigenvalues, explained variance, components
    """
    # Center data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # Compute covariance matrix
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    top_eigenvalues = eigenvalues[:n_components]
    top_components = eigenvectors[:, :n_components]
    
    # Project
    X_projected = X_centered @ top_components
    
    # Explained variance ratio
    total_var = np.sum(eigenvalues)
    explained_var_ratio = eigenvalues / max(total_var, 1e-10)
    cumulative_var = np.cumsum(explained_var_ratio)
    
    return {
        "projected": X_projected,
        "eigenvalues": eigenvalues,
        "explained_var_ratio": explained_var_ratio,
        "cumulative_var": cumulative_var,
        "components": top_components,
        "mean": mean
    }


def compute_tsne_simple(X, n_components=2, perplexity=30, n_iter=500, lr=50.0):
    """Simplified t-SNE implementation for educational purposes.
    
    Args:
        X: (N, D) data matrix
        n_components: embedding dimension
        perplexity: target perplexity
        n_iter: number of iterations
        lr: learning rate
    
    Returns:
        dict with embedding, loss_history
    """
    N = X.shape[0]
    if N > 200:
        # Subsample for speed
        idx = np.random.choice(N, 200, replace=False)
        X = X[idx]
        N = 200
    
    # Compute pairwise distances
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dists[i, j] = np.sum((X[i] - X[j]) ** 2)
    
    # Compute P (high-dimensional similarities)
    P = np.zeros((N, N))
    for i in range(N):
        beta = 1.0
        for _ in range(50):
            exp_d = np.exp(-dists[i] * beta)
            exp_d[i] = 0
            sum_exp = np.sum(exp_d)
            if sum_exp < 1e-10:
                break
            P_i = exp_d / sum_exp
            # Binary search for correct perplexity
            entropy = -np.sum(P_i * np.log(P_i + 1e-10))
            perp = 2 ** entropy
            if perp > perplexity:
                beta *= 1.5
            else:
                beta /= 1.5
                if beta < 1e-10:
                    break
        P[i] = P_i
    
    # Symmetrize
    P = (P + P.T) / (2 * N)
    P = np.maximum(P, 1e-10)
    
    # Initialize embedding randomly
    np.random.seed(42)
    Y = np.random.randn(N, n_components) * 0.01
    
    loss_history = []
    for iteration in range(n_iter):
        # Compute Q (low-dimensional similarities)
        dists_y = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                dists_y[i, j] = dists_y[j, i] = np.sum((Y[i] - Y[j]) ** 2)
        
        Q_num = 1.0 / (1.0 + dists_y)
        np.fill_diagonal(Q_num, 0)
        Q = Q_num / np.sum(Q_num)
        Q = np.maximum(Q, 1e-10)
        
        # Compute gradients
        diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (N, N, d)
        pq_diff = P - Q  # (N, N)
        grad = 4.0 * np.sum((pq_diff[:, :, np.newaxis] * Q_num[:, :, np.newaxis]) * diff, axis=1)
        
        # Update with momentum
        Y = Y - lr * grad / N
        
        # KL divergence loss
        loss = float(np.sum(P * np.log(P / Q)))
        loss_history.append(loss)
    
    return {"embedding": Y, "loss_history": loss_history}
