"""
Post-hoc confidence calibration via temperature scaling.
Learns a single scalar T on a held-out calibration set such that:
  calibrated_probs = softmax(logits / T)
T > 1: model was overconfident (typical for deep CNNs)
T < 1: model was underconfident (rare)

Usage:
    scaler = TemperatureScaler()
    scaler.fit(val_logits_np, val_labels_np)
    calibrated_probs = scaler.calibrate_probs(raw_logits_tensor)
    scaler.save('models/temperature.json')
    scaler2 = TemperatureScaler.load('models/temperature.json')
"""
import numpy as np
import json
import torch
from scipy.optimize import minimize


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0          # identity (uncalibrated) by default
        self.fitted = False

    def fit(self, logits_np: np.ndarray, labels_np: np.ndarray) -> float:
        """
        Optimise T to minimise NLL on calibration set.
        Args:
            logits_np: (N, C) raw pre-softmax logits
            labels_np: (N,) integer class labels
        Returns:
            Optimal temperature T
        """
        assert logits_np.ndim == 2, "logits must be (N, C)"
        assert labels_np.ndim == 1, "labels must be (N,)"
        assert len(logits_np) == len(labels_np)

        def nll(T_arr):
            T       = float(T_arr[0])
            scaled  = logits_np / T
            # numerically stable softmax
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            exp_s   = np.exp(shifted)
            probs   = exp_s / exp_s.sum(axis=1, keepdims=True)
            correct = probs[np.arange(len(labels_np)), labels_np]
            return -np.log(correct + 1e-8).mean()

        result   = minimize(nll, x0=[1.5], method='L-BFGS-B',
                            bounds=[(0.05, 10.0)],
                            options={'ftol': 1e-10, 'gtol': 1e-8})
        self.T   = float(result.x[0])
        self.fitted = True

        nll_before = nll([1.0])
        nll_after  = nll([self.T])
        print(f"Temperature scaling complete:")
        print(f"  Optimal T   : {self.T:.4f}")
        print(f"  NLL before  : {nll_before:.4f}")
        print(f"  NLL after   : {nll_after:.4f}")
        print(f"  NLL reduction: {(nll_before - nll_after):.4f}")
        return self.T

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Divide logits by T. Apply softmax after this."""
        return logits / self.T

    def calibrate_probs(self, logits_tensor: torch.Tensor) -> torch.Tensor:
        """
        Takes raw logit tensor, returns calibrated probability tensor.
        Safe to call on CPU or GPU.
        """
        with torch.no_grad():
            calibrated = logits_tensor / self.T
            return torch.softmax(calibrated, dim=-1)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({'temperature': self.T, 'fitted': self.fitted}, f, indent=2)
        print(f"[SAVED] Temperature scaler → {path}  (T={self.T:.4f})")

    @classmethod
    def load(cls, path: str) -> 'TemperatureScaler':
        with open(path) as f:
            data = json.load(f)
        scaler       = cls()
        scaler.T     = data['temperature']
        scaler.fitted = data.get('fitted', True)
        print(f"[LOADED] Temperature scaler from {path}  (T={scaler.T:.4f})")
        return scaler

    def __repr__(self):
        status = f"T={self.T:.4f}" if self.fitted else "unfitted"
        return f"TemperatureScaler({status})"
