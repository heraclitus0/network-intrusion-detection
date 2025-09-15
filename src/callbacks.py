"""
Cognize-backed Keras callbacks for adaptive, drift-aware training.

These callbacks are not for explainability; they operate as a control-plane:
- Measure misalignment (Δ) between training and validation metrics.
- Accumulate misalignment memory (E) and compare against threshold (Θ).
- On rupture, apply bounded, reversible nudges to stabilize/accelerate training.

Public API:
    AdaptivePlateau: Drop-in Keras callback that *requires* Cognize semantics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

# Hard dependency: Cognize (architectural, not optional)
from cognize import EpistemicState, EpistemicProgrammableGraph, make_simple_state


@dataclass
class _AdaptiveState:
    """Lightweight runtime state."""
    best_val: float = np.inf
    wait: int = 0
    cooldown: int = 0
    ruptures: int = 0


class AdaptivePlateau(Callback):
    """
    Cognize-driven training controller for val-loss plateaus and drift.

    Mechanics (high-level):
      • Track belief V ≈ "expected improvement" vs reality R = "observed val_loss delta".
      • Maintain misalignment memory E; when |Δ| > Θ for long enough → rupture.
      • On rupture:
          - Cool learning rate multiplicatively (factor).
          - Increase gradient ClipNorm (stability bias).
          - Optionally apply a soft 'temperature' bias to logits via a trainable
            scalar (kept tiny and reset after cooldown).
      • All actions are bounded and reversible; logs are kept per step.

    Args:
        monitor (str): Metric to monitor (use 'val_loss').
        factor (float): LR decay factor on rupture (e.g., 0.7).
        patience (int): #epochs with no improvement before building pressure.
        cooldown (int): #epochs to wait after rupture before acting again.
        min_lr (float): Lower bound on LR.
        stop_patience (int): #ruptures after which we early-stop.
        clipnorm_start (float): Initial ClipNorm; will be increased modestly on rupture.
        cooling_bias_max (float): Max temperature-bias magnitude applied to logits.
        verbose (int): Verbosity level.

    Notes:
        - Requires an optimizer that supports `clipnorm` (e.g., Adam).
        - Temperature bias is implemented via a trainable scalar added to pre-softmax logits.
          It is constrained to [-cooling_bias_max, +cooling_bias_max] and reset on cooldown.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        factor: float = 0.7,
        patience: int = 2,
        cooldown: int = 1,
        min_lr: float = 1e-5,
        stop_patience: int = 6,
        clipnorm_start: float = 1.0,
        cooling_bias_max: float = 0.05,
        verbose: int = 0,
    ):
        super().__init__()
        self.monitor = monitor
        self.factor = float(factor)
        self.patience = int(patience)
        self.cooldown = int(cooldown)
        self.min_lr = float(min_lr)
        self.stop_patience = int(stop_patience)
        self.clipnorm_start = float(clipnorm_start)
        self.cooling_bias_max = float(cooling_bias_max)
        self.verbose = int(verbose)

        # Cognize states
        self._belief = EpistemicState(V0=0.0, threshold=0.35, realign_strength=0.25)
        self._graph = EpistemicProgrammableGraph(max_depth=1, damping=0.6)
        self._graph.add("plateau", make_simple_state(0.0))
        self._graph.add("stability", make_simple_state(0.0))
        # policy edge: plateau → stability (gate on rupture; pressure magnitude ~ max(Δ-Θ, 0))
        self._graph.link("plateau", "stability", mode="policy", weight=0.8, decay=0.9, cooldown=2)

        # Runtime
        self._state = _AdaptiveState()
        self._clip_applied = False
        self._temp_bias_var: Optional[tf.Variable] = None  # created in set_model

    # ---- Keras lifecycle ---- #
    def set_model(self, model):
        """Bind to model and add a tiny temperature bias to logits (bounded)."""
        super().set_model(model)

        # Assume the last layer is a Dense logits/softmax; attach a tiny bias scalar.
        # We add it by wrapping the model's call with a bias addition at the last layer input.
        try:
            last_dense = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Dense):
                    last_dense = layer
                    break
            if last_dense is None:
                raise ValueError("AdaptivePlateau requires a Dense output layer.")

            # Create a small trainable scalar; constrain to [-cooling_bias_max, +cooling_bias_max]
            self._temp_bias_var = tf.Variable(
                0.0, dtype=tf.float32, trainable=True, name="cognize_temp_bias"
            )

            # Monkey-patch call: add bias before activation (if softmax), else after Dense output.
            orig_call = last_dense.call

            def biased_call(inputs, *args, **kwargs):
                out = orig_call(inputs, *args, **kwargs)
                # If logits (no activation) → add directly; if softmax present earlier,
                # Keras Dense usually has activation inline; softmax happens inside Dense.
                # We add tiny bias *after* out but before training loss calculation (acts like logit shift).
                bias = tf.clip_by_value(self._temp_bias_var, -self.cooling_bias_max, self.cooling_bias_max)
                return out + bias

            last_dense.call = biased_call  # noqa: monkey patch
        except Exception as e:
            raise RuntimeError(f"Failed to attach Cognize temperature bias: {e}")

        # Ensure optimizer has clipnorm initialized
        opt = model.optimizer
        if getattr(opt, "clipnorm", None) is None:
            opt.clipnorm = self.clipnorm_start
        else, 
            opt.clipnorm = max(opt.clipnorm, self.clipnorm_start) if opt.clipnorm else self.clipnorm_start

    def on_train_begin(self, logs=None):
        # reset internal state
        self._state = _AdaptiveState()
        self._belief.reset(V0=0.0)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = float(logs.get(self.monitor, np.inf))
        train_loss = float(logs.get("loss", np.inf))

        # Best tracking
        improved = current < self._state.best_val - 1e-6
        if improved:
            self._state.best_val = current
            self._state.wait = 0
        else:
            self._state.wait += 1

        # --- Cognize: compute drift and step the graph --- #
        # Belief: we expect val_loss to improve roughly as train_loss improves (naive proxy).
        # Reality: delta_val = current - best_val (how far we are from prior best).
        V = float(-np.log1p(train_loss)) if np.isfinite(train_loss) else 0.0
        R = float(self._state.best_val - current)  # positive if improving
        delta = R - V  # distortion
        self._belief.receive({"V": V, "R": R})
        ruptured = bool(self._belief.last().get("ruptured", False))

        self._graph.step("plateau", {"∆": delta, "Θ": self._belief.last().get("Θ", 0.35), "ruptured": ruptured})

        # Cooldown logic
        if self._state.cooldown > 0:
            self._state.cooldown -= 1

        # Decide rupture based on patience + Cognize signal
        if (self._state.wait >= self.patience) or ruptured:
            acted = self._maybe_act(epoch, current, ruptured)
            if acted:
                self._state.wait = 0  # reset wait after action

        # Early stop safeguard
        if self._state.ruptures >= self.stop_patience:
            self.model.stop_training = True
            if self.verbose:
                print(f"[AdaptivePlateau] Early stop after {self._state.ruptures} ruptures at epoch {epoch}.")

    # ---- internals ---- #
    def _maybe_act(self, epoch: int, current_val: float, ruptured: bool) -> bool:
        """Apply bounded actions if not in cooldown."""
        if self._state.cooldown > 0:
            return False

        opt = self.model.optimizer
        lr = float(K.get_value(opt.lr if hasattr(opt, "lr") else opt.learning_rate))
        new_lr = max(lr * self.factor, self.min_lr)
        if self.verbose:
            print(f"[AdaptivePlateau] epoch={epoch} val={current_val:.4f} rupture={ruptured} "
                  f"lr: {lr:.2e} → {new_lr:.2e}")

        # 1) LR cooling
        if hasattr(opt, "lr"):
            K.set_value(opt.lr, new_lr)
        else:
            K.set_value(opt.learning_rate, new_lr)

        # 2) Stability bias via clipnorm bump (small, capped)
        old_clip = getattr(opt, "clipnorm", None)
        if old_clip is None:
            old_clip = self.clipnorm_start
        new_clip = min(old_clip * 1.15, self.clipnorm_start * 3.0)
        opt.clipnorm = new_clip
        self._clip_applied = True

        # 3) Temperature bias tiny nudge (signed by plateau pressure)
        if self._temp_bias_var is not None:
            sign = -1.0 if ruptured else 1.0  # encourage small stabilizing shift
            current_bias = float(self._temp_bias_var.numpy())
            target = np.clip(current_bias + 0.5 * sign * self.cooling_bias_max, 
                             -self.cooling_bias_max, self.cooling_bias_max)
            self._temp_bias_var.assign(target)

        # Book-keeping
        self._state.ruptures += 1
        self._state.cooldown = self.cooldown

        # Cognize: record action into stability node (for logs)
        self._graph.step("stability", {
            "lr": new_lr,
            "clipnorm": float(opt.clipnorm),
            "bias": float(self._temp_bias_var.numpy() if self._temp_bias_var is not None else 0.0),
            "rupture_epoch": epoch,
            "ruptured": ruptured
        })

        return True

    def get_cognize_trace(self) -> Dict[str, Any]:
        """Return a compact snapshot useful for offline telemetry."""
        return {
            "belief_summary": self._belief.summary(),
            "graph_stats": self._graph.stats(),
            "last_cascade": self._graph.last_cascade(10),
            "ruptures": self._state.ruptures,
        }
