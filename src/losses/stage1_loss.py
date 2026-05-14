# -*- coding: utf-8 -*-
"""Stage1 supervised loss for the Conditional DECA Encoder.

Target DECA parameters are pre-extracted offline, so Stage1 uses supervised MSE
plus DECA-style regularization:

    L = lambda_psi  * MSE(psi_pred, psi_Tgt)
      + lambda_jaw  * MSE(jaw_pred, jaw_Tgt)
      + lambda_reg_psi       * ||psi_pred||^2 / 2
      + lambda_reg_jaw_roll  * jaw_pred[:, 2]^2 / 2
      + lambda_reg_jaw_close * ReLU(-jaw_pred[:, 0])^2 / 2
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Stage1LossWeights:
    lambda_psi: float = 1.0
    lambda_jaw: float = 10.0
    lambda_reg_psi: float = 1e-4
    lambda_reg_jaw_roll: float = 100.0
    lambda_reg_jaw_close: float = 10.0


class Stage1Loss(nn.Module):
    """Supervised MSE loss with DECA-style regularization."""

    def __init__(self, weights: Optional[Stage1LossWeights] = None):
        super().__init__()
        self.w = weights if weights is not None else Stage1LossWeights()

    @classmethod
    def from_config(cls, cfg: Dict) -> "Stage1Loss":
        """Create loss weights from a config dict."""
        w = Stage1LossWeights()
        for k in w.__dataclass_fields__.keys():
            if k in cfg and cfg[k] is not None:
                setattr(w, k, float(cfg[k]))
        return cls(w)

    def forward(
        self,
        psi_pred: torch.Tensor,
        jaw_pred: torch.Tensor,
        psi_Tgt: torch.Tensor,
        jaw_Tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            psi_pred: (B, 50)
            jaw_pred: (B, 3)
            psi_Tgt : (B, 50)
            jaw_Tgt : (B, 3)

        Returns:
            total_loss: scalar tensor
            loss_dict : {"psi_mse", "jaw_mse", "reg_psi", "reg_jaw_roll",
                         "reg_jaw_close", "total"} with weighted scalar values
        """
        assert psi_pred.shape == psi_Tgt.shape, \
            f"psi shape mismatch: {psi_pred.shape} vs {psi_Tgt.shape}"
        assert jaw_pred.shape == jaw_Tgt.shape, \
            f"jaw shape mismatch: {jaw_pred.shape} vs {jaw_Tgt.shape}"
        assert jaw_pred.shape[-1] == 3, \
            f"jaw last dim must be 3, got {jaw_pred.shape}"

        # ---- 1) Supervised MSE ----
        psi_mse = F.mse_loss(psi_pred, psi_Tgt)
        jaw_mse = F.mse_loss(jaw_pred, jaw_Tgt)

        # ---- 2) DECA-style regularization, normalized by batch size ----
        reg_psi = 0.5 * psi_pred.pow(2).mean()
        reg_jaw_roll = 0.5 * jaw_pred[:, 2].pow(2).mean()
        reg_jaw_close = 0.5 * F.relu(-jaw_pred[:, 0]).pow(2).mean()

        # ---- 3) Weighted sum ----
        w = self.w
        l_psi_mse = w.lambda_psi * psi_mse
        l_jaw_mse = w.lambda_jaw * jaw_mse
        l_reg_psi = w.lambda_reg_psi * reg_psi
        l_reg_jaw_roll = w.lambda_reg_jaw_roll * reg_jaw_roll
        l_reg_jaw_close = w.lambda_reg_jaw_close * reg_jaw_close

        total = (
            l_psi_mse
            + l_jaw_mse
            + l_reg_psi
            + l_reg_jaw_roll
            + l_reg_jaw_close
        )

        loss_dict = {
            "psi_mse": l_psi_mse.detach(),
            "jaw_mse": l_jaw_mse.detach(),
            "reg_psi": l_reg_psi.detach(),
            "reg_jaw_roll": l_reg_jaw_roll.detach(),
            "reg_jaw_close": l_reg_jaw_close.detach(),
            "total": total.detach(),
        }
        return total, loss_dict

    def extra_repr(self) -> str:
        w = self.w
        return (
            f"lambda_psi={w.lambda_psi}, lambda_jaw={w.lambda_jaw}, "
            f"lambda_reg_psi={w.lambda_reg_psi}, "
            f"lambda_reg_jaw_roll={w.lambda_reg_jaw_roll}, "
            f"lambda_reg_jaw_close={w.lambda_reg_jaw_close}"
        )


# ---------------------------------------------------------------------------
# Smoke test: python -m src.losses.stage1_loss
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    psi_pred = torch.randn(B, 50, requires_grad=True)
    jaw_pred = torch.randn(B, 3, requires_grad=True)
    psi_Tgt = torch.randn(B, 50)
    jaw_Tgt = torch.randn(B, 3)

    loss_fn = Stage1Loss()
    print(loss_fn)
    total, d = loss_fn(psi_pred, jaw_pred, psi_Tgt, jaw_Tgt)
    total.backward()

    for k, v in d.items():
        print(f"  {k:14s}: {v.item():.6f}")
    print(f"  psi.grad norm = {psi_pred.grad.norm().item():.6f}")
    print(f"  jaw.grad norm = {jaw_pred.grad.norm().item():.6f}")
