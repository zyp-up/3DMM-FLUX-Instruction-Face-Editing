# -*- coding: utf-8 -*-
"""
Stage1 Loss: Conditional DECA Encoder 的训练损失.

我们 (Stage1) 与原版 DECA 训练的差异:
  - 原版 DECA 是自监督 (用 landmark + photometric + id_loss), 没有参数真值 MSE.
  - 我们有 target image 的离线提取参数 (psi_Tgt, jaw_Tgt), 所以使用纯监督 MSE.

Loss 组成 (与论文 + 原版 DECA 正则项兼容):
    L = lambda_psi  * MSE(psi_pred, psi_Tgt)
      + lambda_jaw  * MSE(jaw_pred, jaw_Tgt)
      + lambda_reg_psi       * ||psi_pred||^2 / 2          # 对齐原版 expression_reg
      + lambda_reg_jaw_roll  * (jaw_pred[:, 2])^2 / 2      # 对齐原版 reg_jawpose_roll
      + lambda_reg_jaw_close * ReLU(-jaw_pred[:, 0])^2 / 2 # 对齐原版 reg_jawpose_close

默认权重参考:
    lambda_psi            = 1.0
    lambda_jaw            = 10.0   # psi / jaw 数值尺度差约 10 倍, 同时原版 DECA 对 jaw 权重也偏大
    lambda_reg_psi        = 1e-4   # 原版 DECA config.py 的 reg_exp
    lambda_reg_jaw_roll   = 100.0  # 原版 DECA trainer.py 的 reg_jawpose_roll
    lambda_reg_jaw_close  = 10.0   # 原版 DECA trainer.py 的 reg_jawpose_close

返回:
    total_loss: 标量, 用来 backward
    loss_dict : 每一项 (已乘权重) 的字典, 方便打 log / tensorboard

约定:
    psi_pred / psi_Tgt : (B, 50)  — FLAME expression 参数
    jaw_pred / jaw_Tgt : (B, 3)   — pose[:, 3:6], axis-angle 形式
                                     [0] = 张嘴 (yaw 分量, 正值张嘴)
                                     [1] = 左右摆动 (pitch)
                                     [2] = 左右歪斜 (roll, 希望接近 0)
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
    """Conditional DECA Encoder 的监督 + 正则复合 loss."""

    def __init__(self, weights: Optional[Stage1LossWeights] = None):
        super().__init__()
        self.w = weights if weights is not None else Stage1LossWeights()

    @classmethod
    def from_config(cls, cfg: Dict) -> "Stage1Loss":
        """从 yaml 解析出来的 dict 构造 loss (支持部分字段覆盖)."""
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
            total_loss: 标量 tensor
            loss_dict : {"psi_mse", "jaw_mse", "reg_psi", "reg_jaw_roll",
                         "reg_jaw_close", "total"}, 值均为已乘权重后的标量
        """
        assert psi_pred.shape == psi_Tgt.shape, \
            f"psi shape mismatch: {psi_pred.shape} vs {psi_Tgt.shape}"
        assert jaw_pred.shape == jaw_Tgt.shape, \
            f"jaw shape mismatch: {jaw_pred.shape} vs {jaw_Tgt.shape}"
        assert jaw_pred.shape[-1] == 3, \
            f"jaw last dim must be 3, got {jaw_pred.shape}"

        # ---- 1) 主监督: MSE ----
        psi_mse = F.mse_loss(psi_pred, psi_Tgt)
        jaw_mse = F.mse_loss(jaw_pred, jaw_Tgt)

        # ---- 2) 正则项 (沿用原版 DECA 的形式, 做 batch-mean 方便跨 batch_size 迁移) ----
        # 原版是 sum/2, 我们这里用 mean 使 loss 对 batch_size 不敏感.
        reg_psi = 0.5 * psi_pred.pow(2).mean()
        reg_jaw_roll = 0.5 * jaw_pred[:, 2].pow(2).mean()
        reg_jaw_close = 0.5 * F.relu(-jaw_pred[:, 0]).pow(2).mean()

        # ---- 3) 加权求和 ----
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
# 自测: 直接 python -m src.losses.stage1_loss 跑一下形状/数值是否正常
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