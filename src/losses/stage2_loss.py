# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.stage1_loss import Stage1Loss, Stage1LossWeights


@dataclass
class Stage2LossWeights:
    lambda_flow: float = 1.0
    lambda_aux: float = 0.1
    lambda_psi: float = 1.0
    lambda_jaw: float = 10.0
    lambda_reg_psi: float = 1e-4
    lambda_reg_jaw_roll: float = 100.0
    lambda_reg_jaw_close: float = 10.0


class Stage2Loss(nn.Module):
    def __init__(self, weights: Stage2LossWeights):
        super().__init__()
        self.w = weights
        self.aux_loss = Stage1Loss(
            Stage1LossWeights(
                lambda_psi=weights.lambda_psi,
                lambda_jaw=weights.lambda_jaw,
                lambda_reg_psi=weights.lambda_reg_psi,
                lambda_reg_jaw_roll=weights.lambda_reg_jaw_roll,
                lambda_reg_jaw_close=weights.lambda_reg_jaw_close,
            )
        )

    def forward(
        self,
        flow_pred: torch.Tensor,
        flow_tgt: torch.Tensor,
        psi_pred: torch.Tensor,
        jaw_pred: torch.Tensor,
        psi_tgt: torch.Tensor,
        jaw_tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flow_loss = F.mse_loss(flow_pred.float(), flow_tgt.float())
        aux_total, aux_dict = self.aux_loss(
            psi_pred.float(),
            jaw_pred.float(),
            psi_tgt.float(),
            jaw_tgt.float(),
        )
        total = self.w.lambda_flow * flow_loss + self.w.lambda_aux * aux_total

        out = {
            "flow": (self.w.lambda_flow * flow_loss).detach(),
            "aux_total": (self.w.lambda_aux * aux_total).detach(),
            "psi_mse": aux_dict["psi_mse"],
            "jaw_mse": aux_dict["jaw_mse"],
            "reg_psi": aux_dict["reg_psi"],
            "reg_jaw_roll": aux_dict["reg_jaw_roll"],
            "reg_jaw_close": aux_dict["reg_jaw_close"],
            "total": total.detach(),
        }
        return total, out