from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


class ExitHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        return self.fc(x)


class TokenExitHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, use_cls: bool = True):
        super().__init__()
        self.use_cls = use_cls
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        if self.use_cls:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        x = self.norm(x)
        return self.fc(x)


class ConfidenceHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class MultiExitWrapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        exit_heads: nn.ModuleList,
        feature_extractor,
        conf_head_type: str = "maxprob",
        conf_heads: Optional[nn.ModuleList] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.exit_heads = exit_heads
        self.feature_extractor = feature_extractor
        self.conf_head_type = conf_head_type
        self.conf_heads = conf_heads

    def forward(self, x: torch.Tensor):
        features = self.feature_extractor(x)
        logits = [head(f) for head, f in zip(self.exit_heads, features)]
        return logits

    def confidences(self, logits: List[torch.Tensor], features: Optional[List[torch.Tensor]] = None):
        if self.conf_head_type == "mlp" and self.conf_heads is not None and features is not None:
            confs = []
            for f, head in zip(features, self.conf_heads):
                if f.dim() == 4:
                    pooled = f.mean(dim=(2, 3))
                else:
                    pooled = f.mean(dim=1)
                confs.append(head(pooled.detach()))
            return confs
        # default max softmax
        return [torch.softmax(l, dim=1).max(dim=1).values for l in logits]
