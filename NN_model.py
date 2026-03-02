import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormScore(nn.Module):
    def __init__(self, alpha=0.7, T=600.0, scaling=200.0):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.scaling = scaling
    
    def norm_score(self, score, game_result):
        score_norm = math.tanh(score / self.T)
        return self.alpha * score_norm + (1 - self.alpha) * game_result

    def recover_score_log(self, score):
        return int(self.scaling * math.log((1 + score) / (1 - score)))

    def recover_score_atanh(self, score_norm):
        eps = 1e-9
        score_norm_clipped = max(min(score_norm, 1.0 - eps), -1.0 + eps)
        return int(self.T * math.atanh(score_norm_clipped))

class ResBlock(NormScore):
    def __init__(self, c: int, alpha=0.7, T=600.0, scaling=200.0):
        super().__init__(alpha, T, scaling)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)

class ValueNet_useRes(NormScore):
    def __init__(self, blocks: int = 8, channels: int = 128, alpha=0.7, T=600.0, scaling=200.0):
        super().__init__(alpha, T, scaling)
        self.stem = nn.Sequential(
            nn.Conv2d(119, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).flatten(1)      # (B,C)
        v = self.head(h).squeeze(-1)     # (B,)
        return torch.tanh(v)

class ValueNet(NormScore):
    def __init__(self, alpha=0.7, T=600.0, scaling=200.0):
        super().__init__(alpha, T, scaling)
        self.body = nn.Sequential(
            nn.Conv2d(119, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B,119,9,9)
        h = self.body(x).squeeze(-1).squeeze(-1) # (B,128)
        v = self.head(h).squeeze(-1)             # (B,)
        return torch.tanh(v)