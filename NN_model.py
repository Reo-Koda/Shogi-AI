import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_target(score, game_result, T=600.0, alpha=0.7):
    score_norm = math.tanh(score / T)
    return alpha * score_norm + (1 - alpha) * game_result

# def output_target(score, scaling=200.0):
#     return int(scaling * math.log((1 + score) / (1 - score)))

def output_target(score_norm, T=600.0):
    eps = 1e-9
    score_norm_clipped = max(min(score_norm, 1.0 - eps), -1.0 + eps)
    return int(T * math.atanh(score_norm_clipped))

class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)

class ValueNet_useRes(nn.Module):
    def __init__(self, blocks: int = 8, channels: int = 128):
        super().__init__()
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

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
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