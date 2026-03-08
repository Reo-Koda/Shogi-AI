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
        return v

class ValueNet_useResMulti(nn.Module):
    def __init__(self, stage_blocks=(6, 6), stage_channels=(128, 256)):
        super().__init__()
        assert len(stage_blocks) == len(stage_channels), "stage_blocks と stage_channels の要素数の不一致"

        c0 = stage_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(119, c0, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(),
        )

        layers = []
        c_prev = c0
        for i, (nb, c) in enumerate(zip(stage_blocks, stage_channels)):
            if i > 0:
                layers += [
                    nn.Conv2d(c_prev, c, kernel_size=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                ]
            layers += [ResBlock(c) for _ in range(nb)]
            c_prev = c
        
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c_prev, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).flatten(1)      # (B,C)
        v = self.head(h).squeeze(-1)     # (B,)
        return v

class ValueNet(nn.Module):
    def __init__(self, channels=256):
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
        return v

class PolicyNet(nn.Module):
    """
    入力:  x (B,119,9,9)
    出力:  policy_logits (B,K,9,9)  ※Kはmove-planes数（固定）
          （必要なら合法手マスク後に softmax して確率にする）
    """
    def __init__(self, policy_planes: int = 137):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(119, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        # policy head: (B,128,9,9) -> (B,K,9,9)
        self.policy_head = nn.Conv2d(128, policy_planes, kernel_size=1, bias=True)

    def forward(self, x, legal_mask: torch.Tensor | None = None):
        """
        legal_mask: (B,K,9,9) のbool/0-1テンソル（True/1が合法）
                   与えられた場合は非合法手を大負値でマスクして返す
        """
        h = self.body(x)                       # (B,128,9,9)
        logits = self.policy_head(h)           # (B,K,9,9)

        if legal_mask is not None:
            # boolでない場合も受ける
            legal_mask = legal_mask.to(dtype=torch.bool)
            logits = logits.masked_fill(~legal_mask, -1e9)

        return logits # 合法手だけのスコア群を返す

    @staticmethod
    def logits_to_policy(logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,K,9,9)
        return: policy probs (B,K,9,9)
        """
        B, K, H, W = logits.shape
        p = F.softmax(logits.view(B, K * H * W), dim=1).view(B, K, H, W)
        return p