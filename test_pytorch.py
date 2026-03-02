import torch
import torch.nn as nn

# シードを固定
torch.manual_seed(0)

# 3 nodes -> 2 nodes 全結合層
fc = nn.Linear(3, 2)

# 線形変換 tensor型に変える必要がある
x = torch.tensor([1., 2., 3.])

u = fc(x)
print(u)

# 非線形変換
import torch.nn.functional as F

h = F.relu(u)
print(h)

# 目的関数
t = torch.tensor([[1.], [3.]]) # 目標値
y = torch.tensor([[2.], [4.]]) # 予想値

# 平均二乗誤差
print(F.mse_loss(y, t))


# 入力値と目標値をまとめる
dataset = torch.utils.data.TensorDataset(x, t)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = (4, 4)
        self.fc2 = (4, 3)

    # 順伝播
    def forward(self):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        return h

net = Net()
# 最急降下法
optimizer = torch.optim.SGD(net.parameters(), lr=0.01) # lr は学習係数