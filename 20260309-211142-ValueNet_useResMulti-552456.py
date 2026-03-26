import os
import sys
import cshogi
import torch
import numpy as np
import math
from searchEngine import SearchEngine
import NN_model

class NNEngine(SearchEngine):
    def __init__(self, engine_name, paramater_path, inf=float("inf")):
        super().__init__(engine_name, inf)

        self.device = "cpu"

        ckpt = torch.load(paramater_path, map_location=self.device)  # パラメータのダウンロード

        # 勝敗のみで学習したモデル（alpha=0.0）
        self.model = NN_model.ValueNet_useResMulti().to(self.device) # モデルの生成
        self.model.load_state_dict(ckpt["model"])                    # パラメータのインストール
        self.model.eval()                                            # 推論モードに切り替え

        FEATURES_NUM = 119
        self.features = np.zeros((FEATURES_NUM, 9, 9), dtype=np.float32)
    
    def eval(self):
        isBlack = (self.board.turn == cshogi.BLACK)
        
        self.board.piece_planes(self.features) if isBlack else self.board.piece_planes_rotate(self.features)

        x = torch.from_numpy(self.features).to(self.device) # shape: (119,9,9)
        x = x.unsqueeze(0) # shape: (1,119,9,9)
        
        with torch.no_grad():
            y = self.model(x)
        
        eval_y = y.item() if isBlack else -y.item() # 先手の評価値はそのまま、後手の評価値は符号反転
        
        eps = 1e-14
        border = 1.0
        border_m = -border + eps
        border_p = border - eps
        # 評価値を0.0～1.0の範囲に収める
        if eval_y < border_m:
            return (border_m + 1.0) / 2.0
        elif border_p < eval_y:
            return (border_p + 1.0) / 2.0
        else:
            return (eval_y + 1.0) / 2.0
    
    def recover_score(self, score, a=600.0):
        isBlack = (self.board.turn == cshogi.BLACK)

        if isBlack and 0.0 < score < 1.0:
            return int(-a * math.log((1 / score) - 1))
        elif not isBlack and 0.0 < score < 1.0:
            return -int(-a * math.log((1 / score) - 1))
        else:
            return score

if __name__ == "__main__":
    # エンジンの始動
    ckpt = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    engine = NNEngine(ckpt, f"../../pram/ckpt-{ckpt}.pt") # 実行ファイルからの相対パス
    engine.loop()
