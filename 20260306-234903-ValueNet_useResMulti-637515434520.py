import os
import sys
import cshogi
import torch
import numpy as np
from searchEngine import SearchEngine
import NN_model

class NNEngine(SearchEngine):
    def __init__(self, engine_name, paramater_path, inf=float('inf')):
        super().__init__(engine_name, inf)

        self.device = "cpu"

        ckpt = torch.load(paramater_path, map_location=self.device)  # パラメータのダウンロード

        self.model = NN_model.ValueNet_useResMulti().to(self.device) # モデルの生成
        self.model.load_state_dict(ckpt["model"])                    # パラメータのインストール
        self.model.eval()                                            # 推論モードに切り替え
    
    def eval(self):
        if self.board.is_game_over():
            return -self.inf if self.board.turn == cshogi.BLACK else self.inf
        
        FEATURES_NUM = 119
        features = np.zeros((FEATURES_NUM, 9, 9), dtype=np.float32)
        self.board.piece_planes(features)

        x = torch.from_numpy(features).to(self.device) # shape: (119,9,9)
        x = x.unsqueeze(0) # shape: (1,119,9,9)
        
        with torch.no_grad():
            y = self.model(x)
        
        return y.item()
    
    def recover_score(self, score):
        return int(score)

if __name__ == "__main__":
    # エンジンの始動
    ckpt = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    engine = NNEngine(ckpt, f"../../pram/ckpt-{ckpt}.pt") # 実行ファイルからの相対パス
    engine.loop()
