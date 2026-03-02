import os
import sys
import time
import traceback
import cshogi
import torch
import numpy as np
import usi_server
import NN_model

class NNEngine(usi_server.USIEngine):
    def __init__(self, engine_name, paramater_path):
        super().__init__(engine_name)

        self.device = "cpu"

        ckpt = torch.load(paramater_path, map_location=self.device) # パラメータのダウンロード

        self.model = NN_model.ValueNet().to(self.device)            # モデルの生成
        self.model.load_state_dict(ckpt["model"])                   # パラメータのインストール
        self.model.eval()                                           # 推論モードに切り替え

        self.inf = 1.0
        self.max_depth = 3 # 読む手の深さ
    
    def eval(self):
        if self.board.is_game_over():
            return -self.inf if self.board.turn == cshogi.BLACK else self.inf
        
        FEATURES_NUM = 119
        features = np.zeros((FEATURES_NUM, 9, 9), dtype=np.float32)
        self.board.piece_planes(features)

        x = torch.from_numpy(features).to(self.device) # shape: (119,9,9)

        with torch.no_grad():
            y = self.model(x)
        
        return y.item()
    
    def alpha_beta(self, depth, alpha, beta):
        self.nodes += 1
        if depth == 0 or self.board.is_game_over():
            return self.eval()
        
        moves = list(self.board.legal_moves)

        if self.board.turn == cshogi.BLACK: # 先手（最大化）
            v = -self.inf
            for move in moves:
                self.board.push(move)
                v = max(v, self.alpha_beta(depth - 1, alpha, beta))
                self.board.pop()
                alpha = max(alpha, v)
                if beta <= alpha:
                    break # Betaカット
            return v
        else: # 後手（最小化）
            v = self.inf
            for move in moves:
                self.board.push(move)
                v = min(v, self.alpha_beta(depth - 1, alpha, beta))
                self.board.pop()
                beta = min(beta, v)
                if beta <= alpha:
                    break # Alphaカット
            return v
    
    def select_best_move(self, moves):
        """
        最も評価値の高い手を選択する
        """
        self.nodes = 0
        start = time.perf_counter()
        for current_depth in range(1, self.max_depth):
            current_best_move = None
            best_value = -self.inf if self.board.turn == cshogi.BLACK else self.inf
        
            for move in moves:
                self.board.push(move)
                value = self.alpha_beta(self.max_depth - 1, -self.inf, self.inf)
                self.board.pop()
            
                if self.board.turn == cshogi.BLACK:
                    if value >= best_value:
                        best_value = value
                        current_best_move = move
                else:
                    if value <= best_value:
                        best_value = value
                        current_best_move = move
            
            best_move = current_best_move
            self.info(current_depth, int((time.perf_counter() - start) * 1000), NN_model.output_target(value), [cshogi.move_to_usi(best_move)])
                    
        return cshogi.move_to_usi(best_move)
    
    def think(self):
        try:
            moves = list(self.board.legal_moves)

            if not moves:
                self.send("bestmove resign")
                self.thinking = False
                return

            # alpha-beta 法
            move = self.select_best_move(moves)

            self.send(f"bestmove {move}")
            self.thinking = False
        except Exception:
            traceback.print_exc(file=sys.stderr)
            self.send("bestmove resign")  # 最低限プロトコル応答
        finally:
            self.thinking = False

if __name__ == "__main__":
    # エンジンの始動
    ckpt = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    engine = NNEngine(ckpt, f"../../pram/ckpt-{ckpt}.pt") # 実行ファイルからの相対パス
    engine.loop()
