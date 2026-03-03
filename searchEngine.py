import sys
import time
import traceback
import cshogi
import usi_server

class SearchEngine(usi_server.USIEngine):
    def __init__(self, engine_name):
        super().__init__(engine_name)

        self.inf = 1.0
        self.max_depth = 4 # 読む手の深さ
    
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
            best_move = None
            best_value = -self.inf if self.board.turn == cshogi.BLACK else self.inf
        
            for move in moves:
                self.board.push(move)
                value = self.alpha_beta(self.max_depth - 1, -self.inf, self.inf)
                self.board.pop()
            
                if self.board.turn == cshogi.BLACK:
                    if value >= best_value:
                        best_value = value
                        best_move = move
                else:
                    if value <= best_value:
                        best_value = value
                        best_move = move
            
            self.info(current_depth, int((time.perf_counter() - start) * 1000), self.recover_score(value), [cshogi.move_to_usi(best_move)])
                    
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