import sys
import time
import traceback
import cshogi
import usi_server
from transposition_table import TT_valueOnly

class SearchEngine(usi_server.USIEngine):
    def __init__(self, engine_name, inf):
        super().__init__(engine_name)

        self.inf = inf
        self.max_depth = 3 # 読む手の深さ
        self.table = TT_valueOnly() # 局面のキャッシュ
    
    def get_value_chain(self, depth, alpha, beta):
        self.nodes += 1
        key = self.board.zobrist_hash()
        scores = self.table.lookup(key)
        if scores is not None and scores['depth'] >= depth:
            value = scores['value']
            child_chain = scores['chain']
            self.hits += 1
        else:
            # value, child_chain = self.min_max(depth - 1)
            value, child_chain = self.alpha_beta(depth - 1, alpha, beta)
            self.table.store(key, depth, value, child_chain)
        return value, child_chain
        
    def select_best_move(self, moves):
        self.nodes = 1 # 探索局面数の初期化
        self.hits  = 0 # キャッシュヒットした回数の初期化
        curr_depth = self.max_depth
        best_chain = [] # 最も良い読み筋
        best_value = -self.inf if self.board.turn == cshogi.BLACK else self.inf
        alpha = -self.inf
        beta  =  self.inf

        start = time.perf_counter()
        for move in moves:
            self.board.push(move)
            # value, child_chain = self.get_value_chain(curr_depth, alpha=alpha, beta=beta)
            # value, child_chain = self.min_max(curr_depth - 1)
            value, child_chain = self.alpha_beta(curr_depth - 1, alpha, beta)
            self.board.pop()

            if self.board.turn == cshogi.BLACK:
                # 最善手の更新
                if best_value < value:
                    best_value = value
                    best_chain = [cshogi.move_to_usi(move)] + child_chain
                # alpha値の更新
                if alpha < best_value:
                    alpha = best_value
                # betaカット
                if alpha >= beta:
                    break
            else:
                # 最善手の更新
                if best_value > value:
                    best_value = value
                    best_chain = [cshogi.move_to_usi(move)] + child_chain
                # beta値の更新
                if beta > best_value:
                    beta = best_value
                # alphaカット
                if alpha >= beta:
                    break

            passed_time = time.perf_counter() - start # 秒
            elapsed_time = int(passed_time * 1000) # ミリ秒
            # 探索した手の情報
            self.info(
                depth=curr_depth,
                time=elapsed_time,
                nps=int(self.nodes / passed_time),
                cp=self.recover_score(value),
                pv=[cshogi.move_to_usi(move)] + child_chain,
                hashfull=int(self.hits / self.nodes * 1000),
                currmove=cshogi.move_to_usi(move)
            )
        # 最善手の情報
        self.info(
            depth=curr_depth,
            time=elapsed_time,
            nps=int(self.nodes / passed_time),
            cp=self.recover_score(best_value),
            pv=best_chain,
            hashfull=int(self.hits / self.nodes * 1000)
        )
        return best_chain[0]
    
    def min_max(self, depth):
        # 1. 終了判定（ゲーム終了または深さ制限）
        if depth == 0 or self.board.is_game_over():
            return self.eval(), [] # 盤面を評価して値を返す
        
        moves = list(self.board.legal_moves)
        isBlack = (self.board.turn == cshogi.BLACK)
        if isBlack:
            # 先手：最大値を追求
            best_score = -self.inf
            for move in moves:
                self.board.push(move)
                score, child_chain = self.get_value_chain(depth, alpha=-self.inf, beta=self.inf)
                # score, child_chain = self.min_max(depth - 1)
                if best_score < score:
                    best_score = score
                    best_chain = [cshogi.move_to_usi(move)] + child_chain
                self.board.pop()
            return best_score, best_chain
        else:
            # 後手：最小値を追求
            best_score = self.inf
            for move in moves:
                self.board.push(move)
                score, child_chain = self.get_value_chain(depth, alpha=-self.inf, beta=self.inf)
                # score, child_chain = self.min_max(depth - 1)
                if best_score > score:
                    best_score = score
                    best_chain = [cshogi.move_to_usi(move)] + child_chain
                self.board.pop()
            return best_score, best_chain
    
    def alpha_beta_test(self, depth, alpha, beta):
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
    
    def alpha_beta(self, depth, alpha, beta):
        # 1. 終了判定（ゲーム終了または深さ制限）
        if depth == 0 or self.board.is_game_over():
            return self.eval(), [] # 盤面を評価して値を返す
        
        moves = list(self.board.legal_moves)
        isBlack = (self.board.turn == cshogi.BLACK)
        if isBlack:
            # 先手：最大値を追求
            best_score = -self.inf
            for move in moves:
                # 指し手の評価
                self.board.push(move)
                # score, child_chain = self.get_value_chain(depth, alpha, beta)
                # score, child_chain = self.min_max(depth - 1)
                score, child_chain = self.alpha_beta(depth - 1, alpha, beta)
                self.board.pop()
                # 最善手の更新
                if best_score < score:
                    best_score = score
                    best_chain = [cshogi.move_to_usi(move)] + child_chain
                # alpha値の更新
                if alpha < best_score:
                    alpha = best_score
                # betaカット
                if alpha >= beta:
                    break
            return best_score, best_chain
        else:
            # 後手：最小値を追求
            best_score = self.inf
            for move in moves:
                # 指し手の評価
                self.board.push(move)
                # score, child_chain = self.get_value_chain(depth, alpha, beta)
                # score, child_chain = self.min_max(depth - 1)
                score, child_chain = self.alpha_beta(depth - 1, alpha, beta)
                self.board.pop()
                # 最善手の更新
                if best_score > score:
                    best_score = score
                    best_chain = [cshogi.move_to_usi(move)] + child_chain
                # beta値の更新
                if beta > best_score:
                    beta = best_score
                # alphaカット
                if alpha >= beta:
                    break
            return best_score, best_chain

    def think(self):
        try:
            moves = list(self.board.legal_moves)

            if not moves:
                self.send("bestmove resign")
                self.thinking = False
                return
            
            move = self.select_best_move(moves)

            self.send(f"bestmove {move}")
        except Exception:
            traceback.print_exc(file=sys.stderr)
            self.send("bestmove resign")  # 最低限プロトコル応答
        finally:
            self.thinking = False