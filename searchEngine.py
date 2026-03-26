import sys
import time
import traceback
import cshogi
import usi_server
from transposition_table import TranspositionTable

class SearchEngine(usi_server.USIEngine):
    def __init__(self, engine_name, inf):
        super().__init__(engine_name)

        self.inf = inf
        self.mate_value = 30000
        self.mate_border = 1000
        self.max_depth = 3 # 読む手の深さ
        self.extend_depth = 3 # 延長して読むときの手の深さ
        self.table = TranspositionTable() # 局面のキャッシュ

        self.MINUTES = 60 # 秒
    
    def get_value_chain(self, depth, alpha, beta, extend=False):
        self.nodes += 1
        key = self.board.zobrist_hash()
        cache = self.table.lookup(key)
        # キャッシュがあって、かつ、キャッシュの深さが十分であるか、詰みが見えている場合は、キャッシュから値を取得する
        if cache is not None and (cache[self.table.DEPTH] >= depth or cache[self.table.VALUE] < -self.mate_value + self.mate_border or self.mate_value - self.mate_border < cache[self.table.VALUE]):
            value = cache[self.table.VALUE]
            child_chain = cache[self.table.CHAIN]
            bound = cache[self.table.BOUND]
            self.hits += 1
            if bound == self.table.TT_EXACT: # 正確な値
                return value, child_chain
            elif bound == self.table.TT_LOWER and value > alpha: # 評価値 >= Value (Betaカット発生)
                alpha = value
            elif bound == self.table.TT_UPPER and value < beta: # 評価値 <= Value (全ての指し手が Alpha 以下)
                beta = value
            if alpha >= beta:
                return value, child_chain

        value, child_chain, bound = self.alpha_beta(depth - 1, alpha, beta, extend=extend)
        self.table.store(key, len(child_chain)+1, value, child_chain, bound)
        self.currmove(child_chain[0]) if child_chain and depth > 1 else None
        return value, child_chain
    
    def order_moves(self, set_moves, moves):
        key = self.board.zobrist_hash()
        cache = self.table.lookup(key)
        if cache is not None:
            pre_best = cache[self.table.CHAIN][0]
        # 最善手を先頭に移動させて、最初に探索するようにする
        if cache is not None and pre_best in set_moves:
            pre_idx = moves.index(pre_best)
            moves[0], moves[pre_idx] = moves[pre_idx], moves[0]
        return moves
        
    def select_best_move(self):
        self.nodes = 0 # 探索局面数の初期化
        self.hits  = 0 # キャッシュヒットした回数の初期化
        isBlack = (self.board.turn == cshogi.BLACK)
        # curr_depth = self.max_depth
        best_chain = [] # 最も良い読み筋
        skip_moves = set() # 合法じゃない手を記録する集合
        move_value = {} # 指し手の評価値の辞書
        
        # 探索する指し手の生成
        is_safe = not self.board.is_check() # 王手がかかっていないときは高速モードで生成する
        if is_safe:
            moves = list(self.board.pseudo_legal_moves) # 疑似合法手（自ら王手になる手も含む）
        else:
            moves = list(self.board.legal_moves) # 完全合法手
        
        # 前回探索時の最善手を取得
        set_moves = set(moves)
        moves = self.order_moves(set_moves, moves)

        start = time.perf_counter()
        # 反復深化探索
        for curr_depth in range(1, self.max_depth + 1):
            best_value = -self.inf if isBlack else self.inf
            alpha = -self.inf
            beta  =  self.inf
            is_skip = False # skip_movesが更新されたかどうか
            # 前回ループの評価値が良い順に探索する
            if curr_depth > 1:
                moves = sorted(move_value, key=move_value.get, reverse=True) if isBlack else sorted(move_value, key=move_value.get) # 評価値の良い順にソート
            for move in moves:
                # 自ら王手になる手はスキップ
                if is_safe and not self.board.pseudo_legal_move_is_legal(move):
                    skip_moves.add(move)
                    is_skip = True
                    continue

                # 指し手の評価
                self.board.push(move)
                value, child_chain = self.get_value_chain(curr_depth, alpha=alpha, beta=beta)
                # value, child_chain = self.min_max(curr_depth - 1)
                # value, child_chain = self.alpha_beta(curr_depth - 1, alpha, beta)
                self.board.pop()
                move_value[move] = value # 指し手の評価値を記録                
                
                # 詰みが見えていたら、詰みまでの手数分評価値を調整する（詰みの手数が少ない順に指すため）
                if isBlack and value > self.mate_value - self.mate_border:
                    value -= 1
                elif not isBlack and value < -self.mate_value + self.mate_border:
                    value += 1

                if isBlack:
                    # 最善手の更新
                    if best_value < value:
                        best_value = value
                        best_chain = [move] + child_chain
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
                        best_chain = [move] + child_chain
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
                    pv=[move] + child_chain,
                    hashfull=int(self.hits / self.nodes * 1000)
                )
                # ストップがかかったら、探索を終了する
                if self.stop_flag:
                    break
            # 最善手の情報
            self.info(
                depth=curr_depth,
                time=elapsed_time,
                nps=int(self.nodes / passed_time),
                cp=self.recover_score(best_value),
                pv=best_chain,
                hashfull=int(self.hits / self.nodes * 1000)
            )
            # if passed_time > 5*self.MINUTES: # 5分を超えたら探索を終了する
            #     break
            # ストップがかかったら、探索を終了する
            if self.stop_flag:
                break
            
            # スキップする手を合法手のリストから削除する
            if is_skip:
                moves = list(set_moves - skip_moves)

        # 合法手がない（詰み）場合は投了
        if self.nodes == 0:
            return "resign"
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
    
    def alpha_beta(self, depth, alpha, beta, extend=False):
        isBlack = (self.board.turn == cshogi.BLACK)
        draw_reason = self.board.is_draw()
        # 千日手
        if draw_reason == cshogi.REPETITION_DRAW:
            return 0, [], self.table.TT_EXACT
        # 連続王手の千日手で手番側の勝ち
        elif draw_reason == cshogi.REPETITION_WIN:
            return self.mate_value if isBlack else -self.mate_value, [], self.table.TT_EXACT
        # 連続王手の千日手で手番側の負け
        elif draw_reason == cshogi.REPETITION_LOSE:
            return -self.mate_value if isBlack else self.mate_value, [], self.table.TT_EXACT
        
        # 入玉の判定で手番側の勝ち
        if self.board.is_nyugyoku():
            return self.mate_value if isBlack else -self.mate_value, [], self.table.TT_EXACT
        # 対局の終了判定
        if self.board.is_game_over():
            # 手番側の負け
            return -self.mate_value if isBlack else self.mate_value, [], self.table.TT_EXACT

        # 安全かどうかの判定
        is_safe = not self.board.is_check()

        # 葉ノード到達
        if depth == 0:
            last_move = self.board.peek()
            # 王手がかかっていて、かつ、まだ延長していないときは、探索を2手分延長する
            if not is_safe and not extend:
                score, child_chain = self.get_value_chain(self.extend_depth, alpha, beta, extend=True)
            # 駒を取る手かつ、まだ延長していないときは、探索を2手分延長する
            elif is_safe and cshogi.move_cap(last_move) and not extend:
                score, child_chain = self.get_value_chain(self.extend_depth, alpha, beta, extend=True)
            else:
                return self.eval(), [], self.table.TT_EXACT # 盤面を評価して値を返す
            return score, child_chain, self.table.TT_EXACT
        
        # 探索する指し手の生成
        if is_safe: # 王手がかかっていないときは高速モードで生成する
            moves = list(self.board.pseudo_legal_moves) # 疑似合法手（自ら王手になる手も含む）
        else:
            moves = list(self.board.legal_moves) # 完全合法手
            
        # 前回探索時の最善手を先に探索するようにする
        set_moves = set(moves)
        moves = self.order_moves(set_moves, moves)
        
        if isBlack:
            # 先手：最大値を追求
            best_score = -self.inf
            bound = self.table.TT_EXACT
            for move in moves:
                # 自ら王手になる手はスキップ
                if is_safe and not self.board.pseudo_legal_move_is_legal(move):
                    continue
                # 指し手の評価
                self.board.push(move)
                score, child_chain = self.get_value_chain(depth, alpha, beta, extend=extend)
                # score, child_chain = self.min_max(depth - 1)
                # score, child_chain = self.alpha_beta(depth - 1, alpha, beta)
                self.board.pop()
                # 詰みが見えていたら、詰みまでの手数分評価値を調整する（詰みの手数が少ない順に指すため）
                if score > self.mate_value - self.mate_border:
                    score -= 1
                # 最善手の更新
                if best_score < score:
                    best_score = score
                    best_chain = [move] + child_chain
                # alpha値の更新
                if alpha < best_score:
                    alpha = best_score
                # betaカット
                if alpha >= beta:
                    bound = self.table.TT_LOWER
                    break
            return best_score, best_chain, bound
        else:
            # 後手：最小値を追求
            best_score = self.inf
            bound = self.table.TT_EXACT
            for move in moves:
                # 自ら王手になる手はスキップ
                if is_safe and not self.board.pseudo_legal_move_is_legal(move):
                    continue
                # 指し手の評価
                self.board.push(move)
                score, child_chain = self.get_value_chain(depth, alpha, beta, extend=extend)
                # score, child_chain = self.min_max(depth - 1)
                # score, child_chain = self.alpha_beta(depth - 1, alpha, beta)
                self.board.pop()
                # 詰みが見えていたら、詰みまでの手数分評価値を調整する（詰みの手数が少ない順に指すため）
                if score < -self.mate_value + self.mate_border:
                    score += 1
                # 最善手の更新
                if best_score > score:
                    best_score = score
                    best_chain = [move] + child_chain
                # beta値の更新
                if beta > best_score:
                    beta = best_score
                # alphaカット
                if alpha >= beta:
                    bound = self.table.TT_UPPER
                    break
            return best_score, best_chain, bound

    def think(self):
        try:
            move = self.select_best_move()
            if move != "resign":
                move = cshogi.move_to_usi(move)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            move = "resign"
        finally:
            self.send(f"bestmove {move}")
            self.thinking = False