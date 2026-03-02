
# 置換表のフラグ定数
TT_EXACT = 0  # 正確な値
TT_LOWER = 1  # 評価値 >= Value (Betaカット発生)
TT_UPPER = 2  # 評価値 <= Value (全ての指し手が Alpha 以下)

class TranspositionTable:
    def __init__(self):
        self.table = {}
    
    def store(self, key, depth, value, flag, best_move):
        # より深い探索結果がある場合は上書きしない
        if key in self.table and self.table[key]['depth'] > depth:
            return
        self.table[key] = {
            'depth': depth,
            'value': value,
            'flag': flag,
            'best_move': best_move
        }

    def lookup(self, key):
        return self.table.get(key)

class TT_valueOnly(TranspositionTable):
    def __init__(self):
        super().__init__()
    
    def store(self, key, value, flag):
        if key in self.table:
            return
        self.table[key] = {
            'value': value,
            'flag': flag,
        }
