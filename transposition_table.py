from collections import OrderedDict

# 置換表のフラグ定数
TT_EXACT = 0  # 正確な値
TT_LOWER = 1  # 評価値 >= Value (Betaカット発生)
TT_UPPER = 2  # 評価値 <= Value (全ての指し手が Alpha 以下)

class TranspositionTable:
    def __init__(self, max_size=1 << 21): # 2,097,152 局面
        self.table = {} # key -> (key, depth, value, chain, bound, age)
        self.mask = max_size - 1
        
        # 置換表のフラグ定数
        self.TT_EXACT = 0  # 正確な値
        self.TT_LOWER = 1  # 評価値 >= Value (Betaカット発生)
        self.TT_UPPER = 2  # 評価値 <= Value (全ての指し手が Alpha 以下)

        # 置換表のインデックス
        self.KEY   = 0
        self.DEPTH = 1
        self.VALUE = 2
        self.CHAIN = 3
        self.BOUND = 4
        self.AGE   = 5
    
    def get_idx(self, key):
        return key & self.mask
    
    def store(self, key, depth, value, chain, bound, age):
        idx = self.get_idx(key)
        tt = self.table.get(idx, None)
        if tt is None:
            self.table[idx] = (key, depth, value, chain, bound, age) # 追加
        elif tt[self.KEY] != key: # 別の局面が衝突している場合は、更新
            self.table[idx] = (key, depth, value, chain, bound, age) # 更新
        elif tt[self.KEY] == key and tt[self.DEPTH] < depth: # 同一局面でより深い場合は更新
            self.table[idx] = (key, depth, value, chain, bound, age) # 更新

    def lookup(self, key):
        idx = self.get_idx(key)
        tt = self.table.get(idx, None)
        # 同一局面だったら値を返す
        if tt is not None and tt[self.KEY] == key:
            return tt
        return None

class TT_valueOnly:
    def __init__(self, max_size=1 << 21): # 2,097,152 局面
        self.stored_num = 0
        self.max_size = max_size
        self.table = OrderedDict()  # key -> (depth, value, chain)
    
    def store(self, key, depth, value, chain):
        self.table[key] = (depth, value, chain) # 更新、または追加
        self.table.move_to_end(key) # 最後尾に移動
        self.stored_num += 1

        if self.stored_num >= self.max_size:
            self.table.popitem(last=False)  # 最も古く使われてないものを捨てる
            self.stored_num -= 1

    def lookup(self, key):
        v = self.table.get(key)
        if v is None:
            return None
        self.table.move_to_end(key)  # 参照されたので最後尾に移動
        depth, value, chain = v
        return {'depth': depth, 'value': value, 'chain': chain}
