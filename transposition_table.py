from collections import OrderedDict

# 置換表のフラグ定数
TT_EXACT = 0  # 正確な値
TT_LOWER = 1  # 評価値 >= Value (Betaカット発生)
TT_UPPER = 2  # 評価値 <= Value (全ての指し手が Alpha 以下)

class TranspositionTable:
    def __init__(self, max_size=1 << 21): # 2,097,152 局面
        self.table = {}
        self.stored_num = 0
        self.max_size = max_size
        self.table = OrderedDict()  # key -> (depth, value, chain)
    
    def store(self, key, depth, value, chain, bound):
        self.table[key] = (depth, value, chain, bound) # 更新、または追加
        self.table.move_to_end(key) # 最後尾に移動
        self.stored_num += 1

        if self.stored_num >= self.max_size:
            self.table.popitem(last=False)  # 最も古く使われてないものを捨てる

    def lookup(self, key):
        v = self.table.get(key)
        if v is None:
            return None
        self.table.move_to_end(key)  # 参照されたので最後尾に移動
        depth, value, chain, bound = v
        return {'depth': depth, 'value': value, 'chain': chain, 'bound': bound}

class TT_valueOnly:
    def __init__(self, max_size=1 << 21): # 2,097,152 局面
        self.table = {}
        self.stored_num = 0
        self.max_size = max_size
        self.table = OrderedDict()  # key -> (depth, value, chain)
    
    def store(self, key, depth, value, chain):
        self.table[key] = (depth, value, chain) # 更新、または追加
        self.table.move_to_end(key) # 最後尾に移動
        self.stored_num += 1

        if self.stored_num >= self.max_size:
            self.table.popitem(last=False)  # 最も古く使われてないものを捨てる

    def lookup(self, key):
        v = self.table.get(key)
        if v is None:
            return None
        self.table.move_to_end(key)  # 参照されたので最後尾に移動
        depth, value, chain = v
        return {'depth': depth, 'value': value, 'chain': chain}
