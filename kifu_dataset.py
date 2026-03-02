import numpy as np
import cshogi
import torch
from torch.utils.data import Dataset
from bisect import bisect_right
from tqdm import tqdm
import NN_model

_WORKER_BOARD = None
_WORKER_MEMMAPS = None

# データセット用のクラス
class MultiPSVDataset(Dataset):
    def __init__(self, bin_paths):
        self.bin_paths = list(bin_paths)

        # 長さ計算だけは安全にやる（memmapは一時的に作ってすぐ捨てる）
        lens = []
        for p in tqdm(self.bin_paths, leave=False):
            mm = np.memmap(p, dtype=cshogi.PackedSfenValue, mode="r")
            lens.append(len(mm))
            del mm
        print("memmap end", flush=True)
        self.lens = lens
        self.cum = np.cumsum(self.lens, dtype=np.int64)  # e.g. [len0, len0+len1, ...]
        self.total_len = int(self.cum[-1]) if len(self.cum) else 0
    
    def __len__(self):
        return self.total_len
    
    # 局面データのファイルを跨ったインデックス管理
    def _locate(self, index):
        file_id = bisect_right(self.cum, index)
        prev = 0 if file_id == 0 else int(self.cum[file_id - 1])
        local = index - prev
        return file_id, local
    
    def _ensure_worker_state(self):
        global _WORKER_BOARD, _WORKER_MEMMAPS
        if _WORKER_BOARD is None:
            _WORKER_BOARD = cshogi.Board()

        if _WORKER_MEMMAPS is None:
            # ここはワーカープロセス内で初回だけ実行される
            _WORKER_MEMMAPS = [
                np.memmap(p, dtype=cshogi.PackedSfenValue, mode="r")
                for p in self.bin_paths
            ]
    
    def __getitem__(self, index):
        self._ensure_worker_state()
        
        file_id, local = self._locate(index)
        kifu_data = _WORKER_MEMMAPS[file_id][local]
        sfen = kifu_data["sfen"]
        score = float(kifu_data["score"])
        game_result = float(kifu_data["game_result"])

        # スコアの再定義
        target = NN_model.calc_target(score, game_result, alpha=1.0)

        _WORKER_BOARD.set_psfen(sfen)

        FEATURES_NUM = 119
        features = np.zeros((FEATURES_NUM, 9, 9), dtype=np.float32)
        _WORKER_BOARD.piece_planes(features)

        x = torch.from_numpy(features) # shape: (119,9,9)
        y = torch.tensor(target, dtype=torch.float32)
        return x, y