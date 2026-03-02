import os
import numpy as np
import cshogi
import json
from pathlib import Path

files = os.listdir("./kifu_data/bin")
file_num = len(files)
for pos in range(2):
    psv = np.memmap(f"./kifu_data/bin/{files[pos]}", dtype=cshogi.PackedSfenValue, mode="r")
    file_name = os.path.splitext(files[pos])[0]
    n = len(psv)
    board = cshogi.Board()
    with Path(f"./kifu_data/jsonl/{file_name}.jsonl").open("w", encoding="utf-8") as f:
        for i in range(n):
            rec = psv[i]

            # 盤面
            board.set_psfen(np.asarray(rec["sfen"]))
            # 指し手
            move16_psv = int(rec["move"])
            move = board.move_from_psv(move16_psv)
            bestmove_usi = cshogi.move_to_usi(move) if move != 0 else None
            # 評価値
            score = int(rec["score"]) if "score" in rec.dtype.names else None
            # 結果
            game_result = int(rec["game_result"]) if "game_result" in rec.dtype.names else None
            # 手数
            game_ply = int(rec["gamePly"]) if "gamePly" in rec.dtype.names else None

            obj = {
                "index": int(i),
                "sfen": board.sfen(),
                "bestmove": bestmove_usi,
                "score": score,
                "game_result": game_result,
                "gamePly": game_ply,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"\r{i+1} / {n} end", end="", flush=True) if (i+1) % 1e4 == 0 or i+1 == n else None
    print(f"\n{pos+1} / {file_num} completed")