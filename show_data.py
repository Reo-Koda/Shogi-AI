import os
# import json
# import cshogi
# import numpy as np

files = os.listdir("./kifu_data/jsonl")

print(files)
with open(f"./kifu_data/jsonl/{files[0]}","r") as f:
    for i in range(10):
        data_str = f.readline()
        print(data_str)
        # data = json.loads(data_str)
        # sfen = data["sfen"]
        # board = cshogi.Board(sfen)
        # print(board)

        # FEATURES_NUM = 119
        # features = np.zeros((FEATURES_NUM, 9, 9), dtype=np.float32)
        # print(features)
        # if board.turn == cshogi.BLACK:
        #     board.piece_planes(features)
        # else:
        #     board.piece_planes_rotate(features)
        # print(features)