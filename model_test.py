import os
import random
import time
import torch
import argparse
import kifu_dataset
import load_model

if __name__ == "__main__":
    # 引数の設定
    parser = argparse.ArgumentParser(description='モデル性能テストの設定')

    # 引数の追加
    parser.add_argument('--pramPath', type=str, default="./pram/ckpt.pt", help='保存されたパラメータのファイルへのパス')
    parser.add_argument('--iter', type=float, default=1e3, help='反復回数')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='使用デバイス')
    parser.add_argument('--seed', type=int, default=100, help='シード値')
    parser.add_argument('--model_module', type=str, default='NN_model', help='使用するモジュール名')
    parser.add_argument('--model', type=str, default='valueNet', help='使用するモデルクラス名')

    args = parser.parse_args()

    # 定数の設定
    iter         = int(args.iter)
    seed         = args.seed
    device       = args.device
    pramPath     = args.pramPath
    model_name   = args.model
    model_module = args.model_module

    # モジュール、クラスの存在確認
    ModelClass = load_model.load_model_class(model_module, model_name)

    # シード値の設定
    random.seed(seed)

    # 検証証データの作成
    files = os.listdir("./kifu_data/bin")
    random.shuffle(files)
    all_bin_paths = [f"./kifu_data/bin/{fn}" for fn in files[:3]]
    dataset = kifu_dataset.MultiPSVDataset(all_bin_paths)
    print("dataset end")

    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    print(f"device : {device}")

    model = ModelClass().to(device)
    ckpt = torch.load(pramPath, map_location=device)

    model.load_state_dict(ckpt["model"])

    model.eval() # 推論モードに切り替え

    start = time.time()

    delta_list = []
    cnt = 0
    for i in range(iter):
        # 検証局面をランダムに選定
        idx = random.randint(0, len(dataset))
        # 盤面と基準スコアを取得
        x, stnd_y = dataset[idx]
        # 入力値にバッチ次元を追加して4次元にする
        x = x.unsqueeze(0)
        # 推論スコアを取得
        with torch.no_grad():
            eval_y = model(x).item()
        
        if abs(stnd_y) > 3000 and cnt < 10:
            print(f"idx : {idx}")
            print(f"推論スコア : {eval_y}")
            print(f"基準スコア : {stnd_y}")
            cnt += 1
        # スコアの差を保存
        delta_list.append(abs(eval_y - stnd_y))

    end = time.time()
    print(f"評価値差の平均 : {sum(delta_list) / len(delta_list)}")
    print(f"経過時間 : {end - start:.6f}sec")