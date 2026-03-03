import os
import time
import random
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, random_split
import kifu_dataset
from save_pram import save_pram
import argparse
from collections import deque
from tqdm import tqdm
import load_model
from gmail.log_mail import notify_result

if __name__ == "__main__":
    # 引数の設定
    parser = argparse.ArgumentParser(description='モデル学習の設定')

    # 引数の追加
    parser.add_argument('--isContinue', action='store_true', help='学習の途中から進めるかどうかを決定')
    parser.add_argument('--pramPath', type=str, default="./pram/ckpt.pt", help='保存されたパラメータのファイルへのパス')
    parser.add_argument('--savePath', type=str, default='default', help='保存するパラメータのファイルへのパス')
    parser.add_argument('--lr', type=float, default=1e-3, help='学習率')
    parser.add_argument('--iter', type=float, default=1e2, help='学習回数')
    parser.add_argument('--trainLimit', type=float, default=1e6, help='学習する局面数の上限')
    parser.add_argument('--valLimit', type=float, default=1e5, help='検証する局面数の上限')
    parser.add_argument('--batch_size', type=int, default=1024, help='一回のバッチ学習で学習する局面数')
    parser.add_argument('--file_num', type=int, default=25, help='使用するファイル数')
    parser.add_argument('--trainRatio', type=float, default=0.9, help='学習データの割合')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='使用デバイス')
    parser.add_argument('--seed', type=int, default=100, help='シード値')
    parser.add_argument('--model_module', type=str, default='NN_model', help='使用するモジュール名')
    parser.add_argument('--model', type=str, default='valueNet', help='使用するモデルクラス名')
    parser.add_argument('--patience', type=int, default=3, help='終了条件')
    parser.add_argument('--mail', action='store_true', help='学習の終了を通知する')

    args = parser.parse_args()

    # 定数の設定
    lr           = args.lr
    iter         = int(args.iter)
    seed         = args.seed
    mail         = args.mail
    device       = args.device
    file_num     = args.file_num
    pramPath     = args.pramPath
    savePath     = args.savePath
    patience     = args.patience
    valLimit     = int(args.valLimit)
    trainLimit   = int(args.trainLimit)
    batch_size   = args.batch_size
    isContinue   = args.isContinue
    trainRatio   = args.trainRatio
    model_name   = args.model
    model_module = args.model_module

    # モジュール、クラスの存在確認
    ModelClass = load_model.load_model_class(model_module, model_name)

    # シード値の設定
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # 学習データの作成
    files = os.listdir("./kifu_data/bin")
    random.shuffle(files)
    all_bin_paths = [f"./kifu_data/bin/{fn}" for fn in files[:min(file_num, len(files))]]
    dataset = kifu_dataset.MultiPSVDataset(all_bin_paths)
    print("dataset end")

    # 学習用と検証用にデータを分割
    train_size = int(trainRatio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # データを圧縮
    train_indices = torch.randperm(train_size, generator=g)[:min(trainLimit, train_size)]
    val_indices   = torch.randperm(val_size, generator=g)[:min(valLimit, val_size)]
    # データセットを圧縮
    train_dataset_small = Subset(train_dataset, train_indices)
    val_dataset_small   = Subset(val_dataset, val_indices)
    print("split end")

    train_loader = DataLoader(
        train_dataset_small,     # 局面情報とスコアがまとめられているオブジェクト
        batch_size=batch_size,   # 一回のバッチ学習で使用する局面の数
        shuffle=True,            # 学習する局面の順番をシャッフル
        num_workers=2,           # データを読み込む処理の並列数
        pin_memory=True,         # GPUへの転送の高速化
        persistent_workers=True, # データ読み込みの高速化
        prefetch_factor=2,       # データを先読みするバッチ数
        generator=g              # シード値の設定
    )
    print("train_loader end")

    val_loader = DataLoader(
        val_dataset_small,       # 局面情報とスコアがまとめられているオブジェクト
        batch_size=batch_size,   # 一回のバッチ学習で使用する局面の数
        shuffle=False,           # 検証する局面の順番はシャッフルする必要がない
        num_workers=2,           # データを読み込む処理の並列数
        pin_memory=True,         # GPUへの転送の高速化
        persistent_workers=True, # データ読み込みの高速化
        prefetch_factor=2,       # データを先読みするバッチ数
        generator=g              # シード値の設定
    )
    print("val_loader end")

    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    print(f"device : {device}")

    model = ModelClass().to(device)                                              # モデルの生成
    criterion = nn.SmoothL1Loss(beta=0.5)                                        # モデルの評価値と学習データの評価値の差を計算
    # optimizer = optim.Adam(model.parameters(), lr=lr)                            # パラメータの修正
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3) # 学習率の動的最適化
    start_epoch = 0                                                              # epoch数の初期化
    if isContinue:
        ckpt = torch.load(pramPath, map_location=device) # パラメータの読み込み
        model.load_state_dict(ckpt["model"])             # パラメータを設定
        optimizer.load_state_dict(ckpt["optimizer"])     # 最適化パラメータの設定
        scheduler.load_state_dict(ckpt["scheduler"])     # 学習率のスケジュールを読み込み

        start_epoch = ckpt["epoch"] + 1                  # epoch数の読み込み

    # 学習
    end_epoch = start_epoch + iter    # 終了時の epoch 数
    time_hist = deque(maxlen=10)      # 終了予想時間の計算に使用
    stop_cnt  = 0                     # 改善されなかった回数
    best_loss  = 1.0                  # 最高損失率の初期化
    epoch_start = time.perf_counter() # 時間計測開始
    for epoch in range(start_epoch, end_epoch):
        # 訓練
        model.train()
        running = 0.0
        n = 0
        for i, (x, y) in enumerate(tqdm(train_loader, leave=False)):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            n += x.size(0)
        
        train_loss = running / n # 損失率を計算
        
        # テスト
        model.eval()
        vrunning = 0.0
        vn = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_loader, leave=False)):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pred = model(x)
                vloss = criterion(pred, y)

                vrunning += vloss.item() * x.size(0)
                vn += x.size(0)

        val_loss = vrunning / vn # 損失率を計算

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start                                      # 経過時間の計算
        epoch_start = epoch_end                                                   # 時間計測開始地点の更新
        time_hist.append(epoch_time)
        current_lr = optimizer.param_groups[0]['lr']                              # 現在の学習率
        passed_epoch = epoch - start_epoch + 1                                    # 今のところの学習回数
        avg_time_per_epoch = sum(time_hist) / len(time_hist)                      # 一回の学習あたりの平均時間
        remaining_epoch = end_epoch - (epoch + 1)                                 # 残りの学習回数
        remaining_seconds = avg_time_per_epoch * remaining_epoch                  # 学習が終わるまでの経過時間を計算
        expected_end_time = datetime.now() + timedelta(seconds=remaining_seconds) # 予想終了時刻を計算
        print(f"epoch {epoch + 1:>5}, train_loss {train_loss:.6f}, val_loss {val_loss:.6f}, lr {current_lr:.2e}, end {expected_end_time.strftime("%Y/%m/%d-%H:%M:%S")}")

        # 指定数の epoch 連続で損失率が改善しなかったら終了する
        if best_loss <= val_loss:
            stop_cnt += 1
            if stop_cnt >= patience: break
        else:
            stop_cnt = 0
            best_loss = val_loss # 最高損失率を更新

            # パラメータを保存
            save_pram(model, optimizer, scheduler, epoch, val_loss, savePath)

    if mail:
        notify_result(epoch, train_loss, val_loss)