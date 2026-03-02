# 実行方法

## venv(仮想環境)を用いた実行

`C:\Users\reoko\shogi_ai\python>`にいる状態で以下のコマンドを実行することにより、仮想環境に入ることができる。

仮想環境の作成

```
python -m venv .usi_test
```

仮想環境の起動

```
.\.usi_test\Scripts\activate
```

仮想環境の終了

```
deactivate
```

## 実行ファイルにする方法

例 : cshogi, numpy を使用している場合

```bash
python -m PyInstaller sample.py --clean --noconfirm --onedir ^
  --collect-all numpy ^
  --collect-all cshogi ^
  --hidden-import=numpy.core.multiarray ^
  --hidden-import=numpy.core._multiarray_umath ^
  --hidden-import=cshogi._cshogi
```
## ライブラリのインストール
```bash
pip install -r requirements.txt
```

## オンライン対戦の仕方

1. 将棋所を起動

2. サーバに接続する
   > ホスト: wdoor.c.u-tokyo.ac.jp  
   > ポート: 4081  
   > ログイン名: 任意 (登録等は不要)

他のプログラムと重なると不便なので なるべくオリジナルのものを。

> 例: gps, gps_2cpu, gps_3sec など。

パスワード: floodgate-300-10F,trip

> tripの部分はユーザ固有の何らかの文字列で置き換える。

同名ユーザの区別に使うためオリジナルのものにする。

> 平文で流れるため盗聴されて困るものは使わないこと

毎時0分, 30分近辺で参加プログラム同士の対戦が行われる。

終了後はログアウト状態に戻る。

フィッシャークロックルールを用い、当初の持ち時間が10分、自分の手番が回ってくるごとに10秒加算

1に戻り再び接続する。(手動でも問題ないが自動にしておく方が便利)
15試合程度でレーティングが計算される。

## 学習時の判断基準

1. trainだけ下がる

   > データ不足 or 過学習

2. 両方止まる

   > モデル容量不足

3. valが乱高下

   > 局面数不足

## ニューラルネットワークで出力するもの

- 勝率 [0, 1]

  > game_resultのみを学習する

- 評価値 [-1, 1]

  > scoreのみを学習する

- 評価値 [-1, 1]

  > scoreとgame_resultを複合させたものを学習する

- 指し手
  > 指し手を学習する

## ファイルの説明

### kifu_dataset.py

学習に使用するデータセットを作成するプログラム

### model_test.py

モデルおよびパラメータのテストをするプログラム

### NN_model.py

ニューラルネットワークを使用したモデルを収納されているプログラム

### save_pram.py

パラメータを保存するプログラム

### train_pram.py

モデルの学習を行うプログラム

- 損失率の改善が見られなくなったら学習率を下げて深く訓練する

### train_pram_change.py

モデルの学習を行うプログラム

- [train_pram.py](#train_prampy.py) での訓練データがepochごとに固定されているところをepochごとに訓練データを変化させることで汎化性能を向上させた。
  > 検証データの損失率を低くさせることができる

### train_pram_early_stop.py

モデルの学習を行うプログラム

- 学習率を固定して規定回数損失率の改善が見られなかったら学習を終了する
  > 手早く訓練できる

