# 顔画像からの人物推定(スキルゼミSS11課題)
pytorchを用いて，顔画像の分類タスクに取り組みました
## ✅ 設定課題
4種類の絵文字(🤔🥺😳😍)を再現した表情を見分ける

使用例＞
- 主に中高生の間で流行している動画投稿アプリtiktok上での＃絵文字チャレンジのように音楽に合わせいろいろな表情をするゲーム

## ✅ 必要な環境
- Python 3+

`
requirement.txt`に従ってパッケージを一括でインストールする　
```bash
$ pip install -r requirements.txt
```
もしくは，以下のpythonライブラリを手動でインストール:

```bash
$ pip install numpy 
$ pip install torch
$ pip install torchvision
```


## ✅ 動作仕様
- trainパートとpredictパートに分かれる
- 学習用画像(train,valid),test画像を以下のディレクトリ構造になるように格納する
 ```bash
 train.py
 predict.py
 train_face>
	 >test
		> unkown
			>image001.jpg
			>image002.jpg
			...
	 >train
		> 🤔
			>image001.jpg
			>image002.jpg
			...
		> 🥺
			>image100.jpg
			>image101.jpg
			>...
		...
	 >valid
		> 🤔
			>image001.jpg
			>image002.jpg
			...
		> 🥺
			>image100.jpg
			>image101.jpg
			>...
		...
```

## How to train

```bash
python train.py [--VGG] [--AlexNet] [--ResNet] [-h]
```
```bash
--VGG:　VGG16モデルをファインチューニングする
--AlexNet:　AlexNetをファインチューニングする
--ResNet:　ResNetをファインチューニングする
--h:　ヘルプの表示をする
```
- 学習済みモデルは./logsに保存される
- Free>阿部さんへ＞ss11にて画像ファイルや事前学習済みのモデルを使用可能

## How to predict

```bash
python predict.py [--VGG] [--AlexNet] [--ResNet] [-h]
```
```bash
--VGG:　VGG16モデルを利用する
--AlexNet:　AlexNetを利用する
--ResNet:　ResNetを利用する
--h:　ヘルプの表示をする
```
推論結果を配列で返す
test内の画像をランダムで取得し，0,1,2,3のラベルを返す．
- 😳0, 🥺1, 😍2, 🤔3

- confusion_matrix.pyにて評価指標を計算する(おまけ)

## 📝 未実装課題
- その他モデルの比較
- ハイパーパラメータのチューニング
- データ拡張
- CUDAでの動作(Runtimeerror:メモリ不足が解決せず)
	＞試したこと：バッチサイズを下げる

## :wrench: 開発環境
- Python 3.7.3
- PyCharm 2020.3 Professional Edition
