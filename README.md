# RF-DETR Keypoint Web (onnxruntime-web + WebGPU)

RoboflowのRF-DETR([`RFDETRKeypointPreview`](https://github.com/roboflow/rf-detr))によるkeypoint(姿勢推定)推論を、サーバー側の計算なしにブラウザだけで動かすデモです。サーバーはこのリポジトリを配信するだけの静的ホスティング(GitHub Pages)で、推論・描画はすべて利用者のブラウザ上で完結します。

**デモ: https://ys-dirard.github.io/rf-detr-keypoint-web/**

## 概要

- 推論ランタイムは[onnxruntime-web](https://github.com/microsoft/onnxruntime)。WebGPUが使える環境ではWebGPUで、使えない環境は自動的にCPU(Wasm)にフォールバックします。
- モデルはRF-DETRの`RFDETRKeypointPreview`(xlargeチェックポイント)をPyTorchからONNX形式にエクスポートしたもの(約147MB)です。TFLiteのような追加変換は経由していません。
- COCOの17点キーポイントに加え、各キーポイントの位置の不確実性を2×2共分散行列から楕円として可視化します(内側ほど緑=不確実性が小さい、外側ほど赤=不確実性が大きい)。
- 入力は同梱のデモ画像とWebカメラの2種類から切り替えられ、不確実性楕円の表示・非表示も切り替えられます。

## 使い方

1. デモページを開きます。
2. 「1. モデルファイルをダウンロード」のリンクから`rfdetr-keypoint-preview.onnx`(約147MB)をダウンロードします。
3. ダウンロードしたファイルをファイル選択(`<input type="file">`)で選びます。
4. 数秒待つとモデルの読み込みが完了し、デモ画像に対する推論結果が表示されます。

モデルファイルを`fetch`で自動取得せず、ダウンロード+ファイル選択という2手順にしているのは、GitHub Releasesのアセットが`fetch`に対してCORSでブロックされるためです(`<a download>`によるブラウザの通常ダウンロードはCORSの対象になりません)。詳しい経緯は本アプリの検証記録に譲ります。

## ローカルで再現する

このアプリはビルド不要で、静的ファイルをそのまま配信するだけで動きます。

```bash
git clone https://github.com/ys-dirard/rf-detr-keypoint-web.git
cd rf-detr-keypoint-web
python -m http.server 8000
# ブラウザで http://localhost:8000/ を開く
```

`file://`で直接開くと`getUserMedia`(Webカメラ)やESモジュールの読み込みが制限されるため、`http://localhost`などのローカルサーバー経由で開いてください。モデルファイルは上記の「使い方」と同じ手順(Releasesからダウンロード→ファイル選択)で読み込みます。

WebGPUが有効かどうかは、Chrome/Edgeのアドレスバーで`chrome://gpu`を開き、`WebGPU`の項目が`Hardware accelerated`になっているかで確認できます。`Software only`の場合はGPUを積んだ実機でもCPU(Wasm)にフォールバックし、1推論あたり数秒かかります。

## モデルを自分でエクスポートする

配布しているONNXファイルは、`rfdetr`パッケージの`export()`をPython 3.10以上の環境でそのまま実行すれば再現できます。TFLiteへの変換は経由しないため、TFLite変換専用の`tflite` extra(Python 3.12限定)は不要です。

```bash
python -m venv .venv
.venv/Scripts/activate  # Windowsの場合。macOS/Linuxは source .venv/bin/activate
pip install "rfdetr[onnx]"
```

```python
from rfdetr import RFDETRKeypointPreview

# num_classes=1 を指定してもモデル初期化時に
# 「重みが部分的にしかロードされていない」という警告が出るが、無視してよい
# (rfdetr側の既知の警告で、可視化結果には影響しない)
model = RFDETRKeypointPreview(num_classes=1)

# 初回はチェックポイント(約156MB)が自動ダウンロードされる
path = model.export(output_dir="export_out", format="onnx")
print(path)  # export_out/rfdetr-keypoint-preview.onnx
```

生成される`rfdetr-keypoint-preview.onnx`は、入出力の形状・値ともに配布中のファイルと同等です(実際に本READMEの検証時点でも、`test_person.jpg`に対する`dets`の1件目の値が配布中のファイルと一致することを確認しています)。ファイルサイズは、インストールされる`onnx`ライブラリのバージョンによって多少前後します(検証時点で約147〜163MB)。

エクスポートしたファイルは、そのまま`index.html`のファイル選択(`<input type="file">`)から読み込んで使えます。

## 技術構成

```
index.html         推論・描画・ファイル選択のロジックをすべてここに書いている
test_person.jpg    デモ用の静止画像
```

| 項目 | 内容 |
|---|---|
| 推論ランタイム | onnxruntime-web 1.20.1(jsdelivr CDNから`<script type="module">`でimport) |
| モデル形式 | ONNX(`rfdetr`の`export(format="onnx")`で生成) |
| 入力テンソル | `[1, 3, 576, 576]`(float32、NCHW、0〜1に正規化) |
| 出力テンソル | `dets: [1,100,4]`、`labels: [1,100,2]`、`keypoints: [1,100,34,8]` |
| モデル配布 | GitHub Releases(`v1.0.0`アセット) |

モデルの出力テンソルの解釈(クラスパディングされた34スロット、precision-Choleskyパラメータから共分散行列への変換など)は、`index.html`内のコメント、および本アプリの検証記録を参照してください。

## ライセンス・注記

- アプリのコード(`index.html`)はこのリポジトリの内容をそのまま利用して構いません。
- モデルの重みはRoboflowが公開する[`rfdetr`](https://github.com/roboflow/rf-detr)の事前学習済みチェックポイント(`rf-detr-keypoint-preview-xlarge`)を元にしています。モデル自体のライセンス・利用条件はRoboflowの配布元に従います。
- `RFDETRKeypointPreview`は2026年7月時点でRoboflow側のプレビュー(実験的)機能です。
