# photo-upscaler

iPad（Safari）だけで動く、AIによる写真の復元・高画質化アプリです（個人用）。
画質の悪い人物写真を、セブン-イレブンの写真プリント（2L判）できれいに印刷できるようにします。
写真はiPadの中だけで処理し、外部には送りません。

- 仕様書：[docs/SPEC.md](docs/SPEC.md)

## 進み具合
- [x] 1. 写真を選ぶ・切り抜く・ぼかした余白・書き出し（AIなし）
- [x] 2. Real-ESRGAN（小さく分けて処理する、WebGPU）
- [x] 3. 顔検出＋CodeFormer＋強さのスライダー
- [ ] 4. 補正前と比べる画面・PWA化・AIデータの保存
- [ ] 5. セブン-イレブンで試し印刷 → 調整

## 開発
```sh
npm test        # 計算部分の試験
npm run serve   # http://localhost:8080 で表示
```
ビルドは不要です（素の HTML / JavaScript モジュール）。

### AIのデータを作り直すとき
```sh
curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
pip install numpy onnx   # PyTorch は不要
python tools/export_realesrgan_onnx.py RealESRGAN_x4plus.pth models/realesrgan-x4plus.onnx
```

### 顔補正のAIデータを作り直すとき
```sh
git clone --depth 1 https://github.com/sczhou/CodeFormer
curl -LO https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
curl -LO https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth
pip install torch onnx numpy onnxscript
python tools/export_codeformer_onnx.py CodeFormer codeformer.pth detection_mobilenet0.25_Final.pth out
cp out/retinaface.onnx models/retinaface-mnet025.onnx
split -b 95000000 -d -a 1 out/codeformer.onnx models/codeformer.onnx.part
```

## 使っているもの
- Real-ESRGAN（BSD-3-Clause）https://github.com/xinntao/Real-ESRGAN
- ONNX Runtime Web（MIT）https://github.com/microsoft/onnxruntime
- CodeFormer（S-Lab License 1.0：非商用のみ。このアプリは個人用）https://github.com/sczhou/CodeFormer
- RetinaFace（facexlib、MIT）https://github.com/xinntao/facexlib
- coi-serviceworker（MIT）https://github.com/gzuidhof/coi-serviceworker
