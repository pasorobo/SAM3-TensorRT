# SAM3-TensorRT セットアップガイド

**最終更新**: 2025年12月26日
**対象**: Phase 1-B (TensorRT Integration)

---

## 🎯 概要

このガイドでは、SAM3ビデオトラッキングをTensorRTで高速化するための完全なセットアップ手順を説明します。

---

## 📋 システム要件

### 必須要件

- **OS**: Linux (Ubuntu 20.04+ 推奨)
- **GPU**: NVIDIA GPU (Compute Capability 7.0+)
  - 推奨: RTX 3080, RTX 4090, A100, H100
  - 最小: GTX 1080 Ti, RTX 2060
- **CUDA**: 11.8 以上
- **Python**: 3.8 - 3.11
- **メモリ**: 16GB RAM以上
- **ストレージ**: 20GB以上の空き容量

### 推奨スペック

- **GPU**: RTX 4090 または A100
- **RAM**: 32GB以上
- **CUDA**: 12.0+
- **Python**: 3.10

---

## 🔧 インストール手順

### ステップ1: CUDA環境のセットアップ

```bash
# CUDA 12.0のインストール（Ubuntu 22.04の例）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# 環境変数の設定
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 確認
nvidia-smi
nvcc --version
```

### ステップ2: TensorRTのインストール

#### 方法A: NVIDIA公式サイトからダウンロード（推奨）

```bash
# TensorRT 8.6 GA for Ubuntu 22.04 and CUDA 12.0
# https://developer.nvidia.com/tensorrt からダウンロード

# tarファイルを展開
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
cd TensorRT-8.6.1.6

# Pythonパッケージのインストール
pip install python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl

# trtexecのパスを通す
echo 'export PATH=$HOME/TensorRT-8.6.1.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 確認
trtexec --version
```

#### 方法B: Pip経由（簡易版）

```bash
# TensorRT Pythonバインディングのみ
pip install tensorrt
```

**注意**: pip版ではtrtexecが含まれない場合があります。

### ステップ3: Python依存関係のインストール

```bash
# リポジトリのクローン
git clone https://github.com/pasorobo/SAM3-TensorRT.git
cd SAM3-TensorRT

# 仮想環境の作成（推奨）
python3 -m venv venv
source venv/bin/activate

# 基本依存関係
pip install -r requirements.txt

# 最新のTransformers（SAM3サポート付き）
pip install git+https://github.com/huggingface/transformers.git

# TensorRT関連（方法Aを使った場合は不要）
pip install tensorrt pycuda

# 確認
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import tensorrt as trt; print(f'TensorRT: {trt.__version__}')"
```

### ステップ4: SAM3モデルアクセスの取得

```bash
# 1. Hugging Faceアカウントを作成
# https://huggingface.co/join

# 2. SAM3モデルへのアクセスを申請
# https://huggingface.co/facebook/sam3
# 「Access repository」をクリック

# 3. Hugging Face CLIにログイン
pip install huggingface_hub
huggingface-cli login
# トークンを入力

# 4. 環境変数の設定（オプション）
echo 'export HF_TOKEN=<your_token>' >> ~/.bashrc
source ~/.bashrc
```

---

## 🚀 TensorRTエンジンのビルド

### ステップ5: ONNXモデルのエクスポート

```bash
# SAM3モデルをONNX形式にエクスポート
python onnxexport_video.py

# 出力確認
ls -lh onnx_weights/
# sam3_video_encoder.onnx (予想サイズ: 2-3GB)
# sam3_video_tracker.onnx (予想サイズ: 1-2GB)
```

**予想実行時間**: 5-10分

### ステップ6: TensorRTエンジンのビルド

```bash
# FP16エンジンのビルド（推奨・バランス型）
bash scripts/build_video_engines.sh fp16

# または手動でビルド
trtexec \
  --onnx=onnx_weights/sam3_video_encoder.onnx \
  --saveEngine=trt_engines/sam3_video_encoder_fp16.plan \
  --fp16 \
  --minShapes=pixel_values:1x3x512x512 \
  --optShapes=pixel_values:1x3x1024x1024 \
  --maxShapes=pixel_values:1x3x1920x1080 \
  --workspace=4096 \
  --verbose

# トラッカーエンジン
trtexec \
  --onnx=onnx_weights/sam3_video_tracker.onnx \
  --saveEngine=trt_engines/sam3_video_tracker_fp16.plan \
  --fp16 \
  --workspace=4096 \
  --verbose
```

**予想実行時間**: 10-30分（GPU性能に依存）

**出力確認**:
```bash
ls -lh trt_engines/
# sam3_video_encoder_fp16.plan (予想サイズ: 1-2GB)
# sam3_video_tracker_fp16.plan (予想サイズ: 0.5-1GB)
```

### その他の精度モード

```bash
# INT8（最速・精度は若干低下）
bash scripts/build_video_engines.sh int8

# FP8（Hopper GPU以降）
bash scripts/build_video_engines.sh fp8

# すべての精度モード
bash scripts/build_video_engines.sh all
```

---

## ✅ 動作確認

### テスト実行

```bash
# 1. Pythonインポートテスト
python3 -c "
from sam3_video import create_session_manager, get_version_info
info = get_version_info()
print(f'Version: {info[\"version\"]}')
print(f'Phase: {info[\"phase\"]}')
print(f'TensorRT Support: {info[\"tensorrt_support\"]}')
"

# 期待される出力:
# Version: 0.2.0
# Phase: 1-B
# TensorRT Support: True

# 2. TensorRTエンジンの検出
python3 -c "
from sam3_video import Sam3VideoSessionManagerTRT
session = Sam3VideoSessionManagerTRT()
info = session.get_inference_info()
print(f'Inference Mode: {info[\"mode\"]}')
print(f'Encoder Ready: {info.get(\"encoder_ready\", False)}')
print(f'Tracker Ready: {info.get(\"tracker_ready\", False)}')
"

# 期待される出力:
# Inference Mode: tensorrt
# Encoder Ready: True
# Tracker Ready: True
```

### サンプルビデオで実行

```bash
# テストビデオのダウンロード（サンプル）
wget https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4 \
  -O test_video.mp4

# TensorRT対応デモの実行
python examples/video_tracking_demo_trt.py \
  --video test_video.mp4 \
  --prompt "person" \
  --output output_tracked.mp4 \
  --max-frames 100

# 期待される出力:
# Inference Mode: TENSORRT
#   Encoder: ✓
#   Tracker: ✓
# Average FPS: 18-21 (GPU性能に依存)
```

---

## 📊 ベンチマーク実行

```bash
# PyTorch vs TensorRT性能比較
python benchmarks/video_benchmark_trt.py \
  --video test_video.mp4 \
  --prompt "person" \
  --frames 100

# 期待される出力例:
# Mode            FPS          Latency (ms)    Speedup
# ─────────────────────────────────────────────────────
# PyTorch         8.50         117.65          1.00x
# TensorRT        21.30        46.95           2.51x
#
# TensorRT Improvement:
#   Speedup: 2.51x faster
#   Latency reduction: 60.1%
#   FPS improvement: +12.80 FPS
```

---

## 🐛 トラブルシューティング

### 問題1: trtexecが見つからない

```bash
# 症状
bash: trtexec: command not found

# 解決策
# PATHの確認
echo $PATH | grep TensorRT

# PATHの追加
export PATH=$HOME/TensorRT-8.6.1.6/bin:$PATH
export LD_LIBRARY_PATH=$HOME/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH

# .bashrcに永続化
echo 'export PATH=$HOME/TensorRT-8.6.1.6/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 問題2: CUDAが見つからない

```bash
# 症状
nvcc: command not found
または
libnvinfer.so: cannot open shared object file

# 解決策
# CUDA環境変数の設定
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 問題3: SAM3モデルがダウンロードできない

```bash
# 症状
HTTPError: 403 Client Error: Forbidden

# 解決策
# 1. Hugging Faceでアクセス承認を確認
# 2. トークンを再生成
huggingface-cli login

# 3. 環境変数を設定
export HF_TOKEN=<your_new_token>
```

### 問題4: GPU メモリ不足

```bash
# 症状
CUDA out of memory

# 解決策
# 1. 小さいバッチサイズを使用
# 2. より小さいモデルサイズを使用
# 3. FP16/INT8量子化を使用（メモリ削減）
bash scripts/build_video_engines.sh int8

# 4. GPU使用状況の確認
nvidia-smi
```

### 問題5: TensorRTエンジンが検出されない

```bash
# 症状
Inference Mode: pytorch (期待: tensorrt)

# デバッグ
python3 -c "
from sam3_video import Sam3VideoSessionManagerTRT
import os
print('Current directory:', os.getcwd())
print('Engine files:')
os.system('ls -lh trt_engines/*.plan 2>/dev/null || echo \"No engines found\"')
"

# 解決策
# エンジンファイルの場所を確認
ls trt_engines/

# 明示的にパスを指定
python examples/video_tracking_demo_trt.py \
  --encoder-engine trt_engines/sam3_video_encoder_fp16.plan \
  --tracker-engine trt_engines/sam3_video_tracker_fp16.plan \
  --video test.mp4
```

---

## 📈 パフォーマンスチューニング

### エンジン最適化オプション

```bash
# より詳細な最適化（ビルド時間は長くなる）
trtexec \
  --onnx=onnx_weights/sam3_video_encoder.onnx \
  --saveEngine=sam3_video_encoder_fp16_optimized.plan \
  --fp16 \
  --workspace=8192 \
  --avgRuns=100 \
  --duration=30 \
  --best \
  --verbose
```

### バッチサイズの調整

複数フレームを同時処理する場合:

```bash
# バッチサイズ4での最適化
trtexec \
  --onnx=onnx_weights/sam3_video_encoder.onnx \
  --saveEngine=sam3_video_encoder_fp16_batch4.plan \
  --fp16 \
  --minShapes=pixel_values:1x3x1024x1024 \
  --optShapes=pixel_values:4x3x1024x1024 \
  --maxShapes=pixel_values:8x3x1024x1024
```

---

## 🎯 期待パフォーマンス

### RTX 4090での性能

| 設定 | FPS @ 1080p | レイテンシ | メモリ |
|------|-------------|----------|--------|
| PyTorch FP32 | 8-10 FPS | 100-120ms | 12.8GB |
| TensorRT FP16 | 18-21 FPS | 47-55ms | 1.6GB |
| TensorRT INT8 | 22-25 FPS | 40-45ms | 0.8GB |

### A100での性能

| 設定 | FPS @ 1080p | レイテンシ | メモリ |
|------|-------------|----------|--------|
| PyTorch FP32 | 10-12 FPS | 83-100ms | 12.8GB |
| TensorRT FP16 | 20-24 FPS | 41-50ms | 1.6GB |
| TensorRT INT8 | 25-30 FPS | 33-40ms | 0.8GB |

---

## 📚 参考リソース

- **NVIDIA TensorRT**: https://developer.nvidia.com/tensorrt
- **TensorRT Documentation**: https://docs.nvidia.com/deeplearning/tensorrt/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **Hugging Face SAM3**: https://huggingface.co/facebook/sam3
- **SAM3-TensorRT GitHub**: https://github.com/pasorobo/SAM3-TensorRT

---

## ✅ チェックリスト

セットアップ完了前に確認:

- [ ] NVIDIA GPU が利用可能
- [ ] CUDA 11.8+ がインストール済み
- [ ] TensorRT 8.5+ がインストール済み
- [ ] Python 3.8-3.11 が利用可能
- [ ] 必要な依存関係をインストール済み
- [ ] SAM3モデルへのアクセス承認済み
- [ ] ONNXエクスポートが成功
- [ ] TensorRTエンジンのビルドが成功
- [ ] テスト実行でTensorRTモードを確認
- [ ] ベンチマークで性能改善を確認

---

**最終更新**: 2025年12月26日
**ガイドバージョン**: 1.0
**対応Phase**: Phase 1-B (TensorRT Integration)
