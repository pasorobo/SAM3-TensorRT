# 環境別セットアップクイックガイド

**対象**: SAM3-TensorRT Phase 1-B

---

## 🖥️ 環境タイプ別の推奨事項

### 1. ローカル開発環境（GPU付き）

**想定**: 個人のワークステーション、NVIDIA GPU搭載

#### 最小スペック
- GPU: GTX 1080 Ti / RTX 2060以上
- RAM: 16GB
- ストレージ: 20GB

#### セットアップ時間
- 初回: 2-3時間
- 2回目以降: 30分

#### 推奨手順
1. [SETUP_GUIDE.md](SETUP_GUIDE.md) の完全版に従う
2. FP16エンジンを推奨（バランス型）

---

### 2. クラウドGPUインスタンス

**想定**: AWS EC2, Google Cloud, Azure

#### 推奨インスタンス
- **AWS**: `p3.2xlarge` (V100), `g5.xlarge` (A10G)
- **GCP**: `n1-standard-4` + T4 GPU
- **Azure**: `NC6s_v3` (V100)

#### AMI/イメージ
```bash
# AWS Deep Learning AMIを推奨
# CUDA, cuDNN, TensorRTがプリインストール済み

# または、Nvidia NGC Container
docker pull nvcr.io/nvidia/tensorrt:23.12-py3
```

#### セットアップ時間
- AMI使用: 30-60分
- Dockerコンテナ: 15-30分

---

### 3. Jupyter Notebook / Colab

**想定**: Google Colab, Kaggle Notebooks

#### Google Colab

```python
# GPUランタイムを有効化
# Runtime > Change runtime type > GPU

# TensorRTのインストール
!pip install tensorrt pycuda

# SAM3-TensorRTのクローン
!git clone https://github.com/pasorobo/SAM3-TensorRT.git
%cd SAM3-TensorRT

# 依存関係のインストール
!pip install -r requirements.txt
!pip install git+https://github.com/huggingface/transformers.git

# Hugging Faceログイン
from huggingface_hub import notebook_login
notebook_login()

# ONNXエクスポート
!python onnxexport_video.py

# TensorRTエンジンビルド（trtexecが利用可能な場合）
!bash scripts/build_video_engines.sh fp16
```

**注意**: Colabの無料版ではtrtexecが利用できない場合があります。

---

### 4. エッジデバイス（NVIDIA Jetson）

**想定**: Jetson AGX Orin, Jetson Orin Nano

#### JetPack SDK
```bash
# JetPack 5.1+ を推奨
# TensorRT 8.5+ が含まれる

# セットアップ
sudo apt-get update
sudo apt-get install python3-pip

# 依存関係
pip3 install -r requirements.txt

# Jetson用の最適化
# INT8量子化を推奨（メモリとパフォーマンスのバランス）
bash scripts/build_video_engines.sh int8
```

#### 期待性能（Jetson AGX Orin）
- INT8: 10-15 FPS @ 1080p
- FP16: 8-12 FPS @ 1080p

---

### 5. Docker コンテナ

**推奨**: 再現性の高い環境構築

#### Dockerfileサンプル

```dockerfile
FROM nvcr.io/nvidia/tensorrt:23.12-py3

WORKDIR /workspace

# 依存関係のインストール
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install git+https://github.com/huggingface/transformers.git

# プロジェクトのコピー
COPY . .

# Hugging Faceトークンの設定（ビルド時）
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# ONNXエクスポート（オプション）
# RUN python onnxexport_video.py

CMD ["/bin/bash"]
```

#### ビルドと実行

```bash
# ビルド
docker build --build-arg HF_TOKEN=<your_token> -t sam3-tensorrt .

# 実行
docker run --gpus all -it \
  -v $(pwd)/videos:/workspace/videos \
  -v $(pwd)/outputs:/workspace/outputs \
  sam3-tensorrt

# コンテナ内でエンジンビルド
bash scripts/build_video_engines.sh fp16

# デモ実行
python examples/video_tracking_demo_trt.py \
  --video /workspace/videos/test.mp4 \
  --output /workspace/outputs/tracked.mp4
```

---

## ⚡ クイックスタート（各環境共通）

### 最速セットアップ（5分）

```bash
# 1. リポジトリクローン
git clone https://github.com/pasorobo/SAM3-TensorRT.git
cd SAM3-TensorRT

# 2. 依存関係インストール
pip install torch transformers opencv-python numpy tqdm

# 3. PyTorchモードで実行（TensorRTなし）
python examples/video_tracking_demo.py \
  --video your_video.mp4 \
  --prompt "person"
```

### TensorRT有効化（追加30-60分）

```bash
# 4. TensorRT関連のインストール
pip install tensorrt pycuda

# 5. Hugging Faceログイン
huggingface-cli login

# 6. ONNXエクスポート
python onnxexport_video.py

# 7. TensorRTエンジンビルド
bash scripts/build_video_engines.sh fp16

# 8. TensorRTモードで実行
python examples/video_tracking_demo_trt.py \
  --video your_video.mp4 \
  --prompt "person"
```

---

## 🎯 環境チェックスクリプト

環境が正しくセットアップされているか確認:

```bash
cat > check_environment.sh << 'EOF'
#!/bin/bash
echo "======================================================================"
echo "SAM3-TensorRT Environment Check"
echo "======================================================================"

# GPU確認
echo -n "NVIDIA GPU: "
nvidia-smi > /dev/null 2>&1 && echo "✓" || echo "✗"

# CUDA確認
echo -n "CUDA: "
nvcc --version > /dev/null 2>&1 && echo "✓ $(nvcc --version | grep release | awk '{print $5}')" || echo "✗"

# TensorRT確認
echo -n "TensorRT (trtexec): "
trtexec --version > /dev/null 2>&1 && echo "✓" || echo "✗"

# Python依存関係
echo -n "PyTorch: "
python3 -c "import torch; print(f'✓ {torch.__version__}')" 2>/dev/null || echo "✗"

echo -n "Transformers: "
python3 -c "import transformers; print(f'✓ {transformers.__version__}')" 2>/dev/null || echo "✗"

echo -n "TensorRT Python: "
python3 -c "import tensorrt; print(f'✓ {tensorrt.__version__}')" 2>/dev/null || echo "✗"

echo -n "OpenCV: "
python3 -c "import cv2; print(f'✓ {cv2.__version__}')" 2>/dev/null || echo "✗"

# SAM3-TensorRTモジュール
echo -n "SAM3-TensorRT: "
python3 -c "from sam3_video import get_version_info; info=get_version_info(); print(f'✓ v{info[\"version\"]} (Phase {info[\"phase\"]})')" 2>/dev/null || echo "✗"

echo "======================================================================"
EOF

chmod +x check_environment.sh
./check_environment.sh
```

---

## 📊 環境別推奨設定

| 環境 | 精度モード | 解像度 | 期待FPS | メモリ |
|------|----------|--------|---------|--------|
| **RTX 4090** | INT8 | 1080p | 22-25 | 8GB |
| **RTX 3080** | FP16 | 1080p | 14-18 | 10GB |
| **A100** | FP16 | 1080p | 20-24 | 16GB |
| **T4** | INT8 | 720p | 10-14 | 8GB |
| **Jetson Orin** | INT8 | 1080p | 10-15 | 8GB |
| **Colab (Free)** | PyTorch | 720p | 4-6 | 12GB |

---

**最終更新**: 2025年12月26日
