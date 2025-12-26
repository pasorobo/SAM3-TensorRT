# SAM3 ストリーミングトラッキング実装プラン

**作成日**: 2025年12月26日
**プロジェクト**: SAM3-TensorRT
**目的**: ビデオストリーミングトラッキング機能の実装

---

## 🎯 実装可能性の評価

### **結論: 実装可能、ただし段階的アプローチが必要**

SAM3のビデオストリーミングトラッキング機能は**技術的に実装可能**ですが、以下の理由により複雑度が高いです:

1. ✅ **SAM2の実装例が存在**: [ONNX-SAM2-Segment-Anything](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything) が参考実装として利用可能
2. ✅ **コンポーネント分離が可能**: SAM3は検出器とトラッカーを分離したアーキテクチャ
3. ⚠️ **状態管理の複雑さ**: メモリバンクとFIFOキューの実装が必要
4. ⚠️ **TensorRT最適化の制約**: 動的な状態管理とTensorRTの静的グラフの両立

---

## 🏗️ SAM3 アーキテクチャ概要

### コンポーネント構成

```
┌─────────────────────────────────────────────┐
│         SAM3 Video Architecture             │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │   Shared Vision Backbone (Encoder)   │  │
│  │         (848M parameters)             │  │
│  └────────────┬─────────────────────────┘  │
│               │                             │
│       ┌───────┴────────┐                    │
│       │                │                    │
│  ┌────▼─────┐    ┌────▼──────────────┐     │
│  │ DETR-    │    │ SAM2-based        │     │
│  │ based    │    │ Memory Tracker    │     │
│  │ Detector │    │                   │     │
│  └──────────┘    │ ┌───────────────┐ │     │
│                  │ │ Memory Bank   │ │     │
│                  │ │ - Recent (N)  │ │     │
│                  │ │ - Prompted(M) │ │     │
│                  │ └───────────────┘ │     │
│                  └───────────────────┘     │
│                                             │
└─────────────────────────────────────────────┘
```

### メモリバンクの仕組み

- **Recent Frames Queue**: 最近のN個のフレームの空間特徴マップを保持（FIFO）
- **Prompted Frames Queue**: プロンプト付きM個のフレームの空間特徴マップを保持（FIFO）
- **Temporal Consistency**: フレーム間の一貫性を維持

**出典**:
- [EfficientSAM3 Architecture](https://arxiv.org/html/2511.15833v1)
- [SAM2 Memory Bank Architecture](https://ritvik19.medium.com/papers-explained-239-sam-2-6ffb7f187281)
- [Distractor-Aware Memory for SAM2](https://github.com/jovanavidenovic/DAM4SAM)

---

## 📋 実装アプローチ: 3段階戦略

### **フェーズ1: ハイブリッドアプローチ（推奨）** ⭐

**コンセプト**: TensorRTで推論を高速化し、Python側で状態管理を行う

```python
# アーキテクチャ概要
┌──────────────────────────────────────────────┐
│  Python Application Layer                    │
│  ┌────────────────────────────────────────┐  │
│  │ Video Session Manager                  │  │
│  │ - Frame buffering                      │  │
│  │ - Memory bank (Recent/Prompted queues) │  │
│  │ - Object tracking state                │  │
│  └────────────────────────────────────────┘  │
│              │                     ▲          │
│              ▼                     │          │
│  ┌─────────────────┐   ┌──────────────────┐  │
│  │ TensorRT Engine │   │ TensorRT Engine  │  │
│  │   (Encoder)     │   │  (Tracker Head)  │  │
│  └─────────────────┘   └──────────────────┘  │
└──────────────────────────────────────────────┘
```

**利点**:
- ✅ 実装が相対的に容易
- ✅ TensorRTの高速化を活用可能
- ✅ 柔軟な状態管理（デバッグが容易）
- ✅ 段階的な最適化が可能

**欠点**:
- ⚠️ Python-TensorRT間のデータ転送オーバーヘッド
- ⚠️ 完全なEnd-to-End最適化は不可

---

### **フェーズ2: 準フルTensorRT**

**コンセプト**: メモリバンクをTensorRTプラグインとして実装

```python
┌──────────────────────────────────────────────┐
│  Python Orchestrator                         │
│  ┌────────────────────────────────────────┐  │
│  │ Video Stream Handler                   │  │
│  └────────────────────────────────────────┘  │
│              │                               │
│              ▼                               │
│  ┌────────────────────────────────────────┐  │
│  │ TensorRT Engine with Custom Plugins    │  │
│  │  ┌──────────────┐  ┌─────────────────┐ │  │
│  │  │  Encoder     │  │ Custom Plugin:  │ │  │
│  │  └──────────────┘  │ Memory Manager  │ │  │
│  │  ┌──────────────┐  └─────────────────┘ │  │
│  │  │ Tracker Head │                      │  │
│  │  └──────────────┘                      │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

**利点**:
- ✅ より高速（データ転送削減）
- ✅ エンドツーエンド最適化
- ✅ プロダクション環境に適している

**欠点**:
- ⚠️ TensorRTプラグイン開発が必要（C++）
- ⚠️ 実装とデバッグが複雑
- ⚠️ 開発時間が長い

---

### **フェーズ3: フルTensorRTネイティブ（将来的）**

**コンセプト**: すべてをTensorRT内で完結

**実現には**:
- TensorRT 10+の新機能活用
- State Space Models (SSM)の統合
- 完全な静的グラフ化

**タイムライン**: 6-12ヶ月の開発期間

---

## 🛠️ フェーズ1詳細実装プラン（ハイブリッドアプローチ）

### ステップ1: SAM3ビデオコンポーネントの理解とエクスポート

#### 1.1 公式SAM3ビデオAPIの調査

**タスク**:
```bash
# SAM3公式リポジトリのクローン
git clone https://github.com/facebookresearch/sam3.git
cd sam3

# ビデオサンプルの実行
python examples/sam3_video_predictor_example.ipynb
```

**調査項目**:
- [ ] `build_sam3_video_predictor()` の実装
- [ ] `handle_request()` のリクエストタイプ
- [ ] メモリバンクの構造
- [ ] 入力/出力の形状とデータ型
- [ ] セッション管理の仕組み

**参考リソース**:
- [SAM3 Video Predictor Example](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb)
- [Video API Usage Documentation](https://deepwiki.com/facebookresearch/sam3/4.1-video-api-usage)

#### 1.2 コンポーネント分離とエクスポート

**ファイル**: `onnxexport_video.py`

```python
import torch
from pathlib import Path
from transformers import Sam3Processor
# SAM3のビデオモデルをインポート（要調査）
from sam3.model_builder import build_sam3_video_predictor

device = "cpu"  # または "cuda"

# 1. ビデオプレディクターのロード
video_predictor = build_sam3_video_predictor(checkpoint="sam3_large")
video_predictor.eval()

# 2. コンポーネント抽出（要実装）
class Sam3VideoEncoderWrapper(torch.nn.Module):
    """ビジョンエンコーダーを分離"""
    def __init__(self, predictor):
        super().__init__()
        self.encoder = predictor.model.vision_encoder

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [B, C, H, W] - 入力画像
        Returns:
            features: [B, D, H', W'] - エンコードされた特徴
        """
        return self.encoder(pixel_values)

class Sam3VideoTrackerWrapper(torch.nn.Module):
    """トラッカーヘッドを分離"""
    def __init__(self, predictor):
        super().__init__()
        self.tracker = predictor.model.tracker

    def forward(self, current_features, memory_features, text_embeddings):
        """
        Args:
            current_features: [B, D, H', W'] - 現在のフレーム特徴
            memory_features: [B, N, D, H', W'] - メモリバンクの特徴
            text_embeddings: [B, T, D] - テキストプロンプト埋め込み
        Returns:
            masks: [B, Q, H, W] - セグメンテーションマスク
            tracking_ids: [B, Q] - オブジェクトトラッキングID
        """
        outputs = self.tracker(
            current_features=current_features,
            memory_features=memory_features,
            text_embeddings=text_embeddings
        )
        return outputs.masks, outputs.tracking_ids

# 3. エンコーダーのエクスポート
encoder_wrapper = Sam3VideoEncoderWrapper(video_predictor)
dummy_frame = torch.randn(1, 3, 1024, 1024)

torch.onnx.export(
    encoder_wrapper,
    (dummy_frame,),
    "onnx_weights/sam3_video_encoder.onnx",
    input_names=["pixel_values"],
    output_names=["features"],
    dynamic_axes={
        "pixel_values": {0: "batch", 2: "height", 3: "width"},
        "features": {0: "batch"}
    },
    opset_version=17,
)

# 4. トラッカーのエクスポート
tracker_wrapper = Sam3VideoTrackerWrapper(video_predictor)
dummy_current = torch.randn(1, 256, 64, 64)  # エンコーダー出力の形状
dummy_memory = torch.randn(1, 8, 256, 64, 64)  # N=8フレームのメモリ
dummy_text = torch.randn(1, 77, 768)  # テキスト埋め込み

torch.onnx.export(
    tracker_wrapper,
    (dummy_current, dummy_memory, dummy_text),
    "onnx_weights/sam3_video_tracker.onnx",
    input_names=["current_features", "memory_features", "text_embeddings"],
    output_names=["masks", "tracking_ids"],
    dynamic_axes={
        "current_features": {0: "batch"},
        "memory_features": {0: "batch", 1: "num_memory_frames"},
        "text_embeddings": {0: "batch"},
        "masks": {0: "batch", 1: "num_objects"},
        "tracking_ids": {0: "batch", 1: "num_objects"}
    },
    opset_version=17,
)

print("Video components exported successfully!")
```

**予想される課題**:
1. ⚠️ SAM3のビデオコンポーネントが `transformers` に未実装の可能性
   - **解決策**: 公式 `facebookresearch/sam3` リポジトリから直接利用
2. ⚠️ メモリバンクの形状が不明
   - **解決策**: 実際のモデル実行で形状を確認
3. ⚠️ トラッキングIDの生成ロジック
   - **解決策**: 初期はPython側で管理

**推定工数**: 2-3日

---

### ステップ2: TensorRTエンジンのビルド

#### 2.1 エンコーダーエンジンのビルド

```bash
# FP16エンジン（推奨）
trtexec \
  --onnx=onnx_weights/sam3_video_encoder.onnx \
  --saveEngine=sam3_video_encoder_fp16.plan \
  --fp16 \
  --minShapes=pixel_values:1x3x512x512 \
  --optShapes=pixel_values:1x3x1024x1024 \
  --maxShapes=pixel_values:1x3x1920x1080 \
  --verbose

# INT8エンジン（さらに高速）
trtexec \
  --onnx=onnx_weights/sam3_video_encoder.onnx \
  --saveEngine=sam3_video_encoder_int8.plan \
  --int8 \
  --minShapes=pixel_values:1x3x512x512 \
  --optShapes=pixel_values:1x3x1024x1024 \
  --maxShapes=pixel_values:1x3x1920x1080 \
  --verbose
```

#### 2.2 トラッカーエンジンのビルド

```bash
# FP16エンジン
trtexec \
  --onnx=onnx_weights/sam3_video_tracker.onnx \
  --saveEngine=sam3_video_tracker_fp16.plan \
  --fp16 \
  --minShapes=memory_features:1x4x256x64x64 \
  --optShapes=memory_features:1x8x256x64x64 \
  --maxShapes=memory_features:1x16x256x64x64 \
  --verbose
```

**期待されるパフォーマンス**:
- エンコーダー: 100ms → **25-40ms** (FP16/INT8)
- トラッカー: 50ms → **12-20ms** (FP16)
- **合計**: ~60ms/フレーム → **16-20 FPS** のリアルタイム処理

**推定工数**: 1日

---

### ステップ3: Python Video Session Managerの実装

#### 3.1 メモリバンククラス

**ファイル**: `sam3_video/memory_bank.py`

```python
from collections import deque
import numpy as np
from typing import Optional, Dict, List

class MemoryBank:
    """SAM3のメモリバンク実装"""

    def __init__(
        self,
        max_recent_frames: int = 8,
        max_prompted_frames: int = 4,
        feature_dim: int = 256,
        feature_size: tuple = (64, 64)
    ):
        """
        Args:
            max_recent_frames: Recent Framesキューの最大サイズ (N)
            max_prompted_frames: Prompted Framesキューの最大サイズ (M)
            feature_dim: 特徴の次元数
            feature_size: 特徴マップのサイズ (H', W')
        """
        self.max_recent = max_recent_frames
        self.max_prompted = max_prompted_frames
        self.feature_dim = feature_dim
        self.feature_size = feature_size

        # FIFOキュー
        self.recent_frames: deque = deque(maxlen=max_recent_frames)
        self.prompted_frames: deque = deque(maxlen=max_prompted_frames)

        # フレームID管理
        self.frame_indices: Dict[int, int] = {}  # frame_id -> queue_position

    def add_frame(
        self,
        frame_id: int,
        features: np.ndarray,
        has_prompt: bool = False
    ):
        """
        新しいフレームの特徴をメモリバンクに追加

        Args:
            frame_id: フレームの一意ID
            features: エンコードされた特徴 [D, H', W']
            has_prompt: プロンプトが付与されたフレームか
        """
        # Recent Framesに追加（常に）
        self.recent_frames.append({
            'frame_id': frame_id,
            'features': features,
            'timestamp': frame_id
        })

        # プロンプト付きフレームはPrompted Framesにも追加
        if has_prompt:
            self.prompted_frames.append({
                'frame_id': frame_id,
                'features': features,
                'timestamp': frame_id
            })

    def get_memory_features(self) -> np.ndarray:
        """
        トラッカーへの入力用にメモリ特徴を結合

        Returns:
            memory_features: [N+M, D, H', W'] 形状の配列
        """
        all_features = []

        # Recent Framesから特徴を取得
        for frame_data in self.recent_frames:
            all_features.append(frame_data['features'])

        # Prompted Framesから特徴を取得
        for frame_data in self.prompted_frames:
            all_features.append(frame_data['features'])

        if not all_features:
            # 空の場合はダミー特徴を返す
            return np.zeros((1, self.feature_dim, *self.feature_size), dtype=np.float32)

        # [N+M, D, H', W'] に結合
        return np.stack(all_features, axis=0)

    def clear(self):
        """メモリバンクをクリア"""
        self.recent_frames.clear()
        self.prompted_frames.clear()
        self.frame_indices.clear()
```

#### 3.2 ビデオセッションマネージャー

**ファイル**: `sam3_video/session_manager.py`

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import cv2

from .memory_bank import MemoryBank

class TensorRTEngine:
    """TensorRTエンジンのラッパー"""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)

        # エンジンのロード
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # 入出力バッファの準備
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # GPUメモリの確保
            device_mem = cuda.mem_alloc(size * dtype.itemsize)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'name': binding, 'mem': device_mem, 'dtype': dtype})
            else:
                self.outputs.append({'name': binding, 'mem': device_mem, 'dtype': dtype})

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """推論実行"""
        # 入力データをGPUにコピー
        for inp in self.inputs:
            cuda.memcpy_htod_async(
                inp['mem'],
                input_data[inp['name']].astype(inp['dtype']),
                self.stream
            )

        # 推論実行
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # 出力をCPUにコピー
        outputs = {}
        for out in self.outputs:
            output_shape = self.context.get_binding_shape(
                self.engine.get_binding_index(out['name'])
            )
            output_array = np.empty(output_shape, dtype=out['dtype'])
            cuda.memcpy_dtoh_async(output_array, out['mem'], self.stream)
            outputs[out['name']] = output_array

        self.stream.synchronize()
        return outputs


class Sam3VideoSessionManager:
    """SAM3ビデオセッション管理"""

    def __init__(
        self,
        encoder_engine_path: str,
        tracker_engine_path: str,
        max_recent_frames: int = 8,
        max_prompted_frames: int = 4
    ):
        """
        Args:
            encoder_engine_path: エンコーダーTensorRTエンジンのパス
            tracker_engine_path: トラッカーTensorRTエンジンのパス
            max_recent_frames: Recent Framesの最大数
            max_prompted_frames: Prompted Framesの最大数
        """
        # TensorRTエンジンのロード
        self.encoder = TensorRTEngine(encoder_engine_path)
        self.tracker = TensorRTEngine(tracker_engine_path)

        # メモリバンクの初期化
        self.memory_bank = MemoryBank(
            max_recent_frames=max_recent_frames,
            max_prompted_frames=max_prompted_frames
        )

        # セッション状態
        self.is_active = False
        self.frame_count = 0
        self.text_prompt = None
        self.text_embeddings = None

        # トラッキング状態
        self.tracked_objects: Dict[int, Dict] = {}  # object_id -> metadata
        self.next_object_id = 0

    def start_session(
        self,
        text_prompt: str,
        video_source: Optional[str] = None
    ):
        """
        ビデオセッションを開始

        Args:
            text_prompt: テキストプロンプト（例: "person", "car"）
            video_source: ビデオファイルパスまたはカメラID
        """
        self.is_active = True
        self.frame_count = 0
        self.text_prompt = text_prompt

        # テキスト埋め込みの生成（要実装: transformersのprocessor使用）
        self.text_embeddings = self._encode_text_prompt(text_prompt)

        # メモリバンクのリセット
        self.memory_bank.clear()
        self.tracked_objects.clear()

        print(f"✅ Session started with prompt: '{text_prompt}'")

    def process_frame(
        self,
        frame: np.ndarray,
        add_prompt: bool = False
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        単一フレームを処理

        Args:
            frame: 入力フレーム [H, W, 3] BGR形式
            add_prompt: このフレームにプロンプトを追加するか

        Returns:
            masks: セグメンテーションマスク [num_objects, H, W]
            detections: 検出されたオブジェクトのリスト
        """
        if not self.is_active:
            raise RuntimeError("Session not started. Call start_session() first.")

        # 1. 前処理
        preprocessed = self._preprocess_frame(frame)

        # 2. エンコーダーで特徴抽出
        encoder_outputs = self.encoder.infer({
            'pixel_values': preprocessed
        })
        current_features = encoder_outputs['features']  # [1, D, H', W']

        # 3. メモリバンクに追加
        self.memory_bank.add_frame(
            frame_id=self.frame_count,
            features=current_features[0],  # [D, H', W']
            has_prompt=add_prompt
        )

        # 4. メモリ特徴の取得
        memory_features = self.memory_bank.get_memory_features()  # [N+M, D, H', W']
        memory_features = np.expand_dims(memory_features, axis=0)  # [1, N+M, D, H', W']

        # 5. トラッカーで推論
        tracker_outputs = self.tracker.infer({
            'current_features': current_features,
            'memory_features': memory_features,
            'text_embeddings': self.text_embeddings
        })

        masks = tracker_outputs['masks']  # [1, Q, H, W]
        tracking_ids = tracker_outputs.get('tracking_ids', None)  # [1, Q]

        # 6. 後処理
        detections = self._postprocess_outputs(
            masks=masks[0],  # [Q, H, W]
            tracking_ids=tracking_ids[0] if tracking_ids is not None else None,
            original_size=frame.shape[:2]
        )

        self.frame_count += 1
        return masks[0], detections

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        fps: int = 30
    ) -> List[Dict]:
        """
        ビデオファイル全体を処理

        Args:
            video_path: 入力ビデオパス
            output_path: 出力ビデオパス（Noneの場合は保存しない）
            fps: 出力FPS

        Returns:
            all_detections: 全フレームの検出結果
        """
        cap = cv2.VideoCapture(video_path)

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_detections = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # フレーム処理
                masks, detections = self.process_frame(frame)
                all_detections.append({
                    'frame_id': self.frame_count - 1,
                    'detections': detections
                })

                # ビジュアライゼーション
                if output_path:
                    vis_frame = self._visualize(frame, masks, detections)
                    out.write(vis_frame)

                # 進捗表示
                if self.frame_count % 30 == 0:
                    print(f"Processed {self.frame_count} frames...")

        finally:
            cap.release()
            if output_path:
                out.release()

        return all_detections

    def end_session(self):
        """セッションを終了"""
        self.is_active = False
        self.memory_bank.clear()
        print(f"✅ Session ended. Processed {self.frame_count} frames.")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """フレームの前処理"""
        # リサイズ、正規化など（要実装）
        resized = cv2.resize(frame, (1024, 1024))
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 正規化
        normalized = rgb.astype(np.float32) / 255.0
        # CHW形式に変換
        transposed = np.transpose(normalized, (2, 0, 1))
        # バッチ次元追加
        batched = np.expand_dims(transposed, axis=0)
        return batched

    def _encode_text_prompt(self, text: str) -> np.ndarray:
        """テキストプロンプトをエンコード（要実装）"""
        # transformersのprocessorを使用
        # 仮実装: ランダムな埋め込み
        return np.random.randn(1, 77, 768).astype(np.float32)

    def _postprocess_outputs(
        self,
        masks: np.ndarray,
        tracking_ids: Optional[np.ndarray],
        original_size: Tuple[int, int]
    ) -> List[Dict]:
        """出力の後処理"""
        detections = []

        for i, mask in enumerate(masks):
            # マスクの閾値処理
            binary_mask = (mask > 0.5).astype(np.uint8)

            # バウンディングボックスの計算
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])

                detections.append({
                    'object_id': int(tracking_ids[i]) if tracking_ids is not None else i,
                    'mask': binary_mask,
                    'bbox': [x, y, w, h],
                    'confidence': float(mask.max())
                })

        return detections

    def _visualize(
        self,
        frame: np.ndarray,
        masks: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """検出結果の可視化"""
        vis = frame.copy()

        for det in detections:
            # マスクのオーバーレイ
            mask = det['mask']
            color = np.random.randint(0, 255, 3).tolist()
            vis[mask > 0] = vis[mask > 0] * 0.5 + np.array(color) * 0.5

            # バウンディングボックス
            x, y, w, h = det['bbox']
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)

            # ラベル
            label = f"ID: {det['object_id']}"
            cv2.putText(vis, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis
```

#### 3.3 使用例

**ファイル**: `examples/video_tracking_demo.py`

```python
from sam3_video.session_manager import Sam3VideoSessionManager

# セッションマネージャーの初期化
session = Sam3VideoSessionManager(
    encoder_engine_path="sam3_video_encoder_fp16.plan",
    tracker_engine_path="sam3_video_tracker_fp16.plan",
    max_recent_frames=8,
    max_prompted_frames=4
)

# セッション開始
session.start_session(
    text_prompt="person",
    video_source="test_video.mp4"
)

# ビデオ処理
detections = session.process_video(
    video_path="test_video.mp4",
    output_path="output_tracked.mp4",
    fps=30
)

# セッション終了
session.end_session()

print(f"Total detections: {len(detections)}")
```

**推定工数**: 5-7日

---

### ステップ4: ベンチマークと最適化

#### 4.1 パフォーマンステスト

**ファイル**: `benchmarks/video_benchmark.py`

```python
import time
import numpy as np
from sam3_video.session_manager import Sam3VideoSessionManager

def benchmark_video_inference(
    session: Sam3VideoSessionManager,
    video_path: str,
    num_runs: int = 3
):
    """ビデオ推論のベンチマーク"""

    results = {
        'encoder_times': [],
        'tracker_times': [],
        'total_times': [],
        'fps': []
    }

    for run in range(num_runs):
        session.start_session(text_prompt="person")

        cap = cv2.VideoCapture(video_path)
        frame_times = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()
            masks, detections = session.process_frame(frame)
            end = time.perf_counter()

            frame_times.append(end - start)

        cap.release()
        session.end_session()

        # 統計計算
        avg_time = np.mean(frame_times)
        results['total_times'].append(avg_time)
        results['fps'].append(1.0 / avg_time)

    # 結果レポート
    print(f"\n{'='*60}")
    print(f"Video Inference Benchmark Results ({num_runs} runs)")
    print(f"{'='*60}")
    print(f"Average FPS: {np.mean(results['fps']):.2f} ± {np.std(results['fps']):.2f}")
    print(f"Average Latency: {np.mean(results['total_times'])*1000:.2f}ms")
    print(f"{'='*60}\n")

    return results

# 実行
session = Sam3VideoSessionManager(
    encoder_engine_path="sam3_video_encoder_fp16.plan",
    tracker_engine_path="sam3_video_tracker_fp16.plan"
)

results = benchmark_video_inference(session, "test_video.mp4")
```

#### 4.2 期待されるパフォーマンス

| コンポーネント | PyTorch (ms) | TensorRT FP16 (ms) | TensorRT INT8 (ms) | 高速化 |
|--------------|-------------|-------------------|-------------------|-------|
| エンコーダー | 100 | 40 | 25 | 2.5-4.0x |
| トラッカー | 50 | 20 | 12 | 2.5-4.0x |
| メモリ管理 | 10 | 10 | 10 | - |
| **合計** | **160ms** | **70ms** | **47ms** | **2.3-3.4x** |
| **FPS** | **6.25** | **14.3** | **21.3** | **2.3-3.4x** |

**推定工数**: 2日

---

### ステップ5: ドキュメントとサンプル

#### 5.1 README更新

```markdown
## ビデオストリーミングトラッキング

### クイックスタート

1. **ビデオコンポーネントのエクスポート**
```bash
python onnxexport_video.py
```

2. **TensorRTエンジンのビルド**
```bash
bash scripts/build_video_engines.sh
```

3. **ビデオトラッキングの実行**
```python
from sam3_video import Sam3VideoSessionManager

session = Sam3VideoSessionManager(
    encoder_engine_path="sam3_video_encoder_fp16.plan",
    tracker_engine_path="sam3_video_tracker_fp16.plan"
)

session.start_session(text_prompt="person")
session.process_video("input.mp4", "output.mp4")
session.end_session()
```

### 機能

- ✅ リアルタイムビデオセグメンテーション（14-21 FPS）
- ✅ テキストプロンプトベースの追跡
- ✅ 複数オブジェクト同時追跡
- ✅ MP4/AVI/JPEG フォルダー対応
- ✅ TensorRT FP16/INT8 最適化
```

#### 5.2 サンプルコード

**examples/**:
- `video_tracking_demo.py`: 基本的な使用例
- `realtime_webcam.py`: Webカメラのリアルタイム追跡
- `batch_processing.py`: 複数ビデオのバッチ処理
- `interactive_prompting.py`: インタラクティブなプロンプト追加

**推定工数**: 2日

---

## 📊 総合スケジュール

### フェーズ1タイムライン

| ステップ | タスク | 推定工数 | 累積日数 |
|---------|------|---------|----------|
| **1** | SAM3ビデオコンポーネント調査・エクスポート | 2-3日 | 3日 |
| **2** | TensorRTエンジンビルド | 1日 | 4日 |
| **3** | Python Session Manager実装 | 5-7日 | 11日 |
| **4** | ベンチマーク・最適化 | 2日 | 13日 |
| **5** | ドキュメント・サンプル | 2日 | 15日 |
| | **バッファ（テスト・デバッグ）** | 5日 | **20日** |

**合計推定期間**: **3-4週間**（フルタイム1名）

---

## 🚧 技術的課題とリスク

### 高リスク

1. **SAM3ビデオAPIの非公開性** 🔴
   - **リスク**: SAM3のビデオコンポーネントが `transformers` ライブラリに未統合
   - **確率**: 70%
   - **対策**: 公式 `facebookresearch/sam3` リポジトリから直接利用
   - **代替案**: SAM2のビデオコンポーネントを使用し、SAM3のテキストプロンプト機能を統合

2. **メモリバンクの形状不一致** 🔴
   - **リスク**: ONNX エクスポート時にメモリバンクの動的形状が非対応
   - **確率**: 60%
   - **対策**: 最大サイズの固定形状でエクスポート、パディングで対応
   - **代替案**: Python側でメモリ管理を完全に行う（現在のプラン）

### 中リスク

3. **TensorRTビルドの失敗** 🟡
   - **リスク**: 複雑な演算子がTensorRTでサポートされていない
   - **確率**: 40%
   - **対策**: ONNX Simplifier、カスタムプラグインの実装
   - **代替案**: ONNXRuntimeでの推論（TensorRTより遅いが動作保証）

4. **パフォーマンス目標未達** 🟡
   - **リスク**: リアルタイム処理（30 FPS）に到達できない
   - **確率**: 30%
   - **対策**: より積極的な量子化（INT8/INT4）、解像度の削減
   - **代替案**: 準リアルタイム（15 FPS）を許容

### 低リスク

5. **テキスト埋め込みの統合** 🟢
   - **リスク**: テキストエンコーダーのONNX変換が必要
   - **確率**: 20%
   - **対策**: CLIPなどの標準的なテキストエンコーダーは変換実績あり

---

## 💰 コストと ROI 分析

### 開発コスト

| 項目 | コスト（時間） | 備考 |
|-----|-------------|------|
| エンジニアリング工数 | 15-20人日 | 1名フルタイム3-4週間 |
| GPU計算リソース | ~$100 | TensorRTビルド・ベンチマーク |
| 検証・テスト | 5人日 | QA、統合テスト |
| **合計** | **20-25人日** | |

### 期待される効果

| 指標 | 改善 | 金銭的価値 |
|-----|------|----------|
| 推論速度 | **2.3-3.4x** | プロダクション環境でのGPUコスト削減 |
| リアルタイム性 | 6 FPS → **14-21 FPS** | 新たなユースケース対応可能 |
| メモリ使用量 | **87-94% 削減** | 複数ストリーム並列処理可能 |

### ROI（投資対効果）

- **開発期間**: 3-4週間
- **期待効果**: リアルタイムビデオ追跡機能の実現
- **ビジネス価値**:
  - 監視カメラシステム
  - 自動運転シミュレーション
  - スポーツ分析
  - AR/VRアプリケーション

**推定ROI**: 開発投資に対して **3-6ヶ月で回収可能**（プロダクション利用の場合）

---

## 🔄 フェーズ2・3への移行計画

### フェーズ2: 準フルTensorRT（6-8週間）

**前提条件**:
- フェーズ1が成功し、リアルタイム処理を達成
- さらなる性能改善が必要

**実装内容**:
1. TensorRTカスタムプラグインでメモリバンクを実装
2. エンドツーエンドのTensorRTグラフ
3. マルチストリーム推論の最適化

**期待効果**:
- レイテンシ: 70ms → **30-40ms**
- FPS: 14 FPS → **25-30 FPS**

### フェーズ3: フルTensorRTネイティブ（3-6ヶ月）

**前提条件**:
- TensorRT 10+の新機能リリース
- State Space Modelsのサポート

**実装内容**:
1. 完全な静的グラフ化
2. Triton Inference Serverとの統合
3. 複数GPU並列処理

**期待効果**:
- レイテンシ: 30ms → **15-20ms**
- FPS: 25 FPS → **50-60 FPS**
- マルチGPUスケーラビリティ

---

## 📚 参考リソース

### SAM3 公式
- [facebookresearch/sam3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM3 Video Predictor Example](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb)
- [Video API Usage](https://deepwiki.com/facebookresearch/sam3/4.1-video-api-usage)
- [Video Segmentation Guide](https://deepwiki.com/facebookresearch/sam3/4-video-segmentation)

### SAM2 ONNX実装
- [ONNX-SAM2-Segment-Anything](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything)
- [Roboflow: SAM 2 Video Segmentation](https://blog.roboflow.com/sam-2-video-segmentation/)
- [Ultralytics SAM 2 Documentation](https://docs.ultralytics.com/models/sam-2/)

### TensorRT最適化
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TensorRT-LLM (State Management)](https://github.com/NVIDIA/TensorRT-LLM)
- [LSTM State Management](https://saturncloud.io/glossary/stateful-lstm/)

### 研究論文
- [EfficientSAM3](https://arxiv.org/html/2511.15833v1)
- [DAM4SAM: Distractor-Aware Memory](https://github.com/jovanavidenovic/DAM4SAM)
- [SSMTrack: State Space Models for Tracking](https://ieeexplore.ieee.org/document/11223714/)

---

## ✅ 成功の定義

### 最小成功基準（Must Have）

1. ✅ **機能要件**:
   - テキストプロンプトベースのビデオセグメンテーション
   - フレーム間でのオブジェクト追跡
   - MP4ビデオファイル入力対応

2. ✅ **性能要件**:
   - 1080p @ **14+ FPS**（FP16）
   - PyTorchベースラインより **2倍以上**の高速化
   - 精度劣化 **< 5%**

3. ✅ **使いやすさ**:
   - 5行以下のコードで実行可能
   - ドキュメント完備
   - 動作サンプル3個以上

### 理想的成功基準（Should Have）

1. ⭐ **性能**: 1080p @ **20+ FPS**（INT8）
2. ⭐ **機能**: リアルタイムWebカメラ対応
3. ⭐ **拡張性**: 複数プロンプトの同時追跡

### 望ましい成功基準（Nice to Have）

1. 💎 **性能**: 1080p @ **30 FPS**（最適化後）
2. 💎 **機能**: インタラクティブなプロンプト編集
3. 💎 **統合**: Triton Inference Server対応

---

## 🎬 まとめ

### 実装可能性: ✅ **可能**

SAM3のビデオストリーミングトラッキング機能は、**ハイブリッドアプローチ（フェーズ1）により3-4週間で実装可能**です。

### 推奨アプローチ

1. **短期（3-4週間）**: フェーズ1（ハイブリッド）の実装
   - TensorRTで推論高速化
   - Pythonで状態管理
   - 目標: **14-21 FPS @ 1080p**

2. **中期（2-3ヶ月）**: フェーズ2への移行検討
   - カスタムプラグインでさらなる高速化
   - 目標: **25-30 FPS @ 1080p**

3. **長期（6-12ヶ月）**: フェーズ3の研究開発
   - 完全TensorRTネイティブ実装
   - 目標: **50-60 FPS @ 1080p**

### 次のステップ

1. ✅ **このプランのレビュー**と承認
2. 🔬 **SAM3公式リポジトリの詳細調査**（2-3日）
3. 🚀 **PoC（概念実証）の実装**（1週間）
4. 📈 **フル実装への移行**（残り2-3週間）

---

**プラン作成者**: Claude Code
**最終更新**: 2025年12月26日
**ステータス**: レビュー待ち
