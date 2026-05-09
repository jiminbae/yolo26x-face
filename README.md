# YOLO26x WIDERFace Face Detection

---

## 환경 구성

```bash
# 1. Python 환경 (3.10+ 권장)
conda create -n yolo26face python=3.11
conda activate yolo26face

# 2. PyTorch (Blackwell sm_100 지원)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 나머지 패키지
pip install -r requirements.txt
```

---

## 디렉토리 구조

```
yolo26x_widerface/
├── prepare_widerface.py   # 데이터셋 변환
├── train.py               # 학습 메인
├── evaluate.py            # WIDERFace 공식 평가
├── requirements.txt
├── datasets/
│   └── widerface/
│       ├── raw/           # ← 여기에 zip 파일 위치
│       │   ├── WIDER_train.zip
│       │   ├── WIDER_val.zip
│       │   └── wider_face_split.zip
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── widerface.yaml
└── runs/
    └── face/
        └── yolo26x_widerface/
            └── weights/
                ├── best.pt
                └── last.pt
```

---

## 학습된 모델

현재 로컬 학습 산출물은 아래 경로에 있습니다.

| 파일 | 크기 | 용도 |
|---|---:|---|
| `runs/detect/runs/face/yolo26x_widerface/weights/best.pt` | 113M | 검증 성능 기준 best checkpoint |
| `runs/detect/runs/face/yolo26x_widerface/weights/last.pt` | 113M | 마지막 checkpoint |
| `runs/detect/runs/face/yolo26x_widerface/weights/epoch*.pt` | 각 338M | 10 epoch 단위 중간 checkpoint |

GitHub 일반 Git은 단일 파일 100MB를 넘는 파일 push를 막습니다. 그래서 weight 파일은 repo에 직접 커밋하지 않고, GitHub Release asset 또는 Git LFS로 배포하는 것을 권장합니다. 추론 배포만 필요하면 PyTorch checkpoint 대신 ONNX로 export해서 배포할 수 있습니다. 단, ONNX 파일도 100MB를 넘으면 일반 Git push 제한은 동일하게 적용됩니다.

Release asset으로 올린 뒤에는 아래처럼 내려받아 사용합니다.

```bash
mkdir -p runs/face/yolo26x_widerface/weights
# 예시: release asset URL로 교체
curl -L -o runs/face/yolo26x_widerface/weights/best.pt <release-asset-url>
```

---

## 실행 순서

### Step 1. 데이터셋 준비

WIDERFace 공식 사이트에서 다운로드:
- https://shuoyang1213.me/WIDERFACE/

필요 파일을 `datasets/widerface/raw/` 에 위치:
- `WIDER_train.zip`
- `WIDER_val.zip`
- `wider_face_split.zip`

```bash
python prepare_widerface.py
```

### Step 2. 학습

```bash
# 기본 (100 epoch, imgsz=1280, batch=16)
python train.py --mode train

# 커스텀 설정
python train.py --mode train --epochs 200 --batch 32 --imgsz 1280

# 중단된 학습 재개
python train.py --mode train --resume

# 특정 체크포인트에서 재개
python train.py --mode train --resume-from runs/face/yolo26x_widerface/weights/last.pt
```

### Step 3. 평가

```bash
# 빠른 검증 (Ultralytics mAP)
python evaluate.py --mode quick

# WIDERFace 공식 평가 (Easy/Medium/Hard AP)
python evaluate.py --mode official
```

### Step 4. TensorRT 변환 (배포)

```bash
python train.py --mode export --weights runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

### Step 5. ONNX 변환 (범용 배포)

```bash
python train.py --mode export --format onnx --weights runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

### Step 6. 추론

```bash
python train.py --mode predict --source path/to/image.jpg
```

---

## ⚙️ RTX PRO 6000 Blackwell 최적화 설정

| 파라미터 | 값 | 이유 |
|---|---|---|
| `imgsz` | 1280 | 소형 얼굴 탐지 강화 |
| `batch` | 16 | 1280px 학습 안정성 우선 |
| `workers` | 8 | 안정적인 데이터 로딩 |
| `amp` | True | Blackwell BF16 Tensor Core |
| `cache` | ram | 빠른 데이터 로딩 |

---

## 예상 학습 시간

| 조건 | 시간 |
|---|---|
| RTX PRO 6000 Blackwell, 100 epoch, imgsz=1280 | 환경에 따라 변동 |
| RTX PRO 6000 Blackwell, 100 epoch, imgsz=640  | 환경에 따라 변동 |

---

## 예상 성능 (WIDERFace val)

| Subset | 예상 AP |
|---|---|
| Easy   | ~98% |
| Medium | ~97% |
| Hard   | ~93~94% |

---

## 트러블슈팅

**CUDA out of memory:**
```bash
python train.py --mode train --batch 32 --imgsz 1280
# 또는
python train.py --mode train --batch 64 --imgsz 640
```

**YOLO26 모델 없음:**
```python
from ultralytics import YOLO
model = YOLO("yolo26x.pt")  # 자동 다운로드
```
