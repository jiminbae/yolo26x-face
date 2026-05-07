

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
# 기본 (300 epoch, imgsz=1280, batch=64)
python train.py --mode train

# 커스텀 설정
python train.py --mode train --epochs 200 --batch 32 --imgsz 1280
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
python train.py --mode export --weights runs/face/yolo26x_widerface/weights/best.pt
```

### Step 5. 추론

```bash
python train.py --mode predict --source path/to/image.jpg
```

---

## ⚙️ RTX PRO 6000 Blackwell 최적화 설정

| 파라미터 | 값 | 이유 |
|---|---|---|
| `imgsz` | 1280 | 소형 얼굴 탐지 강화 |
| `batch` | 64 | 96GB VRAM 최대 활용 |
| `workers` | 16 | 24,064 CUDA cores |
| `amp` | True | Blackwell BF16 Tensor Core |
| `cache` | ram | 빠른 데이터 로딩 |

---

## 예상 학습 시간

| 조건 | 시간 |
|---|---|
| RTX PRO 6000 Blackwell, 300 epoch, imgsz=1280 | **약 6~9시간** |
| RTX PRO 6000 Blackwell, 300 epoch, imgsz=640  | **약 3~5시간** |

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
