"""
YOLO26x WIDERFace 학습 스크립트
GPU: NVIDIA RTX PRO 6000 Blackwell (96GB GDDR7)

최적화 포인트:
  - 96GB VRAM → batch=64, imgsz=1280 가능
  - Blackwell 5th-gen Tensor Core → AMP BF16
  - 24,064 CUDA cores → workers=16
  - MuSGD 옵티마이저 (YOLO26 기본)
  - STAL: 소형 얼굴 탐지 강화
"""

import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════
#  하이퍼파라미터 설정
# ══════════════════════════════════════════════════════════════════
CFG = {
    # ── 모델 ──────────────────────────────────────────────────────
    "model":        "yolo26x.pt",       # YOLO26x pretrained (COCO)
    "data":         "datasets/widerface/widerface.yaml",
    "project":      "runs/face",
    "name":         "yolo26x_widerface",

    # ── 해상도 ────────────────────────────────────────────────────
    # 1280: 소형 얼굴 탐지 대폭 향상 (96GB VRAM 여유 있음)
    "imgsz":        1280,

    # ── 학습 설정 ─────────────────────────────────────────────────
    "epochs":       100,
    "patience":     20,         # 조기 종료 patience
    "resume":       False,     # 중단된 학습 재개 여부
    "resume_from":  None,      # 명시적으로 재개할 체크포인트
    "batch":        16,         # 1280px 기준 안정적 기본값
    "workers":      8,         # 24,064 CUDA cores 최대 활용
    "device":       0,          # GPU 0

    # ── 옵티마이저 ────────────────────────────────────────────────
    # YOLO26 기본 MuSGD (LLM 훈련에서 영감, 수렴 안정)
    "optimizer":    "auto",     # YOLO26은 MuSGD 자동 적용
    "lr0":          0.01,       # 초기 lr (batch=64 기준 적합)
    "lrf":          0.01,       # 최종 lr = lr0 * lrf
    "momentum":     0.937,
    "weight_decay": 0.0005,
    "warmup_epochs":      3.0,
    "warmup_momentum":    0.8,
    "warmup_bias_lr":     0.1,

    # ── Loss weights ──────────────────────────────────────────────
    # face detection 특화 (단일 클래스)
    "box":          7.5,        # bbox 정확도 최우선
    "cls":          0.3,        # 단일 클래스 → 낮게
    "dfl":          1.5,

    # ── Augmentation ─────────────────────────────────────────────
    "mosaic":       1.0,        # 소형 얼굴 강화 핵심
    "mixup":        0.15,       # 다양성 추가
    "close_mosaic": 15,         # 마지막 15 epoch mosaic off (안정화)
    "degrees":      10.0,       # 회전 (얼굴은 소폭만)
    "translate":    0.1,
    "scale":        0.7,        # 다양한 스케일
    "shear":        2.0,
    "perspective":  0.0,        # 얼굴은 원근 변환 최소화
    "flipud":       0.0,        # 상하 반전 없음 (얼굴)
    "fliplr":       0.5,        # 좌우 반전
    "hsv_h":        0.015,
    "hsv_s":        0.7,
    "hsv_v":        0.4,
    "erasing":      0.4,        # 랜덤 지우기 (occlusion 시뮬레이션)
    "copy_paste":   0.1,        # copy-paste (군중 장면 강화)

    # ── 정밀도 ────────────────────────────────────────────────────
    # Blackwell 5th-gen Tensor Core: BF16이 FP16보다 안정적
    "amp":          True,       # Automatic Mixed Precision

    # ── 체크포인트 ────────────────────────────────────────────────
    "save":         True,
    "save_period":  10,         # 10 epoch마다 중간 저장
    "val":          True,
    "plots":        True,

    # ── 기타 ──────────────────────────────────────────────────────
    "cache":        False,      # 96GB VRAM + 충분한 RAM → RAM 캐싱
    "exist_ok":     False,
    "pretrained":   True,
    "verbose":      True,
    "seed":         42,
    "deterministic": False,     # 속도 우선
    "single_cls":   True,       # 단일 클래스 (face) 강제
    "rect":         False,      # rectangular training off (mosaic와 충돌)
    "cos_lr":       True,       # cosine lr schedule
    "label_smoothing": 0.0,     # face detection은 0
    "nbs":          64,         # nominal batch size (lr 스케일링 기준)
    "overlap_mask": True,
    "mask_ratio":   4,
    "dropout":      0.0,
    "fraction":     1.0,        # 전체 데이터 사용
}


def find_latest_checkpoint() -> Path | None:
    candidates = list(Path("runs").glob("**/weights/last.pt"))
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


# ══════════════════════════════════════════════════════════════════
#  GPU 환경 확인
# ══════════════════════════════════════════════════════════════════
def check_gpu():
    print("=" * 60)
    print("  GPU 환경 확인")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc       = torch.cuda.get_device_capability(0)

    print(f"  GPU    : {gpu_name}")
    print(f"  VRAM   : {vram_gb:.1f} GB")
    print(f"  Compute: {cc[0]}.{cc[1]}")
    print(f"  BF16   : {torch.cuda.is_bf16_supported()}")
    print(f"  Torch  : {torch.__version__}")

    # Blackwell = Compute 10.0+
    if cc[0] >= 10:
        print(f"  ✅ Blackwell 아키텍처 감지 → BF16 최적화 적용")
    elif cc[0] >= 8:
        print(f"  ✅ Ampere/Ada 아키텍처 감지")

    # VRAM 기준 batch 자동 조정
    if vram_gb < 24:
        print(f"  ⚠️  VRAM {vram_gb:.0f}GB: batch=16, imgsz=640으로 자동 조정")
        CFG["batch"] = 16
        CFG["imgsz"] = 640
    elif vram_gb < 48:
        print(f"  VRAM {vram_gb:.0f}GB: batch=32, imgsz=1280으로 자동 조정")
        CFG["batch"] = 32
        CFG["imgsz"] = 1280
    else:
        print(f"  ✅ VRAM {vram_gb:.0f}GB: batch={CFG['batch']}, "
              f"imgsz={CFG['imgsz']} 그대로 사용")

    print()


# ══════════════════════════════════════════════════════════════════
#  데이터셋 확인
# ══════════════════════════════════════════════════════════════════
def check_dataset():
    print("=" * 60)
    print("  데이터셋 확인")
    print("=" * 60)

    yaml = Path(CFG["data"])
    if not yaml.exists():
        raise FileNotFoundError(
            f"{yaml} 없음.\n"
            "먼저 python prepare_widerface.py 를 실행하세요."
        )

    root = yaml.parent
    for split in ["train", "val"]:
        imgs = list((root / "images" / split).glob("*.jpg"))
        lbls = list((root / "labels" / split).glob("*.txt"))
        print(f"  [{split}] 이미지: {len(imgs):,}, 레이블: {len(lbls):,}")

    print()


# ══════════════════════════════════════════════════════════════════
#  학습
# ══════════════════════════════════════════════════════════════════
def train():
    check_gpu()
    check_dataset()

    print("=" * 60)
    print("  YOLO26x WIDERFace 학습 시작")
    print("=" * 60)
    print(f"  모델  : {CFG['model']}")
    print(f"  imgsz : {CFG['imgsz']}")
    print(f"  batch : {CFG['batch']}")
    print(f"  epochs: {CFG['epochs']}")
    print(f"  출력  : {CFG['project']}/{CFG['name']}")
    print()

    # resume 시에는 명시된 checkpoint를 우선 사용
    model_path = Path(CFG["model"])
    if CFG["resume_from"]:
        model_path = Path(CFG["resume_from"])
        print(f"  resume-from 체크포인트: {model_path}")
        CFG["resume"] = True
        CFG["workers"] = min(CFG["workers"], 4)
    elif CFG["resume"]:
        checkpoint = find_latest_checkpoint()
        if checkpoint is not None:
            print(f"  resume 체크포인트: {checkpoint}")
            model_path = checkpoint
            CFG["workers"] = min(CFG["workers"], 4)
        else:
            print("  ⚠️  resume=True 이지만 last.pt를 찾지 못해 새 학습을 시작합니다.")
            CFG["resume"] = False

    # 모델 로드
    model = YOLO(str(model_path))

    # 학습
    t0 = time.time()
    train_cfg = dict(CFG)
    train_cfg.pop("resume_from", None)
    try:
        results = model.train(**train_cfg)
    except RuntimeError as exc:
        if "CUBLAS_STATUS_ALLOC_FAILED" not in str(exc):
            raise

        print("  ⚠️  CUDA/cuBLAS 메모리 할당 실패 → AMP 끄고 workers=0으로 1회 재시도합니다.")
        torch.cuda.empty_cache()

        fallback_cfg = dict(train_cfg)
        fallback_cfg["amp"] = False
        fallback_cfg["workers"] = 0
        results = model.train(**fallback_cfg)
    elapsed = time.time() - t0

    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    print(f"\n✅ 학습 완료! 소요 시간: {h}h {m}m")
    print(f"   best.pt: {CFG['project']}/{CFG['name']}/weights/best.pt")

    return results


# ══════════════════════════════════════════════════════════════════
#  검증
# ══════════════════════════════════════════════════════════════════
def validate(weights: str = None):
    if weights is None:
        weights = f"{CFG['project']}/{CFG['name']}/weights/best.pt"

    print("=" * 60)
    print("  검증")
    print("=" * 60)
    print(f"  가중치: {weights}")

    model = YOLO(weights)
    results = model.val(
        data    = CFG["data"],
        imgsz   = CFG["imgsz"],
        batch   = CFG["batch"],
        device  = CFG["device"],
        split   = "val",
        plots   = True,
        save_json = True,
    )

    print(f"\n  mAP50    : {results.box.map50:.4f}")
    print(f"  mAP50-95 : {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall   : {results.box.mr:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════
#  모델 변환 (배포용)
# ══════════════════════════════════════════════════════════════════
def export_model(weights: str = None, export_format: str = "engine"):
    if weights is None:
        weights = f"{CFG['project']}/{CFG['name']}/weights/best.pt"

    print("=" * 60)
    print(f"  {export_format.upper()} 변환")
    print("=" * 60)

    model = YOLO(weights)
    export_kwargs = {
        "format": export_format,
        "imgsz": CFG["imgsz"],
        "device": CFG["device"],
        "simplify": True,
    }

    if export_format == "engine":
        export_kwargs["half"] = True
        export_kwargs["workspace"] = 8

    model.export(**export_kwargs)
    print(f"  ✅ {export_format.upper()} 저장 완료")


# ══════════════════════════════════════════════════════════════════
#  추론 예시
# ══════════════════════════════════════════════════════════════════
def predict(source: str, weights: str = None):
    if weights is None:
        weights = f"{CFG['project']}/{CFG['name']}/weights/best.pt"

    model = YOLO(weights)
    results = model.predict(
        source     = source,
        imgsz      = CFG["imgsz"],
        conf       = 0.25,
        iou        = 0.45,
        max_det    = 1000,      # 군중 장면 대비
        device     = CFG["device"],
        save       = True,
        save_txt   = True,
        line_width = 1,
    )
    return results


# ══════════════════════════════════════════════════════════════════
#  엔트리포인트
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO26x WIDERFace 학습")
    parser.add_argument(
        "--mode",
        choices=["train", "val", "export", "predict"],
        default="train",
        help="실행 모드"
    )
    parser.add_argument("--weights", type=str, default=None, help="가중치 경로")
    parser.add_argument(
        "--format",
        choices=["engine", "onnx"],
        default="engine",
        help="export 포맷",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="지정한 체크포인트에서 학습 재개 (예: runs/.../weights/epoch10.pt)",
    )
    parser.add_argument("--source",  type=str, default=None, help="추론 소스")
    parser.add_argument("--epochs",  type=int, default=None, help="학습 epoch 수")
    parser.add_argument("--batch",   type=int, default=None, help="배치 크기")
    parser.add_argument("--imgsz",   type=int, default=None, help="이미지 크기")
    parser.add_argument("--resume", action="store_true", help="이전 체크포인트에서 학습 재개")
    args = parser.parse_args()

    # 인자 오버라이드
    if args.epochs:  CFG["epochs"] = args.epochs
    if args.batch:   CFG["batch"]  = args.batch
    if args.imgsz:   CFG["imgsz"]  = args.imgsz
    if args.weights: CFG["model"]  = args.weights
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume-from 경로가 없습니다: {resume_path}")
        CFG["resume_from"] = str(resume_path)
    if args.resume:  CFG["resume"] = True

    if args.mode == "train":
        train()
    elif args.mode == "val":
        validate(args.weights)
    elif args.mode == "export":
        export_model(args.weights, args.format)
    elif args.mode == "predict":
        if not args.source:
            raise ValueError("--source 필요")
        predict(args.source, args.weights)
