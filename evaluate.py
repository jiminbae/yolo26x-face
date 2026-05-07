"""
WIDERFace 공식 벤치마크 평가 (Easy / Medium / Hard AP)
학습 후 yolo26x_widerface/weights/best.pt 기준으로 실행
"""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════
#  설정
# ══════════════════════════════════════════════════════════════════
WEIGHTS      = "runs/face/yolo26x_widerface/weights/best.pt"
VAL_IMG_DIR  = Path("datasets/widerface/raw/WIDER_val/images")
PRED_DIR     = Path("widerface_eval/predictions")
GT_DIR       = Path("widerface_eval/ground_truth")   # 공식 eval 코드 필요
IMGSZ        = 1280
CONF         = 0.001   # 낮게 설정 (AP 계산을 위해 많은 예측 필요)
IOU          = 0.45
DEVICE       = 0


# ══════════════════════════════════════════════════════════════════
#  예측 생성
# ══════════════════════════════════════════════════════════════════
def generate_predictions():
    print("=" * 60)
    print("  WIDERFace 예측 생성")
    print("=" * 60)

    model = YOLO(WEIGHTS)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # WIDERFace val 이벤트별로 처리
    event_dirs = sorted(VAL_IMG_DIR.iterdir())
    print(f"  이벤트 수: {len(event_dirs)}")

    for event_dir in tqdm(event_dirs, desc="이벤트 처리"):
        event_name = event_dir.name
        pred_event = PRED_DIR / event_name
        pred_event.mkdir(parents=True, exist_ok=True)

        img_files = sorted(event_dir.glob("*.jpg"))

        for img_path in img_files:
            results = model.predict(
                source  = str(img_path),
                imgsz   = IMGSZ,
                conf    = CONF,
                iou     = IOU,
                device  = DEVICE,
                verbose = False,
            )

            # WIDERFace 공식 포맷: x1 y1 w h score
            pred_txt = pred_event / (img_path.stem + ".txt")
            lines = [img_path.stem]

            if results and len(results[0].boxes):
                boxes = results[0].boxes
                xyxy  = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                lines.append(str(len(xyxy)))
                for (x1, y1, x2, y2), score in zip(xyxy, confs):
                    w = x2 - x1; h = y2 - y1
                    lines.append(f"{x1:.1f} {y1:.1f} {w:.1f} {h:.1f} {score:.4f}")
            else:
                lines.append("1")
                lines.append("0 0 0 0 0")

            pred_txt.write_text("\n".join(lines))

    print(f"  ✅ 예측 저장: {PRED_DIR}")


# ══════════════════════════════════════════════════════════════════
#  공식 평가 실행
# ══════════════════════════════════════════════════════════════════
def run_evaluation():
    """
    WIDERFace 공식 Python 평가 도구 사용
    https://github.com/wondervictor/WiderFace-Evaluation
    """
    print("\n" + "=" * 60)
    print("  WIDERFace 공식 AP 평가")
    print("=" * 60)

    eval_script = Path("WiderFace-Evaluation/evaluation.py")
    if not eval_script.exists():
        print("  공식 평가 도구 다운로드 중...")
        subprocess.run([
            "git", "clone",
            "https://github.com/wondervictor/WiderFace-Evaluation.git"
        ], check=True)
        subprocess.run(
            ["python", "setup.py", "build_ext", "--inplace"],
            cwd="WiderFace-Evaluation", check=True
        )

    # 평가 실행
    result = subprocess.run([
        "python", str(eval_script),
        "-p", str(PRED_DIR),
        "-g", str(GT_DIR),
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print("  오류:", result.stderr)


# ══════════════════════════════════════════════════════════════════
#  빠른 mAP 확인 (Ultralytics 내장)
# ══════════════════════════════════════════════════════════════════
def quick_val():
    print("=" * 60)
    print("  빠른 검증 (Ultralytics 내장 mAP)")
    print("=" * 60)

    model   = YOLO(WEIGHTS)
    results = model.val(
        data   = "datasets/widerface/widerface.yaml",
        imgsz  = IMGSZ,
        batch  = 32,
        device = DEVICE,
        conf   = 0.001,
        iou    = 0.6,
        plots  = True,
    )

    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │  mAP@0.5    : {results.box.map50:.4f}          │")
    print(f"  │  mAP@0.5:0.95: {results.box.map:.4f}          │")
    print(f"  │  Precision  : {results.box.mp:.4f}          │")
    print(f"  │  Recall     : {results.box.mr:.4f}          │")
    print(f"  └─────────────────────────────┘")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "official"], default="quick")
    parser.add_argument("--weights", type=str, default=WEIGHTS)
    args = parser.parse_args()

    WEIGHTS = args.weights

    if args.mode == "quick":
        quick_val()
    else:
        generate_predictions()
        run_evaluation()
