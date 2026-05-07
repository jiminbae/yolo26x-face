"""
WIDERFace → YOLO26x 포맷 변환 스크립트
RTX PRO 6000 Blackwell (96GB VRAM) 최적화

디렉토리 구조:
datasets/widerface/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
"""

import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATASET_ROOT = Path("datasets/widerface")
MIN_FACE_PX   = 8      # 이 픽셀 미만 얼굴은 노이즈로 제거
IMG_SIZE      = 1280   # 학습 해상도 (소형 얼굴 탐지 강화)


# ──────────────────────────────────────────────
# 1. 다운로드 헬퍼
# ──────────────────────────────────────────────
def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} 이미 존재")
        return
    print(f"  다운로드: {url}")
    r = requests.get(url, stream=True, timeout=60)
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, out_dir: Path):
    print(f"  압축 해제: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


# ──────────────────────────────────────────────
# 2. WIDERFace 어노테이션 파싱
# ──────────────────────────────────────────────
def parse_wider_annotation(anno_file: Path):
    """
    WIDERFace .mat 대신 텍스트 어노테이션 파싱
    반환: {img_path: [(x1,y1,w,h), ...]}
    """
    samples = {}
    with open(anno_file, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    i = 0
    while i < len(lines):
        img_name = lines[i]; i += 1
        num_faces = int(lines[i]); i += 1

        boxes = []
        if num_faces == 0:
            i += 1  # '0 0 0 0 0' 더미 라인 스킵
        else:
            for _ in range(num_faces):
                parts = list(map(int, lines[i].split())); i += 1
                x1, y1, w, h = parts[0], parts[1], parts[2], parts[3]
                boxes.append((x1, y1, w, h))

        samples[img_name] = boxes

    return samples


# ──────────────────────────────────────────────
# 3. YOLO 포맷 변환
# ──────────────────────────────────────────────
def convert_to_yolo(samples: dict, img_src_dir: Path, split: str):
    """WIDERFace bbox → YOLO 정규화 포맷 변환"""
    from PIL import Image

    img_out = DATASET_ROOT / "images"  / split
    lbl_out = DATASET_ROOT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    skipped_img  = 0
    skipped_face = 0
    total_faces  = 0

    for rel_path, boxes in tqdm(samples.items(), desc=f"  변환 [{split}]"):
        src = img_src_dir / rel_path
        if not src.exists():
            skipped_img += 1
            continue

        # 이미지 크기 확인
        with Image.open(src) as im:
            W, H = im.size

        yolo_lines = []
        for (x1, y1, w, h) in boxes:
            # 너무 작은 얼굴 제거
            if w < MIN_FACE_PX or h < MIN_FACE_PX:
                skipped_face += 1
                continue

            # 클리핑
            x1 = max(0, x1); y1 = max(0, y1)
            w  = min(w, W - x1); h = min(h, H - y1)

            # YOLO: cx cy w h (정규화)
            cx = (x1 + w / 2) / W
            cy = (y1 + h / 2) / H
            nw = w / W
            nh = h / H

            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            total_faces += 1

        if not yolo_lines:
            continue  # 유효 얼굴 없는 이미지 스킵

        # 이미지 복사
        dst_img = img_out / rel_path.replace("/", "_")
        shutil.copy(src, dst_img)

        # 레이블 저장
        lbl_name = dst_img.stem + ".txt"
        (lbl_out / lbl_name).write_text("\n".join(yolo_lines))

    print(f"  [{split}] 총 얼굴: {total_faces}, "
          f"스킵(이미지): {skipped_img}, 스킵(소형 얼굴): {skipped_face}")


# ──────────────────────────────────────────────
# 4. data.yaml 생성
# ──────────────────────────────────────────────
def write_yaml():
    yaml_path = DATASET_ROOT / "widerface.yaml"
    content = f"""# WIDERFace - YOLO26x Face Detection
# RTX PRO 6000 Blackwell (96GB) 최적화

path: {DATASET_ROOT.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: face
"""
    yaml_path.write_text(content)
    print(f"  data.yaml 저장: {yaml_path}")


# ──────────────────────────────────────────────
# 5. 메인
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  WIDERFace → YOLO26x 포맷 변환")
    print("=" * 60)

    # ── 다운로드 경로 ──
    raw_dir = DATASET_ROOT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] 데이터셋 다운로드")
    print("  아래 링크에서 수동 다운로드 후 raw/ 폴더에 위치시켜 주세요:")
    print("  https://shuoyang1213.me/WIDERFACE/")
    print("  필요 파일:")
    print("    - WIDER_train.zip")
    print("    - WIDER_val.zip")
    print("    - wider_face_split.zip")

    # ── 압축 해제 ──
    print("\n[2/4] 압축 해제")
    for fname in ["WIDER_train.zip", "WIDER_val.zip", "wider_face_split.zip"]:
        zp = raw_dir / fname
        if not zp.exists():
            print(f"  오류: {fname} 없음. raw/ 폴더에 위치시켜 주세요.")
            return
        extract_zip(zp, raw_dir)

    # ── 어노테이션 파싱 ──
    print("\n[3/4] 어노테이션 파싱 및 변환")
    anno_dir = raw_dir / "wider_face_split"

    for split, anno_name, img_subdir in [
        ("train", "wider_face_train_bbx_gt.txt",  "WIDER_train/images"),
        ("val",   "wider_face_val_bbx_gt.txt",    "WIDER_val/images"),
    ]:
        anno_file = anno_dir / anno_name
        img_dir   = raw_dir  / img_subdir
        samples   = parse_wider_annotation(anno_file)
        convert_to_yolo(samples, img_dir, split)

    # ── YAML 생성 ──
    print("\n[4/4] data.yaml 생성")
    write_yaml()

    # ── 통계 출력 ──
    print("\n✅ 변환 완료!")
    for split in ["train", "val"]:
        n_img = len(list((DATASET_ROOT / "images" / split).glob("*.jpg")))
        n_lbl = len(list((DATASET_ROOT / "labels" / split).glob("*.txt")))
        print(f"  [{split}] 이미지: {n_img}, 레이블: {n_lbl}")


if __name__ == "__main__":
    main()
