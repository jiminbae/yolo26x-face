# YOLO26x-Face

YOLO26x-Face is a face detection model fine-tuned from Ultralytics YOLO26x on the WIDERFace dataset. The model is intended for face bounding-box detection in images and videos, with an emphasis on crowded scenes and small faces.

This repository contains the training, dataset conversion, evaluation, and export scripts. It does not include the WIDERFace dataset or model weights by default.

## Model Summary

| Name | Image Size (pixels) | mAPval 50-95 | Params | GFLOPs |
|---|---:|---:|---:|---:|
| YOLO26x-Face | 1280 | 52.84 | 58.99M | 838.1 |

The mAP value above is from Ultralytics validation on the local WIDERFace validation split. It is not the official WIDERFace Easy/Medium/Hard benchmark score.

## Training Setup

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| VRAM | 96GB GDDR7 |
| Training time | Approximately 12 hours |
| Base model | `yolo26x.pt` |
| Dataset | WIDERFace |
| Task | Face detection |
| Classes | 1 (`face`) |
| Image size | 1280 |
| Batch size | 16 |
| Completed epochs | 70 |
| Configured epochs | 100 |
| Precision | AMP enabled |

## Training Results

| Name | Training Time | Epochs | Batch Size | Non-default parameters | Link |
|---|---:|---:|---:|---|---|
| YOLO26x-Face | ~12 hours | 70 completed / 100 configured | 16 | `imgsz=1280`, `single_cls=True`, `cos_lr=True`, `close_mosaic=15`, `mosaic=1.0`, `mixup=0.15`, `copy_paste=0.1`, `save_period=10` | `runs/detect/runs/face/yolo26x_widerface/results.csv` |

Best local validation metrics observed in `results.csv`:

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 66 | 90.63 | 82.37 | 88.75 | 52.84 |

Last completed epoch:

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 70 | 90.66 | 82.18 | 88.65 | 52.64 |

## Evaluation Results on WIDERFace Dataset

Official WIDERFace evaluation reports AP separately for Easy, Medium, and Hard subsets. This model has not yet been submitted to the official WIDERFace evaluator, so no official Easy/Medium/Hard scores are claimed here.

| Name | Easy | Medium | Hard |
|---|---:|---:|---:|
| YOLO26x-Face | Not evaluated | Not evaluated | Not evaluated |
| YOLOv8n-Face baseline | 93.79 | 91.82 | 79.38 |

## Download Links

Model weights are larger than GitHub's normal 100MB per-file Git limit and should be distributed through GitHub Releases, Git LFS, or another artifact store. The links below should be filled after the files are uploaded as release assets.

| Name | Model Size (MB) | Link | SHA-256 |
|---|---:|---|---|
| YOLO26x-Face `best.pt` | 113 | TBD | `f749791ce9205e2df8bbb479201bce09b43423cac7f8e84d19cae6a01d0cea22` |
| YOLO26x-Face `last.pt` | 113 | TBD | `bae681e5ec385daab7863c5807a07d837ccf148f3788fdffb025ae1320a05c0a` |
| YOLO26x-Face ONNX | TBD | TBD | TBD |

To generate an ONNX model for deployment:

```bash
python train.py --mode export --format onnx --weights runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

If the exported ONNX file is under 100MB, it may be committed directly to Git. If it is over 100MB, distribute it as a release asset or with Git LFS.

## Installation

Python 3.10 or newer is recommended.

```bash
conda create -n yolo26face python=3.11
conda activate yolo26face

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Dataset Preparation

Download WIDERFace from the official dataset page:

https://shuoyang1213.me/WIDERFACE/

Place the downloaded archives here:

```text
datasets/widerface/raw/
|-- WIDER_train.zip
|-- WIDER_val.zip
`-- wider_face_split.zip
```

Convert the dataset to YOLO format:

```bash
python prepare_widerface.py
```

Expected dataset layout after conversion:

```text
datasets/widerface/
|-- images/
|   |-- train/
|   `-- val/
|-- labels/
|   |-- train/
|   `-- val/
`-- widerface.yaml
```

## Training

Start training:

```bash
python train.py --mode train
```

Override common settings:

```bash
python train.py --mode train --epochs 100 --batch 16 --imgsz 1280
```

Resume from the latest detected checkpoint:

```bash
python train.py --mode train --resume
```

Resume from a specific checkpoint:

```bash
python train.py --mode train --resume-from runs/detect/runs/face/yolo26x_widerface/weights/last.pt
```

## Evaluation

Run Ultralytics validation:

```bash
python evaluate.py --mode quick
```

Run the WIDERFace-format evaluation workflow:

```bash
python evaluate.py --mode official
```

Official Easy/Medium/Hard WIDERFace scores should only be reported after running the official evaluation protocol or submitting predictions to the benchmark evaluator.

## Export

Export to TensorRT:

```bash
python train.py --mode export --format engine --weights runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

Export to ONNX:

```bash
python train.py --mode export --format onnx --weights runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

## Inference

Run prediction on an image, directory, video, or stream supported by Ultralytics:

```bash
python train.py --mode predict --weights runs/detect/runs/face/yolo26x_widerface/weights/best.pt --source path/to/image.jpg
```

## Repository Policy

The repository intentionally excludes large generated artifacts:

```text
datasets/
runs/
*.pt
*.pth
*.ckpt
venv/
```

Use release assets, Git LFS, or an external artifact store for trained weights and exported models.

## License and Dataset Compliance

This project uses Ultralytics YOLO tooling and a YOLO26 pretrained model. Ultralytics YOLO software and trained models are subject to Ultralytics licensing terms, including AGPL-3.0 by default or an Enterprise License for use cases that require different commercial terms. Users are responsible for ensuring their use, redistribution, and deployment of derived weights complies with the applicable Ultralytics license.

This repository does not redistribute WIDERFace images, annotations, or archives. Users must download WIDERFace from the official dataset source and comply with the dataset's terms of use. Do not commit or redistribute the dataset files through this repository.

If you use WIDERFace, cite:

```bibtex
@inproceedings{yang2016wider,
  author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
  title = {WIDER FACE: A Face Detection Benchmark},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2016}
}
```

Relevant policy and documentation links:

- Ultralytics license: https://www.ultralytics.com/license
- Ultralytics documentation: https://docs.ultralytics.com/
- WIDERFace dataset: https://shuoyang1213.me/WIDERFACE/
