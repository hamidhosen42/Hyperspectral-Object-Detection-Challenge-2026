"""Fine-tune a YOLO detector on the pseudo-RGB (3ch) or full hyperspectral (16ch) images.

usage: python train.py --kind rgb --model yolo11s.pt --imgsz 1024 --epochs 60 --batch 8 --name rgb_s
"""
import argparse, os
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--kind', choices=['rgb', 'hsi'], default='rgb')
ap.add_argument('--model', default='yolo11s.pt')
ap.add_argument('--imgsz', type=int, default=1024)
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--name', default=None)
ap.add_argument('--device', default='mps')
ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--extra', nargs='*', default=[], help='key=value overrides passed to YOLO.train')
a = ap.parse_args()

extra = {}
for kv in a.extra:
    k, v = kv.split('=', 1)
    try: v = eval(v)
    except Exception: pass
    extra[k] = v

model = YOLO(a.model)
model.train(
    data=os.path.join(ROOT, 'data/yolo', f'data_{a.kind}.yaml'),
    imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device, workers=a.workers,
    project=os.path.join(ROOT, 'runs'), name=a.name or f'{a.kind}_{os.path.splitext(os.path.basename(a.model))[0]}',
    exist_ok=True, resume=a.resume,
    # hyperspectral bands are not RGB: colour-space augmentations are meaningless/harmful for 16ch
    hsv_h=0.0 if a.kind == 'hsi' else 0.015, hsv_s=0.0 if a.kind == 'hsi' else 0.7, hsv_v=0.4,
    bgr=0.0, fliplr=0.5, mosaic=1.0, close_mosaic=10, scale=0.5, degrees=0.0,
    patience=25, plots=True, cache=False, amp=False,
    **extra,
)
