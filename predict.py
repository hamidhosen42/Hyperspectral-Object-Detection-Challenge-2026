"""Run a trained model on the test set and write the Kaggle submission CSV.

usage: python predict.py --weights runs/rgb_yolo11s/weights/best.pt --kind rgb --imgsz 1024 --out submission.csv
Boxes are emitted in cube-resolution pixel coordinates (the same frame the annotations use).
"""
import argparse, os, glob
import pandas as pd
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--weights', nargs='+', required=True)
ap.add_argument('--kind', choices=['rgb', 'hsi'], default='rgb')
ap.add_argument('--imgsz', type=int, default=1024)
ap.add_argument('--conf', type=float, default=0.001)
ap.add_argument('--iou', type=float, default=0.6)
ap.add_argument('--max_det', type=int, default=300)
ap.add_argument('--tta', action='store_true')
ap.add_argument('--device', default='mps')
ap.add_argument('--out', default='submission.csv')
ap.add_argument('--split', default='test')
a = ap.parse_args()

test_dir = os.path.join(ROOT, 'data/yolo', a.kind, 'images', a.split)
files = sorted(glob.glob(os.path.join(test_dir, '*')), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
print(len(files), 'test images')

rows = []
model = YOLO(a.weights[0])
for i in range(0, len(files), 16):
    res = model.predict(files[i:i + 16], imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=a.max_det,
                        augment=a.tta, device=a.device, verbose=False)
    for r in res:
        image_id = int(os.path.splitext(os.path.basename(r.path))[0])
        b = r.boxes
        for (x1, y1, x2, y2), c, s in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()):
            rows.append((image_id, int(c), float(s), float(x1), float(y1), float(x2), float(y2)))
    if i % 160 == 0: print(i, len(rows), flush=True)

df = pd.DataFrame(rows, columns=['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
df = df.sort_values(['image_id', 'confidence'], ascending=[True, False]).reset_index(drop=True)
df.insert(0, 'id', range(len(df)))
df[['confidence']] = df[['confidence']].round(5)
df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].round(2)
df.to_csv(a.out, index=False)
print('wrote', a.out, len(df), 'rows; images covered:', df.image_id.nunique())
