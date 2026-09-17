"""Fuse several submission-style CSVs with Weighted Boxes Fusion into one CSV.

usage: python ensemble.py --preds a.csv b.csv --weights 1 1 --iou 0.55 --skip 0.001 --out fused.csv
Image sizes are read from data/yolo/rgb/images/{split} so coordinates can be normalised for WBF.
"""
import argparse, os, glob
import numpy as np, pandas as pd, cv2, warnings
warnings.filterwarnings("ignore", message="Zero area box")
from ensemble_boxes import weighted_boxes_fusion

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--preds', nargs='+', required=True)
ap.add_argument('--weights', nargs='+', type=float, default=None)
ap.add_argument('--iou', type=float, default=0.55)
ap.add_argument('--skip', type=float, default=0.001)
ap.add_argument('--conf_type', default='avg')
ap.add_argument('--split', default='test')
ap.add_argument('--out', required=True)
a = ap.parse_args()

sizes = {}
for p in glob.glob(os.path.join(ROOT, 'data/yolo/rgb/images', a.split, '*.png')):
    H, W = cv2.imread(p).shape[:2]; sizes[int(os.path.splitext(os.path.basename(p))[0])] = (W, H)
dfs = [pd.read_csv(p) for p in a.preds]
rows = []
for iid, (W, H) in sorted(sizes.items()):
    bl, sl, ll = [], [], []
    for d in dfs:
        s = d[d.image_id == iid]
        b = np.stack([s.x1 / W, s.y1 / H, s.x2 / W, s.y2 / H], 1).clip(0, 1) if len(s) else np.zeros((0, 4))
        bl.append(b.tolist()); sl.append(s.confidence.tolist()); ll.append(s.class_id.tolist())
    if sum(len(x) for x in bl) == 0: continue
    b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=a.weights, iou_thr=a.iou, skip_box_thr=a.skip, conf_type=a.conf_type)
    for (x1, y1, x2, y2), sc, c in zip(b, s, l):
        rows.append((iid, int(c), float(sc), x1 * W, y1 * H, x2 * W, y2 * H))
df = pd.DataFrame(rows, columns=['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
df = df.sort_values(['image_id', 'confidence'], ascending=[True, False]).reset_index(drop=True)
df.insert(0, 'id', range(len(df)))
df[['confidence']] = df[['confidence']].round(5); df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].round(2)
df.to_csv(a.out, index=False); print('wrote', a.out, len(df), 'rows,', df.image_id.nunique(), 'images')
