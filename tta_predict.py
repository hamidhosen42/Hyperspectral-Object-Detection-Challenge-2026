"""Single-model TTA: run one checkpoint on the original and horizontally-flipped image (optionally extra
scales), un-flip the boxes and fuse all passes of the SAME model with Weighted Boxes Fusion.
Allowed by the competition rules (TTA/multi-scale of a single model is not an ensemble).

usage: python tta_predict.py --weights best.pt --kind rgb --imgsz 1024 --split val --out val_tta.csv [--scales 1.0 1.15]
"""
import argparse, os, glob, warnings
import numpy as np, pandas as pd, cv2
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
warnings.filterwarnings('ignore', message='Zero area box')

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--weights', required=True); ap.add_argument('--kind', default='rgb'); ap.add_argument('--imgsz', type=int, default=1024)
ap.add_argument('--split', default='test'); ap.add_argument('--out', required=True)
ap.add_argument('--scales', nargs='+', type=float, default=[1.0]); ap.add_argument('--no_flip', action='store_true')
ap.add_argument('--conf', type=float, default=0.001); ap.add_argument('--iou', type=float, default=0.6)
ap.add_argument('--wbf_iou', type=float, default=0.6); ap.add_argument('--conf_type', default='avg'); ap.add_argument('--device', default='mps')
a = ap.parse_args()

from ultralytics.utils.patches import imread
model = YOLO(a.weights)
files = sorted(glob.glob(os.path.join(ROOT, 'data/yolo', a.kind, 'images', a.split, '*')), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
rows = []
for i in range(0, len(files), 8):
    batch = files[i:i + 8]
    ims = [imread(p) for p in batch]
    passes = []  # list of (flip, scale, results)
    for sc in a.scales:
        sz = int(round(a.imgsz * sc / 32) * 32)
        for flip in ([False] if a.no_flip else [False, True]):
            inp = [np.ascontiguousarray(im[:, ::-1]) if flip else im for im in ims]
            res = model.predict(inp, imgsz=sz, conf=a.conf, iou=a.iou, max_det=300, device=a.device, verbose=False)
            passes.append((flip, res))
    for j, p in enumerate(batch):
        H, W = ims[j].shape[:2]; iid = int(os.path.splitext(os.path.basename(p))[0])
        bl, sl, ll = [], [], []
        for flip, res in passes:
            b = res[j].boxes; xyxy = b.xyxy.cpu().numpy().copy()
            if flip: xyxy[:, [0, 2]] = W - xyxy[:, [2, 0]]
            bl.append((xyxy / [W, H, W, H]).clip(0, 1).tolist()); sl.append(b.conf.cpu().numpy().tolist()); ll.append(b.cls.cpu().numpy().tolist())
        if sum(len(x) for x in bl) == 0: continue
        bb, ss, cc = weighted_boxes_fusion(bl, sl, ll, iou_thr=a.wbf_iou, skip_box_thr=a.conf, conf_type=a.conf_type)
        for (x1, y1, x2, y2), s, c in zip(bb, ss, cc):
            rows.append((iid, int(c), float(s), x1 * W, y1 * H, x2 * W, y2 * H))
    if i % 96 == 0: print(i, len(rows), flush=True)
df = pd.DataFrame(rows, columns=['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2']).sort_values(['image_id', 'confidence'], ascending=[True, False]).reset_index(drop=True)
df.insert(0, 'id', range(len(df))); df[['confidence']] = df[['confidence']].round(5); df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].round(2)
df.to_csv(a.out, index=False); print('wrote', a.out, len(df), 'rows,', df.image_id.nunique(), 'images')
