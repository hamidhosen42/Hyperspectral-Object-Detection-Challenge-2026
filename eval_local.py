"""Score a submission-style CSV on the local val split with pycocotools (COCO mAP@[.5:.95]).

usage: python eval_local.py --pred val_pred.csv
Ground truth is read from data/yolo/rgb/labels/val/*.txt (YOLO format) + image sizes.
"""
import argparse, os, glob, json
import numpy as np, pandas as pd, cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--pred', required=True); ap.add_argument('--max_det', type=int, default=100)
a = ap.parse_args()
CLASSES = [l.strip() for l in open(os.path.join(ROOT, 'data/class.txt')) if l.strip()]

images, anns = [], []
for p in sorted(glob.glob(os.path.join(ROOT, 'data/yolo/rgb/images/val/*.png'))):
    iid = int(os.path.splitext(os.path.basename(p))[0]); H, W = cv2.imread(p).shape[:2]
    images.append({'id': iid, 'width': W, 'height': H})
    lp = p.replace('/images/', '/labels/').replace('.png', '.txt')
    for l in open(lp):
        c, cx, cy, w, h = map(float, l.split())
        anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': int(c), 'bbox': [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H], 'area': w * W * h * H, 'iscrowd': 0})
gt = COCO(); gt.dataset = {'images': images, 'annotations': anns, 'categories': [{'id': i, 'name': n} for i, n in enumerate(CLASSES)]}; gt.createIndex()

df = pd.read_csv(a.pred)
dets = [{'image_id': int(r.image_id), 'category_id': int(r.class_id), 'score': float(r.confidence), 'bbox': [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1]} for r in df.itertuples()]
dt = gt.loadRes(dets)
E = COCOeval(gt, dt, 'bbox'); E.params.maxDets = [1, 10, a.max_det]; E.evaluate(); E.accumulate(); E.summarize()
print('per-class AP50-95:')
prec = E.eval['precision']  # [T,R,K,A,M]
for k, n in enumerate(CLASSES):
    pk = prec[:, :, k, 0, -1]; print(f'  {n:15s} {pk[pk > -1].mean():.3f}')
