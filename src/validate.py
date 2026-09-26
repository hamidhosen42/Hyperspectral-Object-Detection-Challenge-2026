"""Unified validation for every experiment (same GT, same splits, pycocotools = the official metric).

Scores a submission-style prediction CSV (image_id,class_id,confidence,x1,y1,x2,y2 in cube pixels) on
  ALL    = the 300-image hold-out used by every training run (seed-0 split, splits/val_all.txt)
  STRESS = the 100 most test-like hold-out images (splits/val_stress.txt, see src/stress_split.py)
and reports mAP50-95, AP50, AP75, AP small/medium/large, per-class AP and real-vs-fake pair confusion.

usage: python src/validate.py --pred kaggle_out/hod-y26s-1024/val_pred.csv [--name E0b] [--json out.json] [--gt kaggle_out/hod-val/val_gt.json]
"""
import argparse, os, json, contextlib, io
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, 'data/data_train/data_train/Annotations/VIS')
CLASSES = ['apple', 'apple_plastic', 'badminton', 'banana', 'banana_plastic', 'car', 'car_toy', 'charger_head', 'e-bike', 'egg',
           'egg_plastic', 'egg_wood', 'orange', 'orange_plastic', 'people', 'rubik', 'stone_block', 'table_tennis']  # data/class.txt order
PAIRS = {'apple': ['apple', 'apple_plastic'], 'banana': ['banana', 'banana_plastic'], 'orange': ['orange', 'orange_plastic'],
         'egg': ['egg', 'egg_plastic', 'egg_wood'], 'car': ['car', 'car_toy']}


def read_boxes(stems, gt_json=None):
    """{stem: (W, H, [(name, x1, y1, x2, y2), ...])} from the XML files or from a val_gt.json exported by the val kernel."""
    if gt_json:
        g = json.load(open(gt_json)); return {s: (g[s]['W'], g[s]['H'], [tuple(b) for b in g[s]['boxes']]) for s in stems}
    out = {}
    for s in stems:
        r = ET.parse(f'{ANN}/{s}.xml').getroot()
        out[s] = (int(r.find('size/width').text), int(r.find('size/height').text),
                  [(o.find('name').text.strip(), *[float(o.find('bndbox').find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]) for o in r.findall('object')])
    return out


def load_gt(stems, gt_json=None):
    images, anns = [], []
    for s, (W, H, boxes) in read_boxes(stems, gt_json).items():
        images.append({'id': int(s), 'width': W, 'height': H})
        for name, x1, y1, x2, y2 in boxes:  # identical clipping/filtering to the training label writer
            x1, x2 = sorted((max(0, x1), min(W, x2))); y1, y2 = sorted((max(0, y1), min(H, y2)))
            if x2 - x1 < 1 or y2 - y1 < 1: continue
            anns.append({'id': len(anns) + 1, 'image_id': int(s), 'category_id': CLASSES.index(name),
                         'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0})
    gt = COCO(); gt.dataset = {'images': images, 'annotations': anns, 'categories': [{'id': i, 'name': n} for i, n in enumerate(CLASSES)]}
    with contextlib.redirect_stdout(io.StringIO()): gt.createIndex()
    return gt


def coco_eval(gt, df, img_ids):
    d = df[df.image_id.isin(img_ids)]
    if not len(d): return None
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes([{'image_id': int(r.image_id), 'category_id': int(r.class_id), 'score': float(r.confidence),
                          'bbox': [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1]} for r in d.itertuples()])
        E = COCOeval(gt, dt, 'bbox'); E.params.imgIds = sorted(img_ids); E.evaluate(); E.accumulate(); E.summarize()
    p = E.eval['precision']  # [T, R, K, A, M]
    pc = {}
    for k, n in enumerate(CLASSES):
        x = p[:, :, k, 0, -1]; pc[n] = round(float(x[x > -1].mean()), 4) if (x > -1).any() else None
    s = E.stats
    return {'mAP50-95': round(s[0], 4), 'AP50': round(s[1], 4), 'AP75': round(s[2], 4), 'APs': round(s[3], 4), 'APm': round(s[4], 4),
            'APl': round(s[5], 4), 'per_class': pc,
            'pair_AP': {g: round(float(np.mean([pc[c] for c in cl if pc[c] is not None])), 4) for g, cl in PAIRS.items()}}


def pair_confusion(gt, df, img_ids, conf=0.25):
    """For GT boxes of a real/fake group: which class does the best-overlapping confident same-group prediction carry?"""
    out = {}
    d = df[(df.confidence >= conf) & df.image_id.isin(img_ids)]; by_img = dict(tuple(d.groupby('image_id')))
    for g, cl in PAIRS.items():
        ids = [CLASSES.index(c) for c in cl]; M = np.zeros((len(cl), len(cl) + 1), int)  # last col = missed
        for a in gt.dataset['annotations']:
            if a['image_id'] not in img_ids or a['category_id'] not in ids: continue
            p = by_img.get(a['image_id']); i = ids.index(a['category_id'])
            if p is None: M[i, -1] += 1; continue
            p = p[p.class_id.isin(ids)]
            x, y, w, h = a['bbox']
            iw = np.clip(np.minimum(p.x2, x + w) - np.maximum(p.x1, x), 0, None); ih = np.clip(np.minimum(p.y2, y + h) - np.maximum(p.y1, y), 0, None)
            iou = iw * ih / ((p.x2 - p.x1) * (p.y2 - p.y1) + w * h - iw * ih)
            ok = p[iou >= 0.5]
            if not len(ok): M[i, -1] += 1; continue
            M[i, ids.index(int(ok.sort_values('confidence').class_id.iloc[-1]))] += 1
        correct = int(np.trace(M[:, :-1])); matched = int(M[:, :-1].sum())
        out[g] = {'gt': cl, 'rows=gt, cols=pred(+missed)': M.tolist(), 'material_acc_on_matched': round(correct / max(matched, 1), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pred', required=True); ap.add_argument('--name', default=None); ap.add_argument('--json', default=None); ap.add_argument('--gt', default=None)
    a = ap.parse_args()
    val = [l.strip() for l in open(os.path.join(ROOT, 'splits/val_all.txt')) if l.strip()]
    stress = {int(l) for l in open(os.path.join(ROOT, 'splits/val_stress.txt')) if l.strip()}
    gt = load_gt(val, a.gt); df = pd.read_csv(a.pred)
    allv = {int(s) for s in val}
    res = {'name': a.name or os.path.basename(a.pred), 'pred': a.pred, 'ALL': coco_eval(gt, df, allv), 'STRESS': coco_eval(gt, df, stress),
           'pair_confusion_ALL': pair_confusion(gt, df, allv), 'n_pred_rows': len(df[df.image_id.isin(allv)])}
    r = res['ALL']; s = res['STRESS']
    print(f"{res['name']}: ALL mAP {r['mAP50-95']:.4f} AP50 {r['AP50']:.4f} AP75 {r['AP75']:.4f} APs {r['APs']:.4f} | "
          f"STRESS mAP {s['mAP50-95']:.4f} AP75 {s['AP75']:.4f} | pairs " + ' '.join(f"{g}:{v:.3f}" for g, v in r['pair_AP'].items()) +
          ' | material acc ' + ' '.join(f"{g}:{v['material_acc_on_matched']:.3f}" for g, v in res['pair_confusion_ALL'].items()))
    if a.json: json.dump(res, open(a.json, 'w'), indent=1)
    return res


if __name__ == '__main__':
    main()
