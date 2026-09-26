"""Pre-submission checks for a Phase 2 CSV.
usage: python src/check_submission.py <phase2.csv> [--ref submission_rgb_ep40.csv]
- exact columns, unique 0-based ids, numeric sanity, boxes inside frame
- image coverage: test ids (taken from the reference/old submission) vs the rest (= ranking)
- with --ref: agreement of the test part with a previous submission of the SAME model (pipeline equivalence check)
"""
import sys, argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('csv'); ap.add_argument('--ref', default=None); ap.add_argument('--test_ids', default=None)
a = ap.parse_args()
d = pd.read_csv(a.csv)
ok = True
def req(cond, msg):
    global ok
    print(('  OK   ' if cond else '  FAIL ') + msg); ok &= bool(cond)

req(list(d.columns) == ['id', 'image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'], f'columns {list(d.columns)}')
req(d.id.is_unique and d.id.min() == 0 and d.id.max() == len(d) - 1, f'id 0..{len(d) - 1} unique')
req(d.notna().all().all(), 'no NaN')
req(d.class_id.between(0, 17).all(), f'class_id in 0..17 (seen {sorted(d.class_id.unique())[:3]}..{d.class_id.max()})')
req(d.confidence.between(0, 1).all(), 'confidence in [0,1]')
req(((d.x2 > d.x1) & (d.y2 > d.y1)).all(), 'x2>x1, y2>y1')
req((d[['x1', 'y1']] >= -1).all().all() and d.x2.max() < 700 and d.y2.max() < 400, f'boxes in cube frame (max x2 {d.x2.max():.0f}, max y2 {d.y2.max():.0f})')
n_img = d.image_id.nunique(); print(f'  rows {len(d)}, images {n_img}, rows/img {len(d) / n_img:.1f}')
req(n_img >= 1990, 'covers ~2000 images (test + ranking)')
if a.ref:
    r = pd.read_csv(a.ref); tid = set(r.image_id)
    t = d[d.image_id.isin(tid)]; rk = d[~d.image_id.isin(tid)]
    print(f'  test part: {t.image_id.nunique()} images (ref had {len(tid)}), ranking part: {rk.image_id.nunique()} images')
    req(rk.image_id.nunique() >= 990, 'ranking part ~1000 images')
    # same-model agreement: for confident ref boxes, best IoU with same-class box in new file
    hi = r[r.confidence > 0.3]; g = dict(tuple(t.groupby('image_id'))); ious = []
    for row in hi.itertuples():
        c = g.get(row.image_id)
        if c is None: ious.append(0); continue
        c = c[c.class_id == row.class_id]
        if not len(c): ious.append(0); continue
        ix = np.clip(np.minimum(c.x2, row.x2) - np.maximum(c.x1, row.x1), 0, None); iy = np.clip(np.minimum(c.y2, row.y2) - np.maximum(c.y1, row.y1), 0, None)
        inter = ix * iy; u = (c.x2 - c.x1) * (c.y2 - c.y1) + (row.x2 - row.x1) * (row.y2 - row.y1) - inter
        ious.append(float((inter / u).max()))
    ious = np.array(ious); print(f'  agreement with ref (conf>0.3 boxes): median IoU {np.median(ious):.3f}, share IoU>0.9: {(ious > 0.9).mean():.3f}')
print('RESULT:', 'PASS' if ok else 'FAIL')
