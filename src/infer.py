"""Local single-checkpoint inference (MPS/CUDA/CPU) with the exact training preprocessing.

usage:
  python src/infer.py --weights W.pt --bands 5 8 13 --norm global --split val [--gain ranking] [--tta] --out preds.csv
  --split  val | test | ranking         images from data/ (val = splits/val_all.txt hold-out)
  --gain   none | test | ranking        SHIFT-STRESS: multiply each raw band by the measured train->target illumination
                                        gain (splits/illumination_gains.json) before preprocessing (val only; for robustness tests)
  --tta                                 same-checkpoint hflip TTA fused with WBF (the model's own two passes)
Output: submission-style CSV (id,image_id,class_id,confidence,x1,y1,x2,y2) in post-X2Cube pixels.
"""
import argparse, os, json, glob, time, hashlib
import numpy as np, pandas as pd, cv2, torch
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = {'val': 'data/data_train/data_train/VIS', 'test': 'data/data_test/data_test/VIS', 'ranking': 'data/data_ranking/data_ranking/VIS'}
COLS = ['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2']


def x2cube(img, cell=4):
    M, N = img.shape
    img = img[:M // cell * cell, :N // cell * cell]; M, N = img.shape
    return img.reshape(M // cell, cell, N // cell, cell).transpose(0, 2, 1, 3).reshape(M // cell, N // cell, cell * cell)


def to_uint8(cube, norm='global'):
    ax = (0, 1) if norm == 'band' else None
    lo, hi = np.percentile(cube, 0.5, axis=ax), np.percentile(cube, 99.5, axis=ax)
    return np.clip((cube.astype(np.float32) - lo) / np.maximum(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def prep(args):
    src, dst, bands, norm, gain = args
    raw = np.array(Image.open(src)); raw = raw[..., 0] if raw.ndim == 3 else raw
    cube = x2cube(raw).astype(np.float32)
    if gain is not None: cube = cube * np.asarray(gain, np.float32)  # simulated illumination change (stress test only)
    cube = to_uint8(cube, norm)
    if len(bands) == 3: cv2.imwrite(dst + '.png', np.ascontiguousarray(cube[:, :, bands][:, :, ::-1]))
    else: cv2.imwritemulti(dst + '.tiff', [cube[:, :, i] for i in bands])


def stems_for(split):
    if split == 'val': return [l.strip() for l in open(os.path.join(ROOT, 'splits/val_all.txt')) if l.strip()]
    return sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, DIRS[split])) if f.endswith('.png'))


def prepared(split, bands, norm, gain_name):
    gain = None if gain_name == 'none' else json.load(open(os.path.join(ROOT, 'splits/illumination_gains.json')))[f'{gain_name}_gain']
    key = hashlib.md5(json.dumps([split, bands, norm, gain_name]).encode()).hexdigest()[:10]
    out = os.path.join(ROOT, 'cache', f'{split}_{key}'); stems = stems_for(split)
    ext = '.png' if len(bands) == 3 else '.tiff'
    todo = [s for s in stems if not os.path.exists(f'{out}/{s}{ext}')]
    if todo:
        os.makedirs(out, exist_ok=True)
        with ProcessPoolExecutor(os.cpu_count()) as ex:
            list(ex.map(prep, [(os.path.join(ROOT, DIRS[split], f'{s}.png'), f'{out}/{s}', bands, norm, gain) for s in todo], chunksize=8))
    return [f'{out}/{s}{ext}' for s in stems]


def predict(model, files, imgsz, device, flip=False, conf=0.001, iou=0.6, max_det=300):
    from ultralytics.utils.patches import imread
    rows, sizes = [], {}
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        src = [np.ascontiguousarray(imread(p, cv2.IMREAD_UNCHANGED)[:, ::-1]) for p in batch] if flip else batch
        for p, r in zip(batch, model.predict(src, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False)):
            iid = int(os.path.splitext(os.path.basename(p))[0]); H, W = r.orig_shape; sizes[iid] = (H, W)
            xyxy = r.boxes.xyxy.cpu().numpy().copy()
            if flip: xyxy[:, [0, 2]] = W - xyxy[:, [2, 0]]
            rows += [(iid, int(c), float(s), *map(float, b)) for b, c, s in zip(xyxy, r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    return pd.DataFrame(rows, columns=COLS), sizes


def fuse(dfs, sizes, iou_thr=0.6, skip=0.001):
    from ensemble_boxes import weighted_boxes_fusion
    groups = [dict(tuple(d.groupby('image_id'))) for d in dfs]; rows = []
    for iid, (H, W) in sizes.items():
        bl, sl, ll = [], [], []
        for g in groups:
            d = g.get(iid)
            if d is None or not len(d): bl.append([]); sl.append([]); ll.append([]); continue
            bl.append((d[['x1', 'y1', 'x2', 'y2']].values / [W, H, W, H]).clip(0, 1).tolist()); sl.append(d.confidence.tolist()); ll.append(d.class_id.tolist())
        if not any(len(x) for x in sl): continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thr, skip_box_thr=skip, conf_type='avg')
        rows += [(iid, int(c), float(sc), x1 * W, y1 * H, x2 * W, y2 * H) for (x1, y1, x2, y2), sc, c in zip(b, s, l)]
    return pd.DataFrame(rows, columns=COLS)


def write_csv(df, path):
    df = df.sort_values(['image_id', 'confidence'], ascending=[True, False]).reset_index(drop=True)
    df.insert(0, 'id', range(len(df)))
    df['confidence'] = df.confidence.round(5); df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].round(2)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); df.to_csv(path, index=False); return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True); ap.add_argument('--bands', nargs='+', type=int, default=[5, 8, 13])
    ap.add_argument('--norm', default='global', choices=['global', 'band']); ap.add_argument('--split', nargs='+', default=['val'])
    ap.add_argument('--gain', default='none', choices=['none', 'test', 'ranking']); ap.add_argument('--tta', action='store_true')
    ap.add_argument('--imgsz', type=int, default=1024); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO(a.weights); t0 = time.time(); parts = []
    for sp in a.split:
        files = prepared(sp, a.bands, a.norm, a.gain if sp == 'val' else 'none')
        d, sizes = predict(model, files, a.imgsz, device)
        if a.tta: d = fuse([d, predict(model, files, a.imgsz, device, flip=True)[0]], sizes)
        parts.append(d); print(f'{sp}: {len(files)} images, {len(d)} boxes', flush=True)
    df = write_csv(pd.concat(parts), a.out)
    print(f'wrote {a.out}: {len(df)} rows, {df.image_id.nunique()} images, {time.time() - t0:.0f}s on {device}', flush=True)


if __name__ == '__main__':
    main()
