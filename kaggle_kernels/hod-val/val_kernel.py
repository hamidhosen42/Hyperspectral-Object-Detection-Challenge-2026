# Validation kernel: each single checkpoint is run on the 300-image seed-0 hold-out under several conditions.
# Exports raw predictions (+ the hold-out ground truth) so every experiment is scored locally by src/validate.py
# with the same metric code:
#   plain         standard inference
#   tta           same-checkpoint hflip TTA (WBF of the model's own two passes)
#   gain_test     SHIFT-STRESS: raw bands multiplied by the measured train->test illumination gain
#   gain_ranking  SHIFT-STRESS: raw bands multiplied by the measured train->ranking illumination gain
# The gains come from unlabeled image statistics only (splits/illumination_gains.json); no model is fitted on them.
# Outputs (in /kaggle/working): val_gt.json, preds/<tag>_<condition>.csv, runs.json
import os, sys, glob, json, time, random, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.155', 'ensemble-boxes'], check=False)

# ---- CONFIG ----
MODELS = [  # tag, weights glob under /kaggle/input, bands, normalisation
    {'tag': 'E4c', 'weights': '**/e4c_y26s_16b_best.pt', 'bands': list(range(16)), 'norm': 'global'},
]
CONDITIONS = ['plain', 'tta', 'gain_ranking']
PREDICT_TEST = True        # also write preds/<tag>_test.csv (plain) for the Phase 2 test half
GAINS = {'test': [1.0223, 1.0401, 1.0451, 1.0365, 1.0036, 1.0206, 1.0097, 0.9966, 0.9939, 0.996, 0.9979, 0.9963, 0.9875, 0.9919, 1.0008, 0.9964],
         'ranking': [1.0711, 1.126, 1.1204, 1.111, 1.0148, 1.0703, 1.032, 1.0098, 0.9999, 0.9892, 0.9524, 0.9461, 0.9474, 0.9629, 0.9666, 0.9684]}
IMGSZ = 1024
# ----------------

import numpy as np, cv2, pandas as pd, torch
import xml.etree.ElementTree as ET
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO
from ultralytics.utils.patches import imread
from ensemble_boxes import weighted_boxes_fusion

WORK = '/kaggle/working'; OUT = f'{WORK}/imgs'; os.makedirs(f'{WORK}/preds', exist_ok=True)
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
COLS = ['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2']


def find_input():
    for _ in range(60):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'class.txt' in files and os.path.isdir(os.path.join(root, 'data_train')): return root
            if root.count('/') > 7: dirs[:] = []
        time.sleep(10)
    raise FileNotFoundError('data not found: ' + str(os.listdir('/kaggle/input')))


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
    if gain is not None: cube = cube * np.asarray(gain, np.float32)
    cube = to_uint8(cube, norm)
    if len(bands) == 3: cv2.imwrite(dst + '.png', np.ascontiguousarray(cube[:, :, bands][:, :, ::-1]))
    else: cv2.imwritemulti(dst + '.tiff', [cube[:, :, i] for i in bands])


def predict(model, files, flip=False):
    rows, sizes = [], {}
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        if flip or not batch[0].endswith('.png'):  # predictor's file loader reads multi-page TIFFs as 3-channel: pass arrays
            src = [imread(p, cv2.IMREAD_UNCHANGED) for p in batch]
            src = [np.ascontiguousarray(x[:, ::-1]) if flip else x for x in src]
        else: src = batch
        for p, r in zip(batch, model.predict(src, imgsz=IMGSZ, conf=0.001, iou=0.6, max_det=300, device=DEVICE, verbose=False, half=DEVICE == 0)):
            iid = int(os.path.splitext(os.path.basename(p))[0]); H, W = r.orig_shape; sizes[iid] = (H, W)
            xyxy = r.boxes.xyxy.cpu().numpy().copy()
            if flip: xyxy[:, [0, 2]] = W - xyxy[:, [2, 0]]
            rows += [(iid, int(c), float(s), *map(float, b)) for b, c, s in zip(xyxy, r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    return pd.DataFrame(rows, columns=COLS), sizes


def fuse(dfs, sizes):
    groups = [dict(tuple(d.groupby('image_id'))) for d in dfs]; rows = []
    for iid, (H, W) in sizes.items():
        bl, sl, ll = [], [], []
        for g in groups:
            d = g.get(iid)
            if d is None or not len(d): bl.append([]); sl.append([]); ll.append([]); continue
            bl.append((d[['x1', 'y1', 'x2', 'y2']].values / [W, H, W, H]).clip(0, 1).tolist()); sl.append(d.confidence.tolist()); ll.append(d.class_id.tolist())
        if not any(len(x) for x in sl): continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, iou_thr=0.6, skip_box_thr=0.001, conf_type='avg')
        rows += [(iid, int(c), float(sc), x1 * W, y1 * H, x2 * W, y2 * H) for (x1, y1, x2, y2), sc, c in zip(b, s, l)]
    return pd.DataFrame(rows, columns=COLS)


def main():
    t0 = time.time(); IN = find_input(); print('data:', IN, '| device:', DEVICE, flush=True)
    TRAIN_IMG, TRAIN_ANN = f'{IN}/data_train/data_train/VIS', f'{IN}/data_train/data_train/Annotations/VIS'
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(TRAIN_IMG) if f.endswith('.png'))
    random.Random(0).shuffle(stems); val = sorted(stems[:int(len(stems) * 0.1)], key=int)  # same hold-out as training
    # ground truth export (raw XML boxes; src/validate.py applies the same clipping as training)
    gt = {}
    for s in val:
        r = ET.parse(f'{TRAIN_ANN}/{s}.xml').getroot()
        gt[s] = {'W': int(r.find('size/width').text), 'H': int(r.find('size/height').text),
                 'boxes': [[o.find('name').text.strip()] + [float(o.find('bndbox').find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')] for o in r.findall('object')]}
    json.dump(gt, open(f'{WORK}/val_gt.json', 'w'))
    runs = {}
    for m in MODELS:
        cands = sorted(glob.glob(f"/kaggle/input/{m['weights']}", recursive=True))
        if not cands: print('SKIP', m['tag'], 'no weights for', m['weights'], flush=True); continue
        model = YOLO(cands[0]); runs[m['tag']] = {'weights': cands[0], **{k: m[k] for k in ('bands', 'norm')}}
        for cond in CONDITIONS:
            gain = GAINS[cond.split('_')[1]] if cond.startswith('gain_') else None
            d = f"{OUT}/{'-'.join(map(str, m['bands']))}_{m['norm']}_{cond if gain else 'plain'}"
            if not os.path.isdir(d):
                os.makedirs(d)
                with ProcessPoolExecutor(os.cpu_count()) as ex:
                    list(ex.map(prep, [(f'{TRAIN_IMG}/{s}.png', f'{d}/{s}', m['bands'], m['norm'], gain) for s in val], chunksize=8))
            files = sorted(glob.glob(f'{d}/*'), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            df, sizes = predict(model, files)
            if cond == 'tta': df = fuse([df, predict(model, files, flip=True)[0]], sizes)
            df.to_csv(f"{WORK}/preds/{m['tag']}_{cond}.csv", index=False)
            print(f"{m['tag']} {cond}: {len(df)} boxes, {time.time() - t0:.0f}s", flush=True)
        if PREDICT_TEST:
            TEST_IMG = f'{IN}/data_test/data_test/VIS'; d = f"{OUT}/test_{'-'.join(map(str, m['bands']))}_{m['norm']}"
            if not os.path.isdir(d):
                os.makedirs(d)
                with ProcessPoolExecutor(os.cpu_count()) as ex:
                    list(ex.map(prep, [(f'{TEST_IMG}/{f}', f"{d}/{os.path.splitext(f)[0]}", m['bands'], m['norm'], None) for f in os.listdir(TEST_IMG) if f.endswith('.png')], chunksize=8))
            files = sorted(glob.glob(f'{d}/*'), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            df, _ = predict(model, files); df.to_csv(f"{WORK}/preds/{m['tag']}_test.csv", index=False)
            print(f"{m['tag']} test: {len(files)} images, {len(df)} boxes, {time.time() - t0:.0f}s", flush=True)
    json.dump(runs, open(f'{WORK}/runs.json', 'w'), indent=1)
    os.system(f'rm -rf {OUT}'); print('DONE', flush=True)


if __name__ == '__main__':
    main()
