# Phase 2 inference kernel. For each trained checkpoint (one model at a time, never combined):
#   1. pycocotools val score on the same seed-0 hold-out split the models were trained with, plain and with
#      same-checkpoint hflip TTA (WBF of the model's own two passes; allowed by the rules),
#   2. ONE submission CSV covering the test set AND the ranking set (2000 images) -> submission_<tag>[_tta].csv
# Outputs (in /kaggle/working): submission_*.csv, val_pred_*.csv, phase2_summary.json
# Local use: python predict_phase2.py --input data --work out_dir --model ep40=runs/rgb_s1024/weights/best_ep40.pt=5,8,13
import os, sys, glob, json, time, random, subprocess, argparse
if os.path.isdir('/kaggle/working'):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.155', 'pycocotools', 'ensemble-boxes'], check=False)

# ---- CONFIG ----
MODELS = [  # tag, weights glob under /kaggle/input, bands the model was trained on
    {'tag': 'ep40', 'weights': '**/rgb_s1024_ep40.pt', 'bands': [5, 8, 13]},
    {'tag': 'y26s', 'weights': '**/hod-y26s-1024/best.pt', 'bands': [5, 8, 13]},
]
IMGSZ = 1024
CONF, IOU, MAX_DET = 0.001, 0.6, 300
TTA = True                 # also score hflip TTA on val; test+ranking TTA CSV is written only if it beats plain by TTA_MIN_GAIN
TTA_MIN_GAIN = 0.003
EXPECTED_PER_SET = 1000    # sanity check only
# ----------------

import numpy as np, cv2, pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import torch
from ultralytics import YOLO
from ultralytics.utils.patches import imread
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ap = argparse.ArgumentParser()
ap.add_argument('--input', default='/kaggle/input', help='root searched for the competition data and weight globs')
ap.add_argument('--work', default='/kaggle/working')
ap.add_argument('--model', nargs='*', default=[], help='tag=weights_path_or_glob=b1,b2,b3[=global|band] (overrides MODELS)')
ap.add_argument('--no_tta', action='store_true')
A = ap.parse_args()
if A.model: MODELS = [{'tag': p[0], 'weights': p[1], 'bands': [int(b) for b in p[2].split(',')], 'norm': p[3] if len(p) > 3 else 'global'} for p in (m.split('=') for m in A.model)]
if A.no_tta: TTA = False
INPUT = A.input.rstrip('/'); WORK = A.work; OUT = f'{WORK}/imgs'; os.makedirs(WORK, exist_ok=True)
COLS = ['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2']
DEVICE = 0 if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'  # CPU/MPS work too


def find_input():
    for _ in range(60):  # dataset mount can lag
        for root, dirs, files in os.walk(INPUT):
            if 'class.txt' in files and os.path.isdir(os.path.join(root, 'data_train')): return root
            if root.count('/') - INPUT.count('/') > 5: dirs[:] = []
        time.sleep(10)
    raise FileNotFoundError(f'competition data not found under {INPUT}: ' + str(os.listdir(INPUT)))


def find_ranking():
    found = {}
    for root, dirs, files in os.walk(INPUT):
        n = sum(f.lower().endswith('.png') for f in files)
        if n and 'rank' in root.lower(): found[root] = n
    return max(found, key=found.get) if found else None


def x2cube(img, cell=4):
    M, N = img.shape
    img = img[:M // cell * cell, :N // cell * cell]; M, N = img.shape  # frame sizes vary by a few px
    return img.reshape(M // cell, cell, N // cell, cell).transpose(0, 2, 1, 3).reshape(M // cell, N // cell, cell * cell)


def to_uint8(cube, norm='global'):
    """'global': one percentile range for all bands (keeps the illumination spectrum); 'band': per-band range per image
    (white-balance-like; cancels the train/test illumination shift found in the EDA)."""
    ax = (0, 1) if norm == 'band' else None
    lo, hi = np.percentile(cube, 0.5, axis=ax), np.percentile(cube, 99.5, axis=ax)
    return np.clip((cube.astype(np.float32) - lo) / np.maximum(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def process(args):
    src, dst_stem, bands, norm = args
    raw = np.array(Image.open(src)); raw = raw[..., 0] if raw.ndim == 3 else raw
    cube = to_uint8(x2cube(raw), norm)
    if len(bands) == 3: cv2.imwrite(dst_stem + '.png', np.ascontiguousarray(cube[:, :, bands][:, :, ::-1]))
    else: cv2.imwritemulti(dst_stem + '.tiff', [cube[:, :, i] for i in bands])
    return cube.shape[:2]


def parse_xml(path):
    r = ET.parse(path).getroot(); W = int(r.find('size/width').text); H = int(r.find('size/height').text)
    return W, H, [(o.find('name').text.strip(), *[float(o.find('bndbox').find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]) for o in r.findall('object')]


def iid_of(p): return int(os.path.splitext(os.path.basename(p))[0])


def predict(model, files, flip=False):
    """-> DataFrame[COLS] in original-image pixels, {image_id: (H, W)}"""
    rows, sizes = [], {}
    for i in range(0, len(files), 32):
        batch = files[i:i + 32]
        src = [np.ascontiguousarray(imread(p)[:, ::-1]) for p in batch] if flip else batch
        for p, r in zip(batch, model.predict(src, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET, device=DEVICE, verbose=False, half=DEVICE == 0)):
            iid = iid_of(p); H, W = r.orig_shape; sizes[iid] = (H, W); b = r.boxes
            xyxy = b.xyxy.cpu().numpy().copy()
            if flip: xyxy[:, [0, 2]] = W - xyxy[:, [2, 0]]
            rows += [(iid, int(c), float(s), *map(float, bb)) for bb, c, s in zip(xyxy, b.cls.cpu().numpy(), b.conf.cpu().numpy())]
    return pd.DataFrame(rows, columns=COLS), sizes


def fuse(dfs, sizes, iou_thr=0.6):
    """Same-checkpoint TTA: WBF of one model's passes over the same image."""
    groups = [dict(tuple(d.groupby('image_id'))) for d in dfs]; rows = []
    for iid, (H, W) in sizes.items():
        bl, sl, ll = [], [], []
        for g in groups:
            d = g.get(iid)
            if d is None or not len(d): bl.append(np.zeros((0, 4)).tolist()); sl.append([]); ll.append([]); continue
            bl.append((d[['x1', 'y1', 'x2', 'y2']].values / [W, H, W, H]).clip(0, 1).tolist()); sl.append(d.confidence.tolist()); ll.append(d.class_id.tolist())
        if not any(len(x) for x in sl): continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thr, skip_box_thr=CONF, conf_type='avg')
        rows += [(iid, int(c), float(sc), x1 * W, y1 * H, x2 * W, y2 * H) for (x1, y1, x2, y2), sc, c in zip(b, s, l)]
    return pd.DataFrame(rows, columns=COLS)


def coco_score(df, gt, classes):
    if not len(df): return {'mAP50-95': 0.0}
    dt = gt.loadRes([{'image_id': int(r.image_id), 'category_id': int(r.class_id), 'score': float(r.confidence), 'bbox': [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1]} for r in df.itertuples()])
    E = COCOeval(gt, dt, 'bbox'); E.evaluate(); E.accumulate(); E.summarize(); prec = E.eval['precision']
    return {'mAP50-95': round(float(E.stats[0]), 5), 'mAP50': round(float(E.stats[1]), 5),
            'per_class': {n: round(float(prec[:, :, k, 0, -1][prec[:, :, k, 0, -1] > -1].mean()), 4) for k, n in enumerate(classes)}}


def write_sub(df, path):
    df = df.sort_values(['image_id', 'confidence'], ascending=[True, False]).reset_index(drop=True)
    df.insert(0, 'id', range(len(df)))
    df['confidence'] = df.confidence.round(5); df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].round(2)
    df.to_csv(path, index=False); return df


def main():
    t0 = time.time()
    IN = find_input(); print('competition data:', IN, '| device:', DEVICE, '| threads:', torch.get_num_threads(), flush=True)
    classes = [l.strip() for l in open(f'{IN}/class.txt') if l.strip()]; cls2id = {c: i for i, c in enumerate(classes)}
    TRAIN_IMG, TRAIN_ANN, TEST_IMG = f'{IN}/data_train/data_train/VIS', f'{IN}/data_train/data_train/Annotations/VIS', f'{IN}/data_test/data_test/VIS'
    RANK_IMG = find_ranking()
    if not RANK_IMG: raise FileNotFoundError(f'no ranking PNGs under {INPUT}; attach the competition data source')
    if os.path.exists(f'{IN}/val_stems.txt'):  # written by fetch_data.py when only the hold-out images are downloaded
        val = sorted(l.strip() for l in open(f'{IN}/val_stems.txt') if l.strip())
    else:
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(TRAIN_IMG) if f.endswith('.png'))
        random.Random(0).shuffle(stems); val = sorted(stems[:int(len(stems) * 0.1)])  # identical split to training
    sets = {'val': (TRAIN_IMG, val),
            'test': (TEST_IMG, sorted(os.path.splitext(f)[0] for f in os.listdir(TEST_IMG) if f.lower().endswith('.png'))),
            'ranking': (RANK_IMG, sorted(os.path.splitext(f)[0] for f in os.listdir(RANK_IMG) if f.lower().endswith('.png')))}
    for k, (d, s) in sets.items():
        print(f'{k}: {len(s)} images in {d} (first {s[:3]}, last {s[-3:]})', flush=True)
        if k != 'val' and len(s) != EXPECTED_PER_SET: print(f'WARNING: {k} has {len(s)} images, expected {EXPECTED_PER_SET}', flush=True)
        if not all(x.isdigit() for x in s): raise ValueError(f'{k}: non-integer file stems')
    if set(sets['test'][1]) & set(sets['ranking'][1]): raise ValueError('test and ranking image ids overlap')

    gt_images, gt_anns = [], []
    for s in val:
        W, H, boxes = parse_xml(f'{TRAIN_ANN}/{s}.xml'); gt_images.append({'id': int(s), 'width': W, 'height': H})
        for name, x1, y1, x2, y2 in boxes:
            x1, x2 = sorted((max(0, x1), min(W, x2))); y1, y2 = sorted((max(0, y1), min(H, y2)))
            if x2 - x1 < 1 or y2 - y1 < 1: continue
            gt_anns.append({'id': len(gt_anns) + 1, 'image_id': int(s), 'category_id': cls2id[name], 'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0})
    gt = COCO(); gt.dataset = {'images': gt_images, 'annotations': gt_anns, 'categories': [{'id': i, 'name': n} for i, n in enumerate(classes)]}; gt.createIndex()

    prepared = {}
    def files_for(bands, norm):
        key = '-'.join(map(str, bands)) + '_' + norm
        if key not in prepared:
            jobs = []
            for k, (d, s) in sets.items():
                os.makedirs(f'{OUT}/{key}/{k}', exist_ok=True); jobs += [(f'{d}/{x}.png', f'{OUT}/{key}/{k}/{x}', bands, norm) for x in s]
            with ProcessPoolExecutor(os.cpu_count()) as ex: list(ex.map(process, jobs, chunksize=16))
            prepared[key] = {k: sorted(glob.glob(f'{OUT}/{key}/{k}/*'), key=iid_of) for k in sets}
            print(f'prepared bands {key}: ' + str({k: len(v) for k, v in prepared[key].items()}), f'{time.time() - t0:.0f}s', flush=True)
        return prepared[key]

    summary = {'imgsz': IMGSZ, 'conf': CONF, 'iou': IOU, 'max_det': MAX_DET, 'models': {}}
    for m in MODELS:
        cands = [m['weights']] if os.path.isfile(m['weights']) else sorted(glob.glob(f"{INPUT}/{m['weights']}", recursive=True))
        if not cands: print('SKIP', m['tag'], 'weights not found:', m['weights'], flush=True); continue
        model = YOLO(cands[0]); files = files_for(m['bands'], m.get('norm', 'global')); res = {'weights': cands[0], 'bands': m['bands'], 'norm': m.get('norm', 'global')}
        print(f"==== {m['tag']}: {cands[0]}", flush=True)
        plain = {k: predict(model, files[k]) for k in sets}
        res['val_plain'] = coco_score(plain['val'][0], gt, classes); plain['val'][0].to_csv(f"{WORK}/val_pred_{m['tag']}.csv", index=False)
        print(m['tag'], 'VAL plain', res['val_plain']['mAP50-95'], flush=True)
        sub = write_sub(pd.concat([plain['test'][0], plain['ranking'][0]]), f"{WORK}/submission_{m['tag']}.csv")
        res['csv_plain'] = {'rows': len(sub), 'images': int(sub.image_id.nunique()),
                            **{f'{k}_images': int(sub.image_id.isin([int(x) for x in sets[k][1]]).groupby(sub.image_id).any().sum()) for k in ('test', 'ranking')}}
        if TTA:
            fl = predict(model, files['val'], flip=True)[0]
            res['val_tta'] = coco_score(fuse([plain['val'][0], fl], plain['val'][1]), gt, classes)
            print(m['tag'], 'VAL hflip-TTA', res['val_tta']['mAP50-95'], flush=True)
            if res['val_tta']['mAP50-95'] >= res['val_plain']['mAP50-95'] + TTA_MIN_GAIN:
                parts = []
                for k in ('test', 'ranking'):
                    parts.append(fuse([plain[k][0], predict(model, files[k], flip=True)[0]], plain[k][1]))
                sub = write_sub(pd.concat(parts), f"{WORK}/submission_{m['tag']}_tta.csv")
                res['csv_tta'] = {'rows': len(sub), 'images': int(sub.image_id.nunique())}
        summary['models'][m['tag']] = res
        json.dump(summary, open(f'{WORK}/phase2_summary.json', 'w'), indent=1)
        print(m['tag'], json.dumps({k: v for k, v in res.items() if not k.startswith('val_')}), flush=True)
    summary['runtime_s'] = round(time.time() - t0); json.dump(summary, open(f'{WORK}/phase2_summary.json', 'w'), indent=1)
    os.system(f'rm -rf {OUT}')
    print('DONE', json.dumps({t: {'val_plain': r['val_plain']['mAP50-95'], 'val_tta': r.get('val_tta', {}).get('mAP50-95')} for t, r in summary['models'].items()}), flush=True)


if __name__ == '__main__':
    main()
