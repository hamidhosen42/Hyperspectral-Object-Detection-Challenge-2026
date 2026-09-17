# Kaggle GPU training kernel for the Hyperspectral Object Detection Challenge 2026.
# One experiment per kernel; the CONFIG block is rewritten by push_experiment.py.
# Outputs (in /kaggle/working): best.pt, results.csv, val_pred.csv, val_score.json, submission.csv
import os, sys, json, glob, random, subprocess, time
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'ultralytics', 'pycocotools'], check=False)

# ---- CONFIG ----
MODEL = 'yolo11s.pt'
IMGSZ = 1024
EPOCHS = 50
BATCH = 16
BANDS = [5, 8, 13]
EXTRA = {}
FULL_DATA = False
# ----------------

import numpy as np, cv2, pandas as pd
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO, RTDETR

IN = '/kaggle/input/hyperspectral-object-detection-challenge-2026'
WORK = '/kaggle/working'; DATA = '/kaggle/working/yolo'
TRAIN_IMG = f'{IN}/data_train/data_train/VIS'; TRAIN_ANN = f'{IN}/data_train/data_train/Annotations/VIS'; TEST_IMG = f'{IN}/data_test/data_test/VIS'
CLASSES = [l.strip() for l in open(f'{IN}/class.txt') if l.strip()]; CLS2ID = {c: i for i, c in enumerate(CLASSES)}
NCH = len(BANDS)

def x2cube(img, cell=4):
    M, N = img.shape
    return img.reshape(M // cell, cell, N // cell, cell).transpose(0, 2, 1, 3).reshape(M // cell, N // cell, cell * cell)

def to_uint8(cube):
    lo, hi = np.percentile(cube, 0.5), np.percentile(cube, 99.5)
    return np.clip((cube.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

def process(args):
    src, stem, split = args
    from PIL import Image
    cube = to_uint8(x2cube(np.array(Image.open(src))))
    if NCH == 3:
        cv2.imwrite(f'{DATA}/images/{split}/{stem}.png', np.ascontiguousarray(cube[:, :, BANDS][:, :, ::-1]))
    else:
        cv2.imwritemulti(f'{DATA}/images/{split}/{stem}.tiff', [cube[:, :, i] for i in BANDS])
    return cube.shape[:2]

def parse_xml(path):
    r = ET.parse(path).getroot(); W = int(r.find('size/width').text); H = int(r.find('size/height').text)
    out = []
    for o in r.findall('object'):
        b = o.find('bndbox'); out.append((o.find('name').text.strip(), *[float(b.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]))
    return W, H, out

# ---- data prep (identical split to the local pipeline: seed-0 shuffle, first 10% = val) ----
t0 = time.time()
stems = sorted(os.path.splitext(f)[0] for f in os.listdir(TRAIN_IMG) if f.endswith('.png'))
random.Random(0).shuffle(stems); n_val = int(len(stems) * 0.1)
val_stems = set(stems[:n_val])
split_of = {s: ('val' if s in val_stems else 'train') for s in stems}
if FULL_DATA: split_of = {s: 'train' for s in stems}
test_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(TEST_IMG) if f.endswith('.png'))
for sp in ('train', 'val', 'test'): os.makedirs(f'{DATA}/images/{sp}', exist_ok=True)
for sp in ('train', 'val'): os.makedirs(f'{DATA}/labels/{sp}', exist_ok=True)
gt_images, gt_anns = [], []
for s in stems:
    W, H, boxes = parse_xml(f'{TRAIN_ANN}/{s}.xml'); lines = []
    for name, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, x1), min(W, x2))); y1, y2 = sorted((max(0, y1), min(H, y2)))
        if x2 - x1 < 1 or y2 - y1 < 1: continue
        lines.append(f"{CLS2ID[name]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        if s in val_stems:
            gt_anns.append({'id': len(gt_anns) + 1, 'image_id': int(s), 'category_id': CLS2ID[name], 'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0})
    if s in val_stems: gt_images.append({'id': int(s), 'width': W, 'height': H})
    open(f'{DATA}/labels/{split_of[s]}/{s}.txt', 'w').write('\n'.join(lines))
jobs = [(f'{TRAIN_IMG}/{s}.png', s, split_of[s]) for s in stems] + [(f'{TEST_IMG}/{s}.png', s, 'test') for s in test_stems]
if not FULL_DATA: pass
else: jobs += [(f'{TRAIN_IMG}/{s}.png', s, 'val') for s in val_stems]  # keep a val dir so trainer can run
with ProcessPoolExecutor(os.cpu_count()) as ex: list(ex.map(process, jobs, chunksize=32))
yaml = f"path: {DATA}\ntrain: images/train\nval: images/val\ntest: images/test\nchannels: {NCH}\nnc: {len(CLASSES)}\nnames: {CLASSES}\n"
open(f'{DATA}/data.yaml', 'w').write(yaml)
print(f'prep done in {time.time()-t0:.0f}s: train {sum(v=="train" for v in split_of.values())} val {len(val_stems)} test {len(test_stems)}', flush=True)

# ---- train ----
is_detr = 'rtdetr' in MODEL
model = (RTDETR if is_detr else YOLO)(MODEL)
kw = dict(data=f'{DATA}/data.yaml', imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, device=0, workers=4, project=WORK, name='run', exist_ok=True,
          hsv_h=0.015 if NCH == 3 else 0.0, hsv_s=0.7 if NCH == 3 else 0.0, hsv_v=0.4, bgr=0.0, fliplr=0.5, mosaic=1.0, close_mosaic=10, scale=0.5,
          patience=100, plots=False, cache=False, amp=True, seed=0, deterministic=False)
kw.update(EXTRA)
try:
    import torch
    if torch.cuda.device_count() > 1 and not is_detr: kw['device'] = [0, 1]
    model.train(**kw)
except Exception as e:
    print('multi-GPU training failed, falling back to single GPU:', repr(e)[:300], flush=True)
    kw['device'] = 0; model = (RTDETR if is_detr else YOLO)(MODEL); model.train(**kw)
best = f'{WORK}/run/weights/best.pt'
os.system(f'cp {best} {WORK}/best.pt; cp {WORK}/run/results.csv {WORK}/results.csv')

# ---- predict helper (single model; optional hflip TTA via WBF of the model's own outputs is done offline) ----
def predict_csv(weights, split, out, conf=0.001, iou=0.6, max_det=300):
    m = (RTDETR if is_detr else YOLO)(weights)
    files = sorted(glob.glob(f'{DATA}/images/{split}/*'), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    rows = []
    for i in range(0, len(files), 32):
        for r in m.predict(files[i:i + 32], imgsz=IMGSZ, conf=conf, iou=iou, max_det=max_det, device=0, verbose=False, half=True):
            iid = int(os.path.splitext(os.path.basename(r.path))[0]); b = r.boxes
            for (x1, y1, x2, y2), c, s in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()):
                rows.append((iid, int(c), float(s), float(x1), float(y1), float(x2), float(y2)))
    df = pd.DataFrame(rows, columns=['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2']).sort_values(['image_id', 'confidence'], ascending=[True, False]).reset_index(drop=True)
    df.insert(0, 'id', range(len(df))); df.to_csv(out, index=False); return df

# ---- val score with pycocotools (official protocol) ----
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
score = {}
if not FULL_DATA:
    df = predict_csv(best, 'val', f'{WORK}/val_pred.csv')
    gt = COCO(); gt.dataset = {'images': gt_images, 'annotations': gt_anns, 'categories': [{'id': i, 'name': n} for i, n in enumerate(CLASSES)]}; gt.createIndex()
    dt = gt.loadRes([{'image_id': int(r.image_id), 'category_id': int(r.class_id), 'score': float(r.confidence), 'bbox': [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1]} for r in df.itertuples()])
    E = COCOeval(gt, dt, 'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    prec = E.eval['precision']
    score = {'mAP50-95': float(E.stats[0]), 'mAP50': float(E.stats[1]), 'per_class': {n: float(prec[:, :, k, 0, -1][prec[:, :, k, 0, -1] > -1].mean()) for k, n in enumerate(CLASSES)}}
    print('VAL_SCORE', json.dumps(score), flush=True)
score.update({'model': MODEL, 'imgsz': IMGSZ, 'epochs': EPOCHS, 'batch': BATCH, 'bands': BANDS, 'extra': EXTRA, 'full_data': FULL_DATA})
json.dump(score, open(f'{WORK}/val_score.json', 'w'), indent=1)
predict_csv(best, 'test', f'{WORK}/submission.csv')
os.system(f'rm -rf {WORK}/run/weights/last.pt {WORK}/run/weights/epoch*.pt {DATA}')
print('DONE', flush=True)
