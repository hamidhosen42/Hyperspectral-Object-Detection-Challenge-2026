"""Parse VOC XML annotations, convert 16-band mosaic PNGs into YOLO-ready images.

Outputs (under data/yolo/):
  rgb/images/{train,val,test}/*.png   pseudo-RGB (3 selected bands, uint8)
  hsi/images/{train,val,test}/*.tiff  all 16 bands as multi-page TIFF (uint8)
  {rgb,hsi}/labels/{train,val}/*.txt  YOLO format labels (identical copies)
  data_rgb.yaml, data_hsi.yaml
"""
import os, sys, glob, random, json
import numpy as np, cv2
import xml.etree.ElementTree as ET
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
TRAIN_IMG = os.path.join(DATA, 'data_train/data_train/VIS')
TRAIN_ANN = os.path.join(DATA, 'data_train/data_train/Annotations/VIS')
TEST_IMG = os.path.join(DATA, 'data_test/data_test/VIS')
OUT = os.path.join(DATA, 'yolo')
CLASSES = [l.strip() for l in open(os.path.join(DATA, 'class.txt')) if l.strip()]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}
CELL = 4
RGB_BANDS = [5, 8, 13]   # from organiser demo; roughly R/G/B-ish bands
VAL_FRAC = 0.1
SEED = 0


def x2cube(img, cell=CELL):
    """(H*cell, W*cell) mosaic -> (H, W, cell*cell) cube."""
    M, N = img.shape
    return img.reshape(M // cell, cell, N // cell, cell).transpose(0, 2, 1, 3).reshape(M // cell, N // cell, cell * cell)


def to_uint8(cube):
    """Per-image global percentile normalisation to uint8 (same scale over bands keeps spectral shape)."""
    lo, hi = np.percentile(cube, 0.5), np.percentile(cube, 99.5)
    return np.clip((cube.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def parse_xml(path):
    r = ET.parse(path).getroot()
    W = int(r.find('size/width').text); H = int(r.find('size/height').text)
    boxes = []
    for o in r.findall('object'):
        name = o.find('name').text.strip()
        b = o.find('bndbox')
        x1, y1, x2, y2 = [float(b.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        boxes.append((name, x1, y1, x2, y2))
    return W, H, boxes


def process(args):
    src, stem, split = args
    img = np.array(Image.open(src))
    cube = to_uint8(x2cube(img))
    rgb = cube[:, :, RGB_BANDS][:, :, ::-1]  # store as BGR for cv2
    cv2.imwrite(os.path.join(OUT, 'rgb/images', split, stem + '.png'), np.ascontiguousarray(rgb))
    pages = [cube[:, :, i] for i in range(cube.shape[2])]
    cv2.imwritemulti(os.path.join(OUT, 'hsi/images', split, stem + '.tiff'), pages)
    return stem, cube.shape[:2]


def main():
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(TRAIN_IMG) if f.endswith('.png'))
    random.Random(SEED).shuffle(stems)
    n_val = int(len(stems) * VAL_FRAC)
    split_of = {s: ('val' if i < n_val else 'train') for i, s in enumerate(stems)}
    test_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(TEST_IMG) if f.endswith('.png'))

    for kind in ('rgb', 'hsi'):
        for sp in ('train', 'val', 'test'):
            os.makedirs(os.path.join(OUT, kind, 'images', sp), exist_ok=True)
        for sp in ('train', 'val'):
            os.makedirs(os.path.join(OUT, kind, 'labels', sp), exist_ok=True)

    # labels
    stats = {'n_boxes': 0, 'per_class': {c: 0 for c in CLASSES}, 'sizes': set(), 'unknown': set()}
    for s in stems:
        W, H, boxes = parse_xml(os.path.join(TRAIN_ANN, s + '.xml'))
        stats['sizes'].add((W, H))
        lines = []
        for name, x1, y1, x2, y2 in boxes:
            if name not in CLS2ID:
                stats['unknown'].add(name); continue
            x1, x2 = sorted((max(0, x1), min(W, x2))); y1, y2 = sorted((max(0, y1), min(H, y2)))
            if x2 - x1 < 1 or y2 - y1 < 1: continue
            cx, cy, bw, bh = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"{CLS2ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats['per_class'][name] += 1; stats['n_boxes'] += 1
        for kind in ('rgb', 'hsi'):
            with open(os.path.join(OUT, kind, 'labels', split_of[s], s + '.txt'), 'w') as f:
                f.write('\n'.join(lines))
    stats['sizes'] = sorted(stats['sizes']); stats['unknown'] = sorted(stats['unknown'])
    stats['n_train'] = len(stems) - n_val; stats['n_val'] = n_val; stats['n_test'] = len(test_stems)
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(os.path.join(OUT, 'stats.json'), 'w'), indent=1)

    # images
    jobs = [(os.path.join(TRAIN_IMG, s + '.png'), s, split_of[s]) for s in stems]
    jobs += [(os.path.join(TEST_IMG, s + '.png'), s, 'test') for s in test_stems]
    with ProcessPoolExecutor(8) as ex:
        for i, (stem, shp) in enumerate(ex.map(process, jobs, chunksize=16)):
            if i % 500 == 0: print(i, stem, shp, flush=True)

    for kind, ch in (('rgb', 3), ('hsi', 16)):
        with open(os.path.join(OUT, f'data_{kind}.yaml'), 'w') as f:
            f.write(f"path: {OUT}/{kind}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    f"channels: {ch}\nnc: {len(CLASSES)}\nnames: {CLASSES}\n")
    print('done')


if __name__ == '__main__':
    main()
