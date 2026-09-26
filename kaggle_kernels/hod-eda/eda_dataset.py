"""Dataset EDA for the Hyperspectral Object Detection Challenge 2026.

Covers: annotation statistics (class balance, object sizes vs COCO buckets, co-occurrence, label sanity),
train-vs-test shift (adversarial validation on radiometric features and scene thumbnails, near-duplicate search),
and spectral separability of object classes (which bands tell real / plastic / wooden objects apart).

Runs locally on the raw competition data or as a Kaggle kernel (data auto-discovered under /kaggle/input).
Prints a text report and writes <out>/eda_report.json, class_stats.csv, box_spectra.csv, image_feats.csv.

usage: python kaggle_kernels/hod-eda/eda_dataset.py [--data data] [--out eda_out] [--workers 8] [--limit 0]
"""
import argparse, os, json, time, random, itertools, warnings
import numpy as np, pandas as pd, cv2
import xml.etree.ElementTree as ET
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

CELL = 4
NB = CELL * CELL
CUR_BANDS = [5, 8, 13]  # bands used by the current pseudo-RGB pipeline
MATERIAL_GROUPS = {'apple': ['apple', 'apple_plastic'], 'banana': ['banana', 'banana_plastic'], 'orange': ['orange', 'orange_plastic'],
                   'egg': ['egg', 'egg_plastic', 'egg_wood'], 'car': ['car', 'car_toy']}
ON_KAGGLE = os.path.isdir('/kaggle/input')


def find_data(hint):
    for attempt in range(60 if ON_KAGGLE else 1):  # Kaggle dataset mount can lag
        for base in ([hint] if hint else ['data', '/kaggle/input']):
            for root, dirs, files in os.walk(base):
                if 'class.txt' in files and os.path.isdir(os.path.join(root, 'data_train')): return root
                if root.count(os.sep) - base.count(os.sep) > 5: dirs[:] = []
        if ON_KAGGLE: time.sleep(10)
    raise FileNotFoundError('competition data (class.txt + data_train/) not found')


def parse_xml(path):
    r = ET.parse(path).getroot()
    W, H = int(r.find('size/width').text), int(r.find('size/height').text)
    objs = []
    for o in r.findall('object'):
        b = o.find('bndbox')
        objs.append({'name': o.find('name').text.strip(), **{k: float(b.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')},
                     'difficult': int(o.findtext('difficult', '0') or 0), 'truncated': int(o.findtext('truncated', '0') or 0)})
    return W, H, objs


def load_cube(path):
    raw = np.array(Image.open(path))
    if raw.ndim == 3: raw = raw[..., 0]
    raw = raw[:raw.shape[0] // CELL * CELL, :raw.shape[1] // CELL * CELL]; M, N = raw.shape  # frame sizes vary by a few px
    cube = raw.reshape(M // CELL, CELL, N // CELL, CELL).transpose(0, 2, 1, 3).reshape(M // CELL, N // CELL, NB)
    return raw, cube.astype(np.float32)


def image_job(job):
    """Per-image radiometric features, a scene thumbnail and (train only) per-box mean spectra + pixel samples."""
    path, stem, split, objs, xml_wh = job
    raw, cube = load_cube(path)
    H, W = cube.shape[:2]; flat = cube.reshape(-1, NB)
    top = np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype, np.integer) else raw.max()
    p1, p50, p99 = np.percentile(raw, [1, 50, 99])
    bm, bs = flat.mean(0), flat.std(0)
    feats = {'stem': stem, 'split': split, 'H': H, 'W': W, 'dtype': str(raw.dtype), 'raw_max': float(raw.max()), 'p1': p1, 'p50': p50, 'p99': p99,
             'sat_frac': float((raw >= top).mean()), 'zero_frac': float((raw == 0).mean()),
             **{f'mean{i}': bm[i] for i in range(NB)}, **{f'std{i}': bs[i] for i in range(NB)}, **{f'shape{i}': bm[i] / max(bm.mean(), 1e-6) for i in range(NB)}}
    thumb = cv2.resize(cube.mean(2), (64, 32), interpolation=cv2.INTER_AREA).ravel().astype(np.float32)
    spectra, obj_px, bg_px = [], np.zeros((0, NB), np.float32), np.zeros((0, NB), np.float32)
    if objs:
        sx, sy = W / xml_wh[0], H / xml_wh[1]; mask = np.zeros((H, W), bool)
        for o in objs:
            x1, y1, x2, y2 = o['xmin'] * sx, o['ymin'] * sy, o['xmax'] * sx, o['ymax'] * sy
            mask[max(int(y1), 0):int(np.ceil(y2)), max(int(x1), 0):int(np.ceil(x2))] = True
            w, h = x2 - x1, y2 - y1  # central 50% of the box: mostly object, little background
            ix1 = int(np.clip(round(x1 + w / 4), 0, W - 1)); ix2 = int(np.clip(max(round(x2 - w / 4), ix1 + 1), 1, W))
            iy1 = int(np.clip(round(y1 + h / 4), 0, H - 1)); iy2 = int(np.clip(max(round(y2 - h / 4), iy1 + 1), 1, H))
            spectra.append((stem, o['name'], *cube[iy1:iy2, ix1:ix2].reshape(-1, NB).mean(0)))
        rng = np.random.default_rng(int(stem) if stem.isdigit() else 0)
        o_idx, b_idx = np.flatnonzero(mask.ravel()), np.flatnonzero(~mask.ravel())
        obj_px = flat[rng.choice(o_idx, min(64, len(o_idx)), replace=False)] if len(o_idx) else obj_px
        bg_px = flat[rng.choice(b_idx, min(64, len(b_idx)), replace=False)] if len(b_idx) else bg_px
    return feats, thumb, spectra, obj_px, bg_px


def iou_matrix(b):
    x1, y1, x2, y2 = [b[:, i][:, None] for i in range(4)]
    iw = np.clip(np.minimum(x2, x2.T) - np.maximum(x1, x1.T), 0, None); ih = np.clip(np.minimum(y2, y2.T) - np.maximum(y1, y1.T), 0, None)
    inter = iw * ih; area = (x2 - x1) * (y2 - y1)
    return inter / (area + area.T - inter + 1e-9)


def section(t): print(f"\n{'=' * 8} {t} {'=' * (70 - len(t))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None, help='dir with class.txt, data_train/, data_test/ (auto-found if omitted)')
    ap.add_argument('--out', default='/kaggle/working/eda' if ON_KAGGLE else 'eda_out')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--limit', type=int, default=0, help='debug: only use the first N train and N test images')
    a = ap.parse_args()
    warnings.filterwarnings('ignore', category=RuntimeWarning)  # spurious BLAS matmul warnings inside sklearn
    from scipy.stats import ks_2samp
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score, cross_val_predict
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import balanced_accuracy_score

    t0 = time.time(); os.makedirs(a.out, exist_ok=True); rep = {}
    root = find_data(a.data); print('data:', root, flush=True)
    classes = [l.strip() for l in open(f'{root}/class.txt') if l.strip()]
    TR_IMG, TR_ANN, TE_IMG = f'{root}/data_train/data_train/VIS', f'{root}/data_train/data_train/Annotations/VIS', f'{root}/data_test/data_test/VIS'
    tr = sorted(os.path.splitext(f)[0] for f in os.listdir(TR_IMG) if f.endswith('.png'))
    te = sorted(os.path.splitext(f)[0] for f in os.listdir(TE_IMG) if f.endswith('.png'))
    shuf = list(tr); random.Random(0).shuffle(shuf); val = set(shuf[:int(len(shuf) * 0.1)])  # same split as the training pipeline
    rank_dirs = {r: sum(f.lower().endswith('.png') for f in fs) for base in {root, '/kaggle/input'} if os.path.isdir(base) for r, _, fs in os.walk(base) if 'rank' in r.lower()}
    RK_IMG = max(rank_dirs, key=rank_dirs.get) if any(rank_dirs.values()) else None
    rk = sorted(os.path.splitext(f)[0] for f in os.listdir(RK_IMG) if f.lower().endswith('.png')) if RK_IMG else []
    print('ranking set:', RK_IMG, len(rk), 'images', flush=True)
    if a.limit: tr, te, rk = tr[:a.limit], te[:a.limit], rk[:a.limit]

    # ---------------- annotations ----------------
    rows, xml_wh = [], {}
    for s in tr:
        W, H, objs = parse_xml(f'{TR_ANN}/{s}.xml'); xml_wh[s] = (W, H)
        rows += [{'stem': s, 'W': W, 'H': H, **o} for o in objs]
    ann = pd.DataFrame(rows)
    ann['w'], ann['h'] = ann.xmax - ann.xmin, ann.ymax - ann.ymin
    ann['side'] = np.sqrt(ann.w.clip(0) * ann.h.clip(0)); ann['ar'] = ann.w / ann.h.replace(0, np.nan)
    ann['bucket'] = pd.cut(ann.w * ann.h, [-1, 32 ** 2, 96 ** 2, np.inf], labels=['small', 'medium', 'large'])
    ann['border'] = (ann.xmin <= 1) | (ann.ymin <= 1) | (ann.xmax >= ann.W - 1) | (ann.ymax >= ann.H - 1)
    n_per_img = ann.groupby('stem').size().reindex(tr, fill_value=0)

    section('ANNOTATIONS')
    rep['n_train'], rep['n_test'], rep['n_boxes'] = len(tr), len(te), len(ann)
    rep['xml_sizes'] = {f'{w}x{h}': int(n) for (w, h), n in pd.Series(list(xml_wh.values())).value_counts().items()}
    rep['boxes_per_image'] = n_per_img.describe().round(2).to_dict(); rep['images_without_boxes'] = int((n_per_img == 0).sum())
    rep['unknown_classes'] = sorted(set(ann['name']) - set(classes))
    rep['size_buckets_coco'] = ann.bucket.value_counts(normalize=True).round(3).to_dict()
    rep['side_quantiles_px'] = dict(zip(['p5', 'p25', 'p50', 'p75', 'p95'], np.percentile(ann.side, [5, 25, 50, 75, 95]).round(1).tolist()))
    bad = ann[(ann.w < 2) | (ann.h < 2) | (ann.xmin < 0) | (ann.ymin < 0) | (ann.xmax > ann.W) | (ann.ymax > ann.H)]
    dups = 0
    for s, g in ann.groupby('stem'):
        if len(g) < 2: continue
        m = iou_matrix(g[['xmin', 'ymin', 'xmax', 'ymax']].values); same = g['name'].values[:, None] == g['name'].values[None]
        dups += int(np.triu((m > 0.8) & same, 1).sum())
    rep['label_sanity'] = {'tiny_or_out_of_bounds': len(bad), 'near_duplicate_same_class_iou>0.8': dups,
                           'difficult': int(ann.difficult.sum()), 'truncated': int(ann.truncated.sum()), 'touching_border': int(ann.border.sum())}
    cs = ann.groupby('name').agg(instances=('stem', 'size'), images=('stem', 'nunique'), side_p10=('side', lambda x: x.quantile(.1)),
                                 side_med=('side', 'median'), side_p90=('side', lambda x: x.quantile(.9)), ar_med=('ar', 'median'),
                                 pct_small=('bucket', lambda x: (x == 'small').mean() * 100), pct_border=('border', lambda x: x.mean() * 100))
    cs['per_img_when_present'] = cs.instances / cs.images
    cs = cs.reindex(classes).round(2); cs.to_csv(f'{a.out}/class_stats.csv')
    rep['class_stats'] = cs.reset_index().to_dict('records')
    for k in ('n_train', 'n_test', 'n_boxes', 'xml_sizes', 'boxes_per_image', 'images_without_boxes', 'unknown_classes', 'size_buckets_coco', 'side_quantiles_px', 'label_sanity'):
        print(f'{k}: {rep[k]}')
    print(cs.to_string())

    # co-occurrence: which classes share scenes
    P = pd.crosstab(ann['stem'], ann['name']).clip(upper=1).reindex(columns=classes, fill_value=0); p = P.mean()
    co = (P.T @ P) / len(P); lift = co / np.outer(p, p)
    pairs = [(classes[i], classes[j], int((P.iloc[:, i] & P.iloc[:, j]).sum()), round(float(lift.iloc[i, j]), 2))
             for i, j in itertools.combinations(range(len(classes)), 2) if co.iloc[i, j] > 0]
    pairs.sort(key=lambda r: -r[2])
    rep['classes_per_image'] = {int(k): int(v) for k, v in P.sum(1).value_counts().sort_index().items()}
    rep['top_cooccurring_pairs'] = pairs[:15]
    rep['material_pairs_cooccur'] = {g: int(P[cl].all(1).sum()) for g, cl in MATERIAL_GROUPS.items()}
    print('classes per image:', rep['classes_per_image'])
    print('top co-occurring pairs (a, b, images, lift):', pairs[:15])
    print('images containing ALL classes of a material group (real+fake side by side):', rep['material_pairs_cooccur'])

    # ---------------- per-image pass ----------------
    section('IMAGE PASS')
    by_stem = {s: g.to_dict('records') for s, g in ann.groupby('stem')}
    jobs = [(f'{TR_IMG}/{s}.png', s, 'val' if s in val else 'train', by_stem.get(s, []), xml_wh[s]) for s in tr]
    jobs += [(f'{TE_IMG}/{s}.png', s, 'test', [], None) for s in te] + [(f'{RK_IMG}/{s}.png', s, 'ranking', [], None) for s in rk]
    feats, thumbs, spectra, opx, bpx = [], [], [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (f, t, sp, o, b) in enumerate(ex.map(image_job, jobs, chunksize=8)):
            feats.append(f); thumbs.append(t); spectra += sp; opx.append(o); bpx.append(b)
            if i % 500 == 0: print(f'  {i}/{len(jobs)} images, {time.time() - t0:.0f}s', flush=True)
    F = pd.DataFrame(feats); F.to_csv(f'{a.out}/image_feats.csv', index=False)
    rep['raw'] = {'dtypes': F['dtype'].value_counts().to_dict(), 'raw_max_quantiles': np.percentile(F.raw_max, [0, 50, 100]).tolist(),
                  'cube_sizes': {f'{w}x{h}': int(n) for (w, h), n in F.groupby(['W', 'H']).size().items()},
                  'sat_frac_mean': round(float(F.sat_frac.mean()), 5), 'images_with_>1%_saturated': int((F.sat_frac > .01).sum())}
    print('raw:', rep['raw'])

    # ---------------- train vs test shift ----------------
    section('TRAIN vs TEST SHIFT')
    is_tr, is_te, is_val, is_rk = F['split'].isin(['train', 'val']).values, (F['split'] == 'test').values, (F['split'] == 'val').values, (F['split'] == 'ranking').values
    num = [c for c in F.columns if c not in ('stem', 'split', 'dtype', 'H', 'W')]
    ks = sorted(((c, ks_2samp(F.loc[is_tr, c], F.loc[is_te, c])) for c in num), key=lambda r: -r[1].statistic)
    rep['ks_top'] = [(c, round(float(r.statistic), 3), float(f'{r.pvalue:.2e}'), round(float(F.loc[is_tr, c].median()), 4), round(float(F.loc[is_te, c].median()), 4)) for c, r in ks[:10]]
    print('largest train/test feature shifts (feature, KS, p, train median, test median):'); [print('  ', r) for r in rep['ks_top']]

    def adv_auc(X, y):
        return round(float(cross_val_score(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05), X, y,
                                           cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc').mean()), 3)
    T = np.stack(thumbs); T -= T.mean(1, keepdims=True); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
    Tp = PCA(32, random_state=0).fit_transform(T)
    Xr = F[num].values
    rep['adversarial_auc'] = {'radiometric_train_vs_test': adv_auc(Xr[is_tr | is_te], is_te[is_tr | is_te]),
                              'radiometric_val_vs_train (control)': adv_auc(Xr[is_tr], is_val[is_tr]),
                              'scene_thumb_train_vs_test': adv_auc(Tp[is_tr | is_te], is_te[is_tr | is_te]),
                              'scene_thumb_val_vs_train (control)': adv_auc(Tp[is_tr], is_val[is_tr])}
    if is_rk.sum() >= 20:
        rep['adversarial_auc'].update({'radiometric_train_vs_ranking': adv_auc(Xr[is_tr | is_rk], is_rk[is_tr | is_rk]), 'radiometric_test_vs_ranking': adv_auc(Xr[is_te | is_rk], is_rk[is_te | is_rk]),
                                       'scene_thumb_train_vs_ranking': adv_auc(Tp[is_tr | is_rk], is_rk[is_tr | is_rk]), 'scene_thumb_test_vs_ranking': adv_auc(Tp[is_te | is_rk], is_rk[is_te | is_rk])})
    print('adversarial validation AUC (0.5 = indistinguishable):', rep['adversarial_auc'])

    def nn_sim(q, ref, self_ex=False):
        S = T[q] @ T[ref].T
        if self_ex: S[np.arange(len(q)), np.searchsorted(ref, q)] = -1
        m = S.max(1); return {'p10': round(float(np.percentile(m, 10)), 3), 'median': round(float(np.median(m)), 3), 'p90': round(float(np.percentile(m, 90)), 3),
                             '>=0.95': round(float((m >= .95).mean()), 3), '>=0.99': round(float((m >= .99).mean()), 3)}
    trn = np.flatnonzero(F['split'] == 'train'); rng = np.random.default_rng(0)
    rep['nearest_train_scene_similarity'] = {'val->train': nn_sim(np.flatnonzero(is_val), trn), 'test->train': nn_sim(np.flatnonzero(is_te), np.flatnonzero(is_tr)),
                                             **({'ranking->train': nn_sim(np.flatnonzero(is_rk), np.flatnonzero(is_tr))} if is_rk.any() else {}),
                                             'train->train (self excluded)': nn_sim(np.sort(rng.choice(trn, min(300, len(trn)), replace=False)), trn, True)}
    print('nearest-neighbour thumbnail cosine similarity (near-duplicate / same-scene check):')
    [print('  ', k, v) for k, v in rep['nearest_train_scene_similarity'].items()]

    # ---------------- spectral separability ----------------
    section('SPECTRAL ANALYSIS')
    S = pd.DataFrame(spectra, columns=['stem', 'name'] + [f'b{i}' for i in range(NB)]); S.to_csv(f'{a.out}/box_spectra.csv', index=False)
    X = S[[f'b{i}' for i in range(NB)]].values; shape = X / (X.mean(1, keepdims=True) + 1e-6)
    med = pd.DataFrame(shape, columns=range(NB)).groupby(S['name'].values).median().reindex(classes).round(3)
    rep['median_normalised_spectrum_per_class'] = med.to_dict('index')
    print('median illumination-normalised spectrum per class (band / mean over bands):'); print(med.to_string())
    pix = np.concatenate(opx + bpx); C = np.corrcoef(pix.T)
    ev = PCA(random_state=0).fit(pix).explained_variance_ratio_
    rep['band_correlation'] = {'mean_offdiag': round(float(C[~np.eye(NB, dtype=bool)].mean()), 3), 'min': round(float(C.min()), 3), 'argmin': list(map(int, np.unravel_index(C.argmin(), C.shape)))}
    rep['pca_cum_explained'] = np.cumsum(ev.astype(float))[:6].round(4).tolist()
    print('band correlation:', rep['band_correlation'], '| PCA cumulative explained variance:', rep['pca_cum_explained'])

    def bacc(Xf, y, g):
        if len(np.unique(g)) < 5 or len(np.unique(y)) < 2: return float('nan')
        pred = cross_val_predict(make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()), Xf, y, groups=g, cv=GroupKFold(5))
        return round(float(balanced_accuracy_score(y, pred)), 3)
    triplets = list(itertools.combinations(range(NB), 3))

    def band_study(mask):
        y_all, g_all = S['name'].values, S['stem'].values
        Xm, sm, y, g = X[mask], shape[mask], y_all[mask], g_all[mask]
        res = {'n': pd.Series(y).value_counts().to_dict(), 'chance': round(1 / len(np.unique(y)), 3), 'gray(1)': bacc(Xm.mean(1, keepdims=True), y, g),
               f'current{CUR_BANDS}': bacc(Xm[:, CUR_BANDS], y, g), 'all16': bacc(Xm, y, g), 'all16_shape_only': bacc(sm, y, g)}
        sc = [(bacc(Xm[:, list(t)], y, g), t) for t in triplets]; sc.sort(key=lambda r: -r[0])
        res['best_triplets'] = [(list(t), s) for s, t in sc[:3]]
        return res
    rep['material_discrimination'] = {}
    for grp, cl in MATERIAL_GROUPS.items():
        m = S['name'].isin(cl).values
        if m.sum() < 20: continue
        rep['material_discrimination'][grp] = r = band_study(m); print(f'{grp:7s}', r)
    rep['all_classes_spectral_only'] = r = band_study(np.ones(len(S), bool)); print('ALL 18 classes, box-mean spectrum only:', r)
    sel, rest, order = [], list(range(NB)), []
    for _ in range(6):  # greedy forward band selection for 18-class spectral discrimination
        scores = {b: bacc(X[:, sel + [b]], S['name'].values, S['stem'].values) for b in rest}
        best = max(scores, key=scores.get); sel.append(best); rest.remove(best); order.append((best, scores[best]))
    rep['greedy_band_order'] = order; print('greedy band selection (band, balanced acc after adding):', order)

    rep['runtime_s'] = round(time.time() - t0)
    json.dump(rep, open(f'{a.out}/eda_report.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f'\nwrote {a.out}/eda_report.json in {rep["runtime_s"]}s', flush=True)


if __name__ == '__main__':
    main()
