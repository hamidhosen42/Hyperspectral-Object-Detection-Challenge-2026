"""Assemble a Phase 2 submission (test + ranking, one CSV) from predictions of ONE checkpoint.

The test half usually comes from the checkpoint's own Kaggle training kernel (submission.csv, test only); the ranking half
comes from local inference (src/infer.py --split ranking) with the same weights and preprocessing.

usage: python src/make_submission.py --test_pred kaggle_out/hod-y26s-1024/submission.csv --ranking_pred preds/E0b_ranking.csv \
           --weights kaggle_out/hod-y26s-1024/best.pt --out submissions/phase2_E0b.csv
Checks: id sets are disjoint, the test ids match the 1000 official test images, and the ranking ids match the 1000 ranking images.
It also records the sha256 of the single checkpoint used for both halves.
"""
import argparse, os, hashlib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--test_pred', required=True); ap.add_argument('--ranking_pred', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--weights', required=True, help='the single checkpoint both halves were produced with (recorded by sha256)')
ap.add_argument('--allow_partial_ranking', action='store_true', help='submit even if some ranking images are not downloaded yet (they score as misses)')
a = ap.parse_args()

test_ids = set(pd.read_csv(os.path.join(ROOT, 'kaggle_out/hod-eda/eda/image_feats.csv'), dtype={'stem': str}).query("split == 'test'").stem.astype(int))
rank_dir = os.path.join(ROOT, 'data/data_ranking/data_ranking/VIS')
rank_ids = {int(os.path.splitext(f)[0]) for f in os.listdir(rank_dir) if f.endswith('.png')}
assert len(test_ids) == 1000, len(test_ids)
if len(rank_ids) != 1000:
    assert a.allow_partial_ranking, f'only {len(rank_ids)} ranking images on disk'
    print(f'WARNING: PARTIAL ranking set: {len(rank_ids)}/1000 images on disk')
assert not test_ids & rank_ids, 'test and ranking ids overlap'

cols = ['image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2']
t = pd.read_csv(a.test_pred)[cols]; r = pd.read_csv(a.ranking_pred)[cols]
t = t[t.image_id.isin(test_ids)]; r = r[r.image_id.isin(rank_ids)]
df = pd.concat([t, r]).sort_values(['image_id', 'confidence'], ascending=[True, False]).reset_index(drop=True)
df.insert(0, 'id', range(len(df)))
df['confidence'] = df.confidence.round(5); df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].round(2)
bad = (df.x2 <= df.x1) | (df.y2 <= df.y1)  # zero-area after rounding: can never match a GT box
if bad.any(): print(f'dropping {int(bad.sum())} zero-area boxes'); df = df[~bad].reset_index(drop=True); df['id'] = range(len(df))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); df.to_csv(a.out, index=False)
sha = hashlib.sha256(open(a.weights, 'rb').read()).hexdigest()[:16]
print(f'wrote {a.out}: {len(df)} rows | test images with boxes {t.image_id.nunique()}/1000 | ranking images with boxes {r.image_id.nunique()}/1000 '
      f'| weights sha256[:16] {sha} | csv sha256 {hashlib.sha256(open(a.out, "rb").read()).hexdigest()[:16]}')
