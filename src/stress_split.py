"""Build the STRESS validation subset: the hold-out (val) images that look most like the test set.

The EDA found a strong train-vs-test radiometric shift (adversarial AUC ~0.89; val-vs-train control ~0.51), mainly in the
relative band intensities (illumination spectrum). A classifier trained on train(2700, non-val) vs test(1000) image
features scores each val image by P(test-like); the top third becomes the stress subset. No labels and no ranking images
are used, and no model is trained on val.

usage: python src/stress_split.py [--feats kaggle_out/hod-eda/eda/image_feats.csv] [--frac 0.334]
writes splits/val_all.txt, splits/val_stress.txt, splits/val_testlike_scores.csv
"""
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--feats', default=os.path.join(ROOT, 'kaggle_out/hod-eda/eda/image_feats.csv'))
ap.add_argument('--frac', type=float, default=0.334)
a = ap.parse_args()

F = pd.read_csv(a.feats, dtype={'stem': str})
num = [c for c in F.columns if c not in ('stem', 'split', 'dtype', 'H', 'W')]
fit = F[F.split.isin(['train', 'test'])]
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
auc = cross_val_score(clf, fit[num], fit.split == 'test', cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc').mean()
clf.fit(fit[num], fit.split == 'test')
val = F[F.split == 'val'].copy(); val['p_testlike'] = clf.predict_proba(val[num])[:, 1]
val = val.sort_values('p_testlike', ascending=False)
k = int(round(len(val) * a.frac)); stress = val.stem.head(k).tolist()

os.makedirs(os.path.join(ROOT, 'splits'), exist_ok=True)
open(os.path.join(ROOT, 'splits/val_all.txt'), 'w').write('\n'.join(sorted(val.stem, key=int)) + '\n')
open(os.path.join(ROOT, 'splits/val_stress.txt'), 'w').write('\n'.join(sorted(stress, key=int)) + '\n')
val[['stem', 'p_testlike']].to_csv(os.path.join(ROOT, 'splits/val_testlike_scores.csv'), index=False)
test_p = clf.predict_proba(F[F.split == 'test'][num])[:, 1]  # in-sample, for scale reference only
print(json.dumps({'adversarial_auc_train_vs_test_cv': round(float(auc), 3), 'val_images': len(val), 'stress_images': k,
                  'p_testlike_val_quantiles': np.percentile(val.p_testlike, [10, 50, 90]).round(3).tolist(),
                  'p_testlike_stress_min': round(float(val.p_testlike.head(k).min()), 3),
                  'p_testlike_test_insample_median': round(float(np.median(test_p)), 3)}, indent=1))
