"""Download competition files one by one (the whole-competition zip is not downloadable) into data/.

usage: python fetch_data.py [--sets val test ranking train] [--workers 16]
  val      = the 300 hold-out train images (seed-0 split, same as training) + their XML annotations
  train    = all train images + XML annotations
  test / ranking = all test / ranking images
Keeps the competition layout (data/data_train/data_train/VIS/..., data/data_ranking/data_ranking/VIS/...), skips files
already present, and is safe to re-run.
"""
import argparse, os, random, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = 'hyperspectral-object-detection-challenge-2026'
MIRROR = 'moderantnoukoussi/hyperspectral-object-detection-challenge-2026'
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ap = argparse.ArgumentParser()
ap.add_argument('--sets', nargs='+', default=['val', 'test', 'ranking'], choices=['val', 'train', 'test', 'ranking'])
ap.add_argument('--source', default='competition', choices=['competition', 'mirror'], help='mirror = public dataset copy (train+test only, no ranking)')
ap.add_argument('--workers', type=int, default=8)  # the API rate-limits (HTTP 429) above ~16 parallel requests
a = ap.parse_args()

api = KaggleApi(); api.authenticate()
names, token = [], None
while True:
    r = api.competition_list_files(COMP, page_token=token, page_size=200) if a.source == 'competition' else api.dataset_list_files(MIRROR, page_token=token, page_size=200)
    names += [f.name for f in r.files]; token = r.next_page_token
    if not token: break
print(len(names), 'files listed', flush=True)

train_png = sorted(n for n in names if n.startswith('data_train/') and n.endswith('.png'))
stem = lambda n: os.path.splitext(os.path.basename(n))[0]
shuf = sorted(stem(n) for n in train_png); random.Random(0).shuffle(shuf); val = set(shuf[:int(len(shuf) * 0.1)])
want = {'class.txt'}
for n in names:
    s = stem(n)
    if 'test' in a.sets and n.startswith('data_test/') and n.endswith('.png'): want.add(n)
    if 'ranking' in a.sets and n.startswith('data_ranking/') and n.endswith('.png'): want.add(n)
    if n.startswith('data_train/') and n.endswith(('.png', '.xml')) and ('train' in a.sets or ('val' in a.sets and s in val)): want.add(n)
os.makedirs(ROOT, exist_ok=True); open(os.path.join(ROOT, 'val_stems.txt'), 'w').write('\n'.join(sorted(val)) + '\n')
todo = sorted(n for n in want if not os.path.exists(os.path.join(ROOT, n)))
print(f'{len(want)} wanted, {len(todo)} to download', flush=True)


def get(n):
    d = os.path.join(ROOT, os.path.dirname(n)); tmp = os.path.join(ROOT, '.tmp', n.replace('/', '__')); os.makedirs(tmp, exist_ok=True)
    for attempt in range(8):
        try:  # download into a private temp dir, then move: an interrupted run never leaves a half-written file behind
            if a.source == 'competition': api.competition_download_file(COMP, n, path=tmp, quiet=True)
            else: api.dataset_download_file(MIRROR, n, path=tmp, quiet=True)
            z = os.path.join(tmp, os.path.basename(n) + '.zip')  # large files may arrive zipped
            if os.path.exists(z):
                with zipfile.ZipFile(z) as f: f.extractall(tmp)
                os.remove(z)
            os.makedirs(d, exist_ok=True); os.replace(os.path.join(tmp, os.path.basename(n)), os.path.join(ROOT, n)); os.rmdir(tmp)
            return n
        except Exception as e:
            if attempt == 7: return f'FAILED {n}: {str(e)[:120]}'
            time.sleep(min(60, (15 if '429' in str(e) else 2) * 2 ** attempt))  # back off harder when rate-limited


t0 = time.time(); done = 0; failed = []
with ThreadPoolExecutor(a.workers) as ex:
    for f in as_completed([ex.submit(get, n) for n in todo]):
        r = f.result(); done += 1
        if r.startswith('FAILED'): failed.append(r); print(r, flush=True)
        if done % 200 == 0: print(f'{done}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
print(f'done {done} files in {time.time() - t0:.0f}s, {len(failed)} failed (re-run to retry)', flush=True)
