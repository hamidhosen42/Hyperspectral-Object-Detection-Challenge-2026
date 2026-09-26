"""Create and push one Kaggle GPU experiment kernel.

usage: python kaggle_kernels/push_experiment.py --slug hod-y26s-1024 --model yolo26s.pt --imgsz 1024 --epochs 50 --batch 16 [--bands 5 8 13] [--extra scale=0.7] [--full]
"""
import argparse, os, json, re, subprocess, shutil
ap = argparse.ArgumentParser()
ap.add_argument('--slug', required=True); ap.add_argument('--model', required=True)
ap.add_argument('--imgsz', type=int, default=1024); ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--bands', nargs='+', type=int, default=[5, 8, 13]); ap.add_argument('--extra', nargs='*', default=[]); ap.add_argument('--full', action='store_true')
ap.add_argument('--stem_init', default='rgb3', choices=['rgb3', 'mean']); ap.add_argument('--norm', default='global', choices=['global', 'band'])
ap.add_argument('--pseudo_from', default='', help='teacher kernel slug; its TEST-only submission.csv becomes pseudo-labels')
ap.add_argument('--pseudo_conf', type=float, default=0.5)
a = ap.parse_args()
extra = {}
for kv in a.extra:
    k, v = kv.split('=', 1)
    try: v = json.loads(v)
    except Exception: pass
    extra[k] = v
here = os.path.dirname(os.path.abspath(__file__)); d = os.path.join(here, a.slug); os.makedirs(d, exist_ok=True)
src = open(os.path.join(here, 'template/train_kernel.py')).read()
cfg = f"MODEL = {a.model!r}\nIMGSZ = {a.imgsz}\nEPOCHS = {a.epochs}\nBATCH = {a.batch}\nBANDS = {a.bands}\nEXTRA = {extra!r}\nFULL_DATA = {a.full}\nSTEM_INIT = {a.stem_init!r}\nNORM = {a.norm!r}\nPSEUDO_FROM = {(f'**/{a.pseudo_from}/submission.csv' if a.pseudo_from else '')!r}\nPSEUDO_CONF = {a.pseudo_conf}\n"
src = re.sub(r"# ---- CONFIG.*?# -{10,}\n", "# ---- CONFIG ----\n" + cfg + "# ----------------\n", src, flags=re.S)
open(os.path.join(d, 'train_kernel.py'), 'w').write(src)
json.dump({'id': f'hosen42/{a.slug}', 'title': a.slug, 'code_file': 'train_kernel.py', 'language': 'python', 'kernel_type': 'script',
           'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True, 'dataset_sources': ['moderantnoukoussi/hyperspectral-object-detection-challenge-2026'], 'kernel_sources': [f'hosen42/{a.pseudo_from}'] if a.pseudo_from else [], 'model_sources': [],
           'competition_sources': ['hyperspectral-object-detection-challenge-2026']}, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
print(subprocess.run(['kaggle', 'kernels', 'push', '-p', d], capture_output=True, text=True).stdout.strip())
