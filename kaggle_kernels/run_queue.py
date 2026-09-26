"""Keep N Kaggle GPU kernels busy from an experiment queue; collect outputs and log val scores.

usage: nohup python kaggle_kernels/run_queue.py --queue kaggle_kernels/queue.json --max_running 2 &
queue.json: [{"slug": "...", "args": ["--model", "yolo26m.pt", "--imgsz", "1024", ...]}, ...]
Results are appended to kaggle_kernels/results.tsv ; outputs land in kaggle_out/<slug>/.
"""
import argparse, json, os, subprocess, time, glob, re
ap = argparse.ArgumentParser(); ap.add_argument('--queue', required=True); ap.add_argument('--max_running', type=int, default=2); ap.add_argument('--poll', type=int, default=180)
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results.tsv'); OUT = os.path.join(ROOT, 'kaggle_out')
def status(slug):
    r = subprocess.run(['kaggle', 'kernels', 'status', f'hosen42/{slug}'], capture_output=True, text=True)
    m = re.search(r'KernelWorkerStatus\.([A-Z_]+)', r.stdout + r.stderr); return m.group(1) if m else 'UNKNOWN'
def push(exp):
    r = subprocess.run(['python3', os.path.join(HERE, 'push_experiment.py'), '--slug', exp['slug'], *exp['args']], capture_output=True, text=True)
    ok = 'successfully pushed' in r.stdout
    print(time.strftime('%H:%M'), 'pushed' if ok else 'PUSH FAILED', exp['slug'], r.stdout.strip()[-120:], r.stderr.strip()[-200:], flush=True)
    return ok
def collect(slug):
    d = os.path.join(OUT, slug); os.makedirs(d, exist_ok=True)
    subprocess.run(['kaggle', 'kernels', 'output', f'hosen42/{slug}', '-p', d], capture_output=True, text=True)
    sc = os.path.join(d, 'val_score.json'); line = f"{slug}\t{time.strftime('%m-%d %H:%M')}\t"
    if os.path.exists(sc):
        j = json.load(open(sc)); pc = j.get('per_class', {})
        line += f"{j.get('mAP50-95', float('nan')):.4f}\t{j.get('mAP50', float('nan')):.4f}\t{j.get('model')}\t{j.get('imgsz')}\t{j.get('epochs')}\t{j.get('batch')}\t{j.get('bands')}\t{j.get('extra')}\t" + ' '.join(f"{k}:{v:.2f}" for k, v in pc.items())
    else:
        log = glob.glob(os.path.join(d, '*.log')); err = ''
        if log:
            try: err = ' | '.join(e['data'].strip()[:200] for e in json.load(open(log[0])) if 'Error' in e.get('data', '') or 'Traceback' in e.get('data', ''))[:600]
            except Exception: pass
        line += f"FAILED\t{err}"
    open(RES, 'a').write(line + '\n'); print(time.strftime('%H:%M'), 'collected', line[:160], flush=True)
done_slugs = set(l.split('\t')[0] for l in open(RES)) if os.path.exists(RES) else set()
running = {}
while True:
    queue = [e for e in json.load(open(a.queue)) if e['slug'] not in done_slugs and e['slug'] not in running]
    for slug in list(running):
        st = status(slug)
        if st in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED'):
            collect(slug); done_slugs.add(slug); del running[slug]
    while len(running) < a.max_running and queue:
        exp = queue.pop(0)
        st = status(exp['slug'])
        if st in ('RUNNING', 'QUEUED'):  # already launched manually: adopt, don't restart
            print(time.strftime('%H:%M'), 'adopting running kernel', exp['slug'], flush=True); running[exp['slug']] = time.time(); continue
        if push(exp): running[exp['slug']] = time.time(); time.sleep(60)
        else: break  # session limit or transient error: retry next poll
    if not running and not queue: print('queue empty, exiting', flush=True); break
    time.sleep(a.poll)
