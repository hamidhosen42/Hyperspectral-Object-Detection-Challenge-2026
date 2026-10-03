"""Render the write-up / README figures (PNG, 1920x1080) from measured results.

usage: python assets/make_figures.py   ->  assets/*.png
All numbers are the measured values recorded in experiments.csv / kaggle_out/hod-eda/eda/eda_report.json.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'  # reference categorical slots 1-3 (all-pairs CVD-safe)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
W, H, DPI = 12, 6.75, 160  # 1920 x 1080


def frame(title, subtitle):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.text(0.06, 0.93, title, fontsize=24, fontweight='bold', color=INK, va='top')
    fig.text(0.06, 0.865, subtitle, fontsize=15, color=INK2, va='top')
    fig.text(0.06, 0.03, 'Hyperspectral Object Detection Challenge 2026 · github.com/hamidhosen42', fontsize=11, color=MUTED)
    return fig


def tidy(ax, grid_axis='x'):
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(length=0)


def cover():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.text(0.07, 0.80, 'Hyperspectral Object Detection Challenge 2026', fontsize=20, color=INK2)
    fig.text(0.07, 0.66, '16 bands beat any 3-band composite', fontsize=34, fontweight='bold', color=INK)
    fig.text(0.07, 0.58, 'A single-checkpoint YOLO solution — and what hold-out validation missed', fontsize=20, color=INK2)
    for x, big, small, col in [(0.07, '+0.026', 'test mAP@[.5:.95]\n16 bands vs pseudo-RGB', BLUE),
                               (0.37, '0.5955', 'best public test score\nYOLO26s · 16 bands', INK),
                               (0.67, 'AUC 0.90', 'train / test / ranking are\nseparable acquisition domains', ORANGE)]:
        fig.text(x, 0.36, big, fontsize=44, fontweight='bold', color=col)
        fig.text(x, 0.22, small, fontsize=16, color=INK2, linespacing=1.4)
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.50, 0.50], color=GRID, lw=2))
    fig.text(0.07, 0.07, 'Md. Hamid Hosen · github.com/hamidhosen42/Hyperspectral-Object-Detection-Challenge-2026', fontsize=13, color=MUTED)
    fig.savefig(f'{OUT}/01_cover.png', facecolor=SURF); plt.close(fig)


def results():
    rows = [('E0k  YOLO11s', 'pseudo-RGB', 0.5649), ('E0a  YOLO11s', 'pseudo-RGB', 0.5791), ('E0b  YOLO26s', 'pseudo-RGB', 0.5803),
            ('E4a  YOLO11s', '16 bands', 0.5912), ('E4c  YOLO26s', '16 bands', 0.5955)]
    fig = frame('Public test score by model and input', 'COCO mAP@[.5:.95] on the 1,000-image test set · each point is one single-checkpoint submission')
    ax = fig.add_axes([0.20, 0.16, 0.72, 0.62]); tidy(ax)
    for i, (name, inp, v) in enumerate(rows):
        c = BLUE if inp == '16 bands' else ORANGE
        ax.plot([0.555, v], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter(v, i, s=180, color=c, edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(v + 0.0012, i, f'{v:.4f}', va='center', fontsize=15, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=15, color=INK)
    ax.set_xlim(0.555, 0.603); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlabel('test mAP@[.5:.95]')
    h = [plt.Line2D([], [], marker='o', ls='', ms=12, color=BLUE, label='all 16 bands'), plt.Line2D([], [], marker='o', ls='', ms=12, color=ORANGE, label='pseudo-RGB (bands 5/8/13)')]
    ax.legend(handles=h, loc='lower right', frameon=False, fontsize=14)
    fig.savefig(f'{OUT}/02_results.png', facecolor=SURF); plt.close(fig)


def val_vs_test():
    fig = frame('The 16-band gain is invisible on a random hold-out', 'Same YOLO11s recipe, only the input changes · mAP@[.5:.95]')
    ax = fig.add_axes([0.24, 0.16, 0.68, 0.62]); tidy(ax)
    data = [('Hold-out val\n(same domain as train)', 0.6960, 0.6992), ('Public test\n(shifted domain)', 0.5649, 0.5912)]
    for i, (lab, a, b) in enumerate(data):
        ax.plot([a, b], [i, i], color=GRID, lw=6, solid_capstyle='round', zorder=1)
        ax.scatter(a, i, s=200, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=3); ax.scatter(b, i, s=200, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(a - 0.002, i + 0.16, f'{a:.3f}', ha='right', fontsize=14, color=INK2); ax.text(b + 0.002, i + 0.16, f'{b:.3f}', ha='left', fontsize=14, color=INK2)
        ax.text(max(a, b) + 0.004, i - 0.02, f'{b - a:+.3f}', va='center', fontsize=18, fontweight='bold', color=INK)
    ax.set_yticks([0, 1]); ax.set_yticklabels([d[0] for d in data], fontsize=15, color=INK)
    ax.set_xlim(0.55, 0.73); ax.set_ylim(-0.6, 1.6); ax.set_xlabel('mAP@[.5:.95]')
    h = [plt.Line2D([], [], marker='o', ls='', ms=12, color=ORANGE, label='pseudo-RGB (3 bands)'), plt.Line2D([], [], marker='o', ls='', ms=12, color=BLUE, label='all 16 bands')]
    ax.legend(handles=h, loc='lower right', frameon=False, fontsize=14)
    fig.savefig(f'{OUT}/03_val_vs_test.png', facecolor=SURF); plt.close(fig)


def domain_shift():
    rows = [('val vs train (control)', 0.51), ('train vs test', 0.89), ('train vs ranking', 0.90), ('test vs ranking', 0.90)]
    fig = frame('Train, test and ranking are three acquisition domains', 'Adversarial validation: classifier on per-image band statistics · ROC-AUC (0.5 = indistinguishable)')
    ax = fig.add_axes([0.24, 0.16, 0.68, 0.62]); tidy(ax)
    for i, (lab, v) in enumerate(rows):
        c = MUTED if 'control' in lab else BLUE
        ax.barh(i, v, height=0.56, color=c, zorder=2)
        ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=16, color=INK)
    ax.axvline(0.5, color=INK2, lw=1.5, ls=(0, (4, 3)), zorder=3)
    ax.text(0.51, len(rows) - 0.42, 'chance', ha='left', fontsize=13, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=15, color=INK)
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(rows) - 0.2); ax.set_xlabel('ROC-AUC')
    fig.savefig(f'{OUT}/04_domain_shift.png', facecolor=SURF); plt.close(fig)


def spectral():
    groups = ['apple vs\nplastic', 'banana vs\nplastic', 'orange vs\nplastic', 'egg / plastic\n/ wood', 'car vs\ntoy car']
    sets = [('pseudo-RGB (bands 5/8/13)', ORANGE, [0.867, 0.681, 0.906, 0.686, 0.748]),
            ('all 16 bands', BLUE, [0.998, 0.927, 0.985, 0.949, 0.947]),
            ('16 bands, brightness-normalised', AQUA, [1.000, 0.996, 0.998, 0.946, 0.944])]
    fig = frame('Spectra separate the look-alike materials', 'Balanced accuracy of a linear classifier on the mean spectrum inside each box · image-grouped 5-fold CV')
    ax = fig.add_axes([0.08, 0.25, 0.86, 0.52]); tidy(ax, 'y'); ax.spines['bottom'].set_color(GRID)
    w = 0.26
    for k, (lab, c, vals) in enumerate(sets):
        xs = [g + (k - 1) * (w + 0.02) for g in range(len(groups))]
        ax.bar(xs, vals, width=w, color=c, label=lab, zorder=2)
        for x, v in zip(xs, vals): ax.text(x, v + 0.012, f'{v:.2f}', ha='center', fontsize=11, color=INK2)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=14, color=INK)
    ax.set_ylim(0, 1.08); ax.set_ylabel('balanced accuracy')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=13)
    fig.savefig(f'{OUT}/05_spectral_separability.png', facecolor=SURF); plt.close(fig)


def pipeline():
    fig = frame('Pipeline', 'Identical preprocessing for train, test and ranking · one checkpoint produces the whole submission')
    steps = [('16-bit mosaic\nPNG', '4×4 snapshot\nfilter pattern'), ('X2Cube', '→ ~512×256×16\ncube'), ('Percentile\nscaling', '0.5–99.5% → uint8\n16-page TIFF'),
             ('YOLO 11s / 26s', '16-channel stem\nCOCO-pretrained\n1024 px · 50 ep'), ('One checkpoint', 'test + ranking\npredictions → CSV')]
    n = len(steps); x0, gap = 0.06, 0.035; bw = (0.88 - gap * (n - 1)) / n
    for i, (head, body) in enumerate(steps):
        x = x0 + i * (bw + gap); col = BLUE if i == 3 else INK2
        fig.add_artist(FancyBboxPatch((x, 0.30), bw, 0.40, boxstyle='round,pad=0.004,rounding_size=0.012', transform=fig.transFigure,
                                      facecolor='#ffffff', edgecolor=col, lw=2))
        fig.text(x + bw / 2, 0.60, head, ha='center', va='center', fontsize=15, fontweight='bold', color=INK)
        fig.text(x + bw / 2, 0.43, body, ha='center', va='center', fontsize=13, color=INK2, linespacing=1.4)
        if i < n - 1: fig.text(x + bw + gap / 2, 0.50, '→', ha='center', va='center', fontsize=22, color=INK2)
    fig.text(0.06, 0.17, 'Validation: 300-image hold-out (ALL) · 100 most test-like images (STRESS) · train→ranking band-gain shift-stress',
             fontsize=13, color=INK2)
    fig.savefig(f'{OUT}/06_pipeline.png', facecolor=SURF); plt.close(fig)


if __name__ == '__main__':
    cover(); results(); val_vs_test(); domain_shift(); spectral(); pipeline()
    print('\n'.join(sorted(f for f in os.listdir(OUT) if f.endswith('.png'))))
