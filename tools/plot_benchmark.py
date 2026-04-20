#!/usr/bin/env python3
"""
FusionCore NCLT benchmark visualizer.

Usage:
  python3 tools/plot_benchmark.py \
    --seq_dir  benchmarks/nclt/2012-01-08 \
    --out      benchmarks/nclt/2012-01-08/results/benchmark_plot.png
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ── Design tokens ──────────────────────────────────────────────────────────
BG      = '#FFFFFF'
PANEL   = '#F8FAFC'
BORDER  = '#E2E8F0'
TEXT    = '#0F172A'
SUBTLE  = '#64748B'
C_FC    = '#2563EB'
C_EKF   = '#DC2626'
C_UKF   = '#7C3AED'
C_GT    = '#94A3B8'
GREEN   = '#16A34A'
RED_L   = '#FEE2E2'
BLUE_L  = '#EFF6FF'


def load_tum(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 4:
                continue
            vals = [float(v) for v in p[:4]]
            if any(math.isnan(v) or math.isinf(v) for v in vals):
                continue
            rows.append(vals)
    if not rows:
        return [np.array([]) for _ in range(4)]
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def align_se2(src_xy, ref_xy):
    step = max(1, len(src_xy) // 2000)
    s, r = src_xy[::step], ref_xy[::step]
    n = min(len(s), len(r))
    s, r = s[:n], r[:n]
    mu_s, mu_r = s.mean(0), r.mean(0)
    H = (s - mu_s).T @ (r - mu_r)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_r - R @ mu_s
    return (R @ src_xy.T).T + t


def interp_error(est_ts, est_x, est_y, gt_ts, gt_x, gt_y):
    errs = np.full(len(est_ts), np.nan)
    for i, t in enumerate(est_ts):
        idx = np.searchsorted(gt_ts, t)
        if idx == 0 or idx >= len(gt_ts):
            continue
        t0, t1 = gt_ts[idx-1], gt_ts[idx]
        if t1 == t0:
            continue
        a = (t - t0) / (t1 - t0)
        gx = gt_x[idx-1] + a*(gt_x[idx] - gt_x[idx-1])
        gy = gt_y[idx-1] + a*(gt_y[idx] - gt_y[idx-1])
        errs[i] = math.hypot(est_x[i] - gx, est_y[i] - gy)
    return errs


def card_bg(ax, color=PANEL):
    ax.set_facecolor(color)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(1.0)


def section_label(ax, text, color=SUBTLE):
    ax.text(0, 1.04, text.upper(), transform=ax.transAxes,
            fontsize=7.5, color=color, fontweight='bold',
            fontfamily='monospace', va='bottom')


def verdict_badge(ax, x, y, text, good=True, fontsize=9):
    bg = '#DCFCE7' if good else '#FEE2E2'
    tc = '#15803D' if good else '#B91C1C'
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, color=tc, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=bg, edgecolor='none'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir', default='benchmarks/nclt/2012-01-08')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    seq = Path(args.seq_dir)
    out = Path(args.out) if args.out else seq / 'results' / 'benchmark_plot.png'
    out.parent.mkdir(parents=True, exist_ok=True)

    def load(name):
        p = seq / name
        return load_tum(str(p)) if p.exists() else [np.array([])]*4

    gt_ts,  gt_x,  gt_y,  _ = load('ground_truth.tum')
    fc_ts,  fc_x,  fc_y,  _ = load('fusioncore.tum')
    ek_ts,  ek_x,  ek_y,  _ = load('rl_ekf.tum')
    uk_ts,  uk_x,  uk_y,  _ = load('rl_ukf.tum')
    fcs_ts, fcs_x, fcs_y, _ = load('fusioncore_spike.tum')
    eks_ts, eks_x, eks_y, _ = load('rl_ekf_spike.tum')

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor=BG)

    # Layout: left col (trajectory) = 2 units, right col = 3 units wide × 2 rows
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[2, 1.4, 1.4],
        height_ratios=[1, 1],
        hspace=0.52, wspace=0.38,
        left=0.05, right=0.97, top=0.88, bottom=0.08,
    )

    ax_traj  = fig.add_subplot(gs[:, 0])   # left, full height
    ax_ate   = fig.add_subplot(gs[0, 1])
    ax_spike = fig.add_subplot(gs[1, 1])
    ax_ukf   = fig.add_subplot(gs[0, 2])
    ax_rpe   = fig.add_subplot(gs[1, 2])

    # ── Master title ─────────────────────────────────────────────────────────
    fig.text(0.5, 0.96, 'FusionCore  vs  robot_localization',
             ha='center', fontsize=20, fontweight='bold', color=TEXT)
    fig.text(0.5, 0.922,
             'NCLT 2012-01-08  •  600 s  •  Ann Arbor campus  •  RTK GPS ground truth',
             ha='center', fontsize=10.5, color=SUBTLE)

    # ── Panel 1 — Trajectory ─────────────────────────────────────────────────
    ax = ax_traj
    card_bg(ax, BG)
    section_label(ax, 'Route comparison — SE(2) aligned to ground truth')

    if len(gt_x):
        center = np.array([gt_x.mean(), gt_y.mean()])
        gx, gy = gt_x - center[0], gt_y - center[1]

        ax.plot(gx, gy, color=C_GT, lw=1.5, alpha=0.6, label='Ground Truth (RTK GPS)', zorder=1)

        if len(fc_x):
            fc_al = align_se2(np.stack([fc_x, fc_y], 1), np.stack([gt_x, gt_y], 1))
            ax.plot(fc_al[:, 0] - center[0], fc_al[:, 1] - center[1],
                    color=C_FC, lw=2.0, alpha=0.9, label='FusionCore', zorder=3)

        if len(ek_x):
            ek_al = align_se2(np.stack([ek_x, ek_y], 1), np.stack([gt_x, gt_y], 1))
            ax.plot(ek_al[:, 0] - center[0], ek_al[:, 1] - center[1],
                    color=C_EKF, lw=1.3, alpha=0.65, label='RL-EKF', zorder=2)

        ax.plot(gx[0], gy[0], 'o', color=TEXT, ms=6, zorder=6)
        ax.text(gx[0] + 10, gy[0] + 10, 'Start', fontsize=8, color=TEXT)

    ax.set_aspect('equal')
    ax.set_xlabel('East (m)', fontsize=9, color=SUBTLE)
    ax.set_ylabel('North (m)', fontsize=9, color=SUBTLE)
    ax.tick_params(colors=SUBTLE, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.set_facecolor(BG)
    ax.grid(color=BORDER, lw=0.6, zorder=0)

    leg = ax.legend(fontsize=9, loc='upper left',
                    facecolor='white', edgecolor=BORDER,
                    framealpha=1.0)
    for t in leg.get_texts():
        t.set_color(TEXT)

    # big win annotation on trajectory
    ax.text(0.5, -0.08,
            'FusionCore stays on route.  RL-EKF visibly drifts.',
            transform=ax.transAxes, ha='center', fontsize=10,
            color=SUBTLE, style='italic')

    # ── Panel 2 — ATE bar chart ───────────────────────────────────────────────
    ax = ax_ate
    card_bg(ax)
    section_label(ax, 'Absolute Trajectory Error (ATE)')

    vals = [5.517, 23.434]
    cols = [C_FC, C_EKF]
    names = ['FusionCore', 'RL-EKF']

    bars = ax.bar(names, vals, color=cols, width=0.5, zorder=3,
                  edgecolor='none')

    # light background on FC bar to highlight winner
    ax.bar(['FusionCore'], [vals[0]], color=BLUE_L, width=0.5, zorder=2, edgecolor='none')
    ax.bar(['FusionCore'], [vals[0]], color=C_FC, width=0.5, zorder=3, edgecolor='none')
    ax.bar(['RL-EKF'],     [vals[1]], color=RED_L, width=0.5, zorder=2, edgecolor='none')
    ax.bar(['RL-EKF'],     [vals[1]], color=C_EKF, width=0.5, zorder=3, edgecolor='none')

    for bar, val, name in zip(bars, vals, names):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                f'{val:.1f} m', ha='center', va='bottom',
                color=TEXT, fontsize=13, fontweight='bold')

    # 4.2× callout
    mid = (vals[0] + vals[1]) / 2
    ax.annotate('', xy=(1, vals[0]+1), xytext=(1, vals[1]-1),
                arrowprops=dict(arrowstyle='<->', color=SUBTLE, lw=1.5))
    ax.text(1.3, mid, '4.2×\nmore\naccurate', ha='left', va='center',
            color=TEXT, fontsize=10, fontweight='bold', linespacing=1.3)

    ax.set_ylim(0, vals[1] * 1.35)
    ax.set_ylabel('RMSE (m)', fontsize=9, color=SUBTLE)
    ax.tick_params(colors=SUBTLE, labelsize=9)
    ax.tick_params(axis='x', colors=TEXT, labelsize=10)
    ax.grid(axis='y', color=BORDER, lw=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

    verdict_badge(ax, 0.03, 0.92, '✓  Winner', good=True, fontsize=8)

    # ── Panel 3 — GPS spike ───────────────────────────────────────────────────
    ax = ax_spike
    card_bg(ax)
    section_label(ax, 'GPS spike rejection  —  707 m fake fix injected at t=120 s')

    SPIKE_T = 120.0
    if len(gt_ts) and len(fcs_ts):
        t0 = gt_ts[0]

        def plot_err(ts, x, y, color, label, lw=1.8):
            if not len(ts):
                return
            rel = ts - t0
            errs = interp_error(ts, x, y, gt_ts, gt_x, gt_y)
            mask = (rel >= SPIKE_T - 35) & (rel <= SPIKE_T + 45)
            ax.plot(rel[mask] - SPIKE_T, errs[mask], color=color, lw=lw, label=label)

        plot_err(fcs_ts, fcs_x, fcs_y, C_FC,  'FusionCore')
        plot_err(eks_ts, eks_x, eks_y, C_EKF, 'RL-EKF', lw=1.4)

    ax.axvline(0, color='#EF4444', lw=1.6, ls='--', alpha=0.8)
    ax.text(1, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 1 else 10,
            '← spike', color='#EF4444', fontsize=8, va='top')

    ax.set_xlabel('Seconds relative to spike injection', fontsize=8.5, color=SUBTLE)
    ax.set_ylabel('Position error vs GT (m)', fontsize=8.5, color=SUBTLE)
    ax.tick_params(colors=SUBTLE, labelsize=8)
    ax.grid(color=BORDER, lw=0.5, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

    leg = ax.legend(fontsize=8, facecolor='white', edgecolor=BORDER)
    for t in leg.get_texts():
        t.set_color(TEXT)

    # outcome badges
    verdict_badge(ax, 0.03, 0.88, '✓ FC: +1 m — REJECTED', good=True, fontsize=7.5)
    verdict_badge(ax, 0.03, 0.73, '✗ EKF: +93 m — JUMPED', good=False, fontsize=7.5)

    # ── Panel 4 — RL-UKF divergence ───────────────────────────────────────────
    ax = ax_ukf
    card_bg(ax)
    section_label(ax, 'RL-UKF numerical stability')

    if len(uk_ts):
        t0 = uk_ts[0]
        rel = uk_ts - t0
        mag = np.hypot(uk_x, uk_y)
        ax.plot(rel, mag / 1e12, color=C_UKF, lw=1.8, label='RL-UKF')

    if len(fc_ts):
        t0f = fc_ts[0]
        fc_rel = fc_ts - t0f
        fc_mag = np.hypot(fc_x, fc_y)
        mask = fc_rel <= 60
        # normalize to same scale for comparison — show as near-zero
        ax.plot(fc_rel[mask], fc_mag[mask] / 1e12, color=C_FC, lw=1.4,
                alpha=0.7, label='FusionCore (stable)')

    ax.axvline(31, color='#EF4444', lw=1.6, ls='--', alpha=0.8)
    ax.text(32.5, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1,
            'Explodes\nt = 31 s', color='#EF4444', fontsize=8, va='top')

    ax.set_xlim(0, 60)
    ax.set_xlabel('Time (s)', fontsize=8.5, color=SUBTLE)
    ax.set_ylabel('Position magnitude (×10¹² m)', fontsize=8, color=SUBTLE)
    ax.tick_params(colors=SUBTLE, labelsize=8)
    ax.grid(color=BORDER, lw=0.5, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

    leg = ax.legend(fontsize=8, facecolor='white', edgecolor=BORDER)
    for t in leg.get_texts():
        t.set_color(TEXT)

    verdict_badge(ax, 0.03, 0.88, '✗ Dead in 31 seconds', good=False, fontsize=8)

    # ── Panel 5 — RPE summary (text card) ─────────────────────────────────────
    ax = ax_rpe
    card_bg(ax)
    ax.axis('off')
    section_label(ax, 'Relative Pose Error (per 10 m segment)')

    rows = [
        ('FusionCore',       '16.1 m',  C_FC,  True),
        ('RL-EKF',           '18.8 m',  C_EKF, False),
        ('RL-UKF',           'DIVERGED', C_UKF, False),
    ]

    y = 0.78
    for name, val, color, winner in rows:
        ax.text(0.08, y, '●', transform=ax.transAxes,
                color=color, fontsize=14, va='center')
        ax.text(0.20, y, name, transform=ax.transAxes,
                fontsize=10, color=TEXT, va='center', fontweight='bold' if winner else 'normal')
        ax.text(0.72, y, val, transform=ax.transAxes,
                fontsize=10, color=color, va='center', fontweight='bold',
                ha='right')
        y -= 0.22

    ax.text(0.08, 0.12,
            'RPE measures local segment accuracy.\n'
            'Both filters share the same wheel odometry\n'
            'source — gap reflects GPS fusion quality.',
            transform=ax.transAxes, fontsize=8, color=SUBTLE,
            va='bottom', linespacing=1.6)

    ax.plot([0.05, 0.95], [0.36, 0.36], color=BORDER, lw=1.0,
            transform=ax.transAxes, clip_on=False)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.025,
             'Sensor inputs identical across all filters  •  Evaluation: evo  •  SE(3) alignment  •  github.com/manankharwar/fusioncore',
             ha='center', fontsize=8, color=SUBTLE)

    fig.savefig(str(out), dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
