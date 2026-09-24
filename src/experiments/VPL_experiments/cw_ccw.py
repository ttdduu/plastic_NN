"""
VPL CW/CCW orientation discrimination with a localized window mask.

Builds directly on the un-masked VPL pipeline in VPL_orientation_DCNN/Train_Test/:
imports `pretrain_custom`, `posttrain_custom`, and the internal
`_make_gabor_pair`, and injects a `stim_transform` that zeroes out every pixel
of the post-fisheye image except a square window of configurable size and
location.

Train: ONE window position.
Test:  MULTIPLE window positions. Accuracy is reported split by
       "test window == train window" vs "!=".

Run:
    python -m src.experiments.VPL_experiments.cw_ccw --phase train
    python -m src.experiments.VPL_experiments.cw_ccw --phase test
    python -m src.experiments.VPL_experiments.cw_ccw --phase all
"""

from __future__ import annotations
import os
import sys
import glob
import re
import argparse
import csv
from typing import List, Tuple, Optional

_this_file = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_this_file, '..', '..', '..'))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, 'VPL_orientation_DCNN', 'model'))
sys.path.insert(0, os.path.join(_repo_root, 'VPL_orientation_DCNN', 'Train_Test'))

import torch
import numpy as np

from pretrain import (
    pretrain_custom,
    _make_gabor_pair,
    _build_fisheye_and_normalize,
    BACKBONE_WEIGHTS,
)
from posttrain import posttrain_custom
from SiameseCornet import build_siamese_from_pretrained_siamese


# ---------------------------------------------------------------------------
# Settings — edit and re-run, or override via CLI
# ---------------------------------------------------------------------------
# Window = (cy, cx, size). Coordinates are pixel positions in the post-fisheye
# image (156x156 for size=256 + the K=-7, rfov=30 fisheye). Image center: (78, 78).
Window = Tuple[int, int, int]

# Train: 14x14 square, centered 20px to the right of the image center.
TRAIN_WINDOW: Window = (78, 98, 14)

# Test: include the train position (→ "matches") plus a few offsets.
TEST_WINDOWS: List[Window] = [
    (78, 98, 14),   # = train window (matches)
    (78, 58, 14),   # 20px LEFT of center
    (58, 78, 14),   # 20px ABOVE center
    (98, 78, 14),   # 20px BELOW center
    (78, 78, 14),   # at center
]

MODEL_NAME    = 'best_nonoverfit_model-qk4_cw_ccw_mask'
SAVE_DIR_BASE = '/home/ttdduu/repos/VPL_orientation_DCNN_results/cw_ccw_window_mask'

REF_ANGLE          = 35
TARGET_SEP         = 1
EPOCHS_PRE         = 300
EPOCHS_POST        = 150
LR                 = 1e-2
SPATIAL_LOSS_ALPHA = 0.0

# Per test window — must be even (one CW + one CCW per pair, so 2 trials per call).
N_TEST_TRIALS_PER_WINDOW = 200


# ---------------------------------------------------------------------------
# Window mask
# ---------------------------------------------------------------------------

class WindowMask:
    """
    Zero out every pixel outside a square window of side `size` centered at
    (cy, cx). Operates on (B, C, H, W) tensors; meant to be applied as the
    last step of the input pipeline (post-fisheye, post-normalize).

    `mask_value=0.0` means tensor-zero (visually ImageNet-mean gray, not pure
    black — see design note in handoff doc).
    """
    def __init__(self, cy: int, cx: int, size: int, mask_value: float = 0.0):
        self.cy = int(cy)
        self.cx = int(cx)
        self.size = int(size)
        self.mask_value = float(mask_value)

    def __repr__(self):
        return f"WindowMask(cy={self.cy}, cx={self.cx}, size={self.size})"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        half = self.size // 2
        y0 = self.cy - half
        x0 = self.cx - half
        y1 = y0 + self.size
        x1 = x0 + self.size
        y0c, y1c = max(0, y0), min(H, y1)
        x0c, x1c = max(0, x0), min(W, x1)
        out = torch.full_like(x, self.mask_value)
        if y1c > y0c and x1c > x0c:
            out[..., y0c:y1c, x0c:x1c] = x[..., y0c:y1c, x0c:x1c]
        return out


# ---------------------------------------------------------------------------
# Weights resolver (mirrors Train_Test._resolve_post_weights)
# ---------------------------------------------------------------------------

def _resolve_weights(save_dir: str, prefer: str = 'post') -> str:
    """
    Pick the right checkpoint inside `save_dir`.

      prefer='post': prefer VPL_{best_epoch}_model.pth (the best-CE-avg
                     snapshot from posttrain) → post_model.pth → pre_model.pth.
                     The VPL_*.pth is preferred because post_model.pth can be
                     NaN if posttrain diverged late.
      prefer='pre' : pre_model.pth only.
    """
    if prefer == 'pre':
        path = os.path.join(save_dir, 'pre_model.pth')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"pre_model.pth not found in {save_dir}")
        return path

    vpl_candidates = glob.glob(os.path.join(save_dir, 'VPL_*_model.pth'))
    if vpl_candidates:
        def _epoch(p):
            m = re.search(r'VPL_(\d+)_model', os.path.basename(p))
            return int(m.group(1)) if m else -1
        vpl_candidates.sort(key=_epoch, reverse=True)
        return vpl_candidates[0]

    post = os.path.join(save_dir, 'post_model.pth')
    if os.path.isfile(post):
        return post
    pre = os.path.join(save_dir, 'pre_model.pth')
    if os.path.isfile(pre):
        return pre
    raise FileNotFoundError(
        f"No checkpoint found in {save_dir} (looked for VPL_*_model.pth, "
        f"post_model.pth, pre_model.pth)."
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_cw_ccw(
    train_window: Window = TRAIN_WINDOW,
    ref_angle: int = REF_ANGLE,
    rep_id: int = 0,
    epochs_pre: int = EPOCHS_PRE,
    epochs_post: int = EPOCHS_POST,
    lr: float = LR,
    spatial_loss_alpha: float = SPATIAL_LOSS_ALPHA,
    backbone_weights: str = BACKBONE_WEIGHTS,
    model_name: str = MODEL_NAME,
    save_dir_base: str = SAVE_DIR_BASE,
    run_pretrain: bool = True,
    run_posttrain: bool = True,
):
    """
    Two-phase Siamese training, with `train_window` masked as the last step of
    the input pipeline (post-fisheye, post-normalize).
    """
    mask = WindowMask(*train_window)
    print(f"[train_cw_ccw] train_window = {mask}")
    print(f"[train_cw_ccw] save base    = {save_dir_base}/{model_name}/ref{ref_angle}/rep{rep_id}")

    if run_pretrain:
        pretrain_custom(
            ref_angle=ref_angle,
            epochs=epochs_pre,
            learning_rate=lr,
            backbone_weights=backbone_weights,
            model_name=model_name,
            save_dir_base=save_dir_base,
            spatial_loss_alpha=spatial_loss_alpha,
            rep_id=rep_id,
            stim_transform=mask,
        )

    if run_posttrain:
        posttrain_custom(
            ref_angle=ref_angle,
            epochs=epochs_post,
            learning_rate=lr,
            model_name=model_name,
            save_dir_base=save_dir_base,
            spatial_loss_alpha=spatial_loss_alpha,
            rep_id=rep_id,
            stim_transform=mask,
        )

    # Persist train_window so test_cw_ccw can later determine which test
    # positions "match" without depending on a constant in this file.
    save_dir = os.path.join(save_dir_base, model_name, f'ref{ref_angle}', f'rep{rep_id}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'train_window.txt'), 'w') as f:
        f.write(f"{train_window[0]} {train_window[1]} {train_window[2]}\n")
    print(f"[train_cw_ccw] wrote train_window → {save_dir}/train_window.txt")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_cw_ccw(
    test_windows: List[Window] = TEST_WINDOWS,
    train_window: Optional[Window] = None,
    ref_angle: int = REF_ANGLE,
    rep_id: int = 0,
    target_sep: int = TARGET_SEP,
    n_trials_per_window: int = N_TEST_TRIALS_PER_WINDOW,
    contrast: float = 1.0,
    p_noise: Optional[float] = None,
    model_name: str = MODEL_NAME,
    save_dir_base: str = SAVE_DIR_BASE,
    prefer: str = 'post',
    weights_path: Optional[str] = None,
    seed: int = 12345,
):
    """
    For each test window, run `n_trials_per_window` 2AFC trials through the
    trained Siamese model and record predictions. Aggregates accuracy split by
    whether the test window equals the train window.

    Writes:
        cw_ccw_test_log.csv      — per-trial log
        cw_ccw_test_summary.txt  — per-window + split accuracies
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.join(save_dir_base, model_name, f'ref{ref_angle}', f'rep{rep_id}')
    if weights_path is None:
        weights_path = _resolve_weights(save_dir, prefer=prefer)

    if train_window is None:
        tw_path = os.path.join(save_dir, 'train_window.txt')
        if not os.path.isfile(tw_path):
            raise FileNotFoundError(
                f"train_window.txt not found in {save_dir}. Either run "
                f"train_cw_ccw() first or pass train_window= explicitly."
            )
        with open(tw_path) as f:
            train_window = tuple(int(v) for v in f.read().split())  # type: ignore

    print(f"[test_cw_ccw] device       : {device}")
    print(f"[test_cw_ccw] weights      : {weights_path}")
    print(f"[test_cw_ccw] ref_angle    : {ref_angle}  (target_sep = {target_sep}°)")
    print(f"[test_cw_ccw] train_window : {train_window}")
    print(f"[test_cw_ccw] test_windows : {test_windows}")
    print(f"[test_cw_ccw] trials/win   : {n_trials_per_window}  (contrast={contrast}, p={p_noise})")

    siamese = build_siamese_from_pretrained_siamese(weights_path, device)
    siamese.eval()
    fisheye, normalize = _build_fisheye_and_normalize()

    if n_trials_per_window % 2 != 0:
        raise ValueError("n_trials_per_window must be even (each pair yields one CW + one CCW).")
    n_pairs = n_trials_per_window // 2

    log_rows: List[dict] = []
    summary = {}

    for win_idx, window in enumerate(test_windows):
        mask = WindowMask(*window)
        matches = (tuple(window) == tuple(train_window))
        n_correct = 0
        n_total = 0
        for _ in range(n_pairs):
            input1, input2, label = _make_gabor_pair(
                ref_angle, target_sep, contrast, p_noise,
                gabor_size=256, sf=40, sigma=50, SD_1=10/255, SD_2=15/100,
                normalize=normalize, fisheye=fisheye, device=device,
                stim_transform=mask,
            )
            with torch.no_grad():
                output = siamese(input1, input2).squeeze(-1)   # (2,)
            # BCEWithLogitsLoss convention: positive logit ⇒ predicted label=1 ⇒ CCW.
            predicted = (output > 0).long()
            label_int = label.long()
            correct = (predicted == label_int)
            for i in range(2):
                log_rows.append({
                    'win_idx':           win_idx,
                    'win_cy':            window[0],
                    'win_cx':            window[1],
                    'win_size':          window[2],
                    'win_matches_train': int(matches),
                    'label':             int(label_int[i].item()),
                    'predicted':         int(predicted[i].item()),
                    'logit':             float(output[i].item()),
                    'correct':           int(correct[i].item()),
                })
                n_correct += int(correct[i].item())
                n_total += 1
        acc_w = n_correct / n_total if n_total else float('nan')
        summary[win_idx] = {
            'window':        tuple(window),
            'matches_train': matches,
            'n_correct':     n_correct,
            'n_total':       n_total,
            'accuracy':      acc_w,
        }
        print(f"  win {win_idx} {tuple(window)} matches={matches}: "
              f"acc = {acc_w:.4f}  ({n_correct}/{n_total})")

    same_rows = [r for r in log_rows if r['win_matches_train']]
    diff_rows = [r for r in log_rows if not r['win_matches_train']]
    n_same_correct = sum(r['correct'] for r in same_rows)
    n_diff_correct = sum(r['correct'] for r in diff_rows)
    acc_same = n_same_correct / len(same_rows) if same_rows else float('nan')
    acc_diff = n_diff_correct / len(diff_rows) if diff_rows else float('nan')

    print()
    print(f"[test_cw_ccw] acc (test_window == train_window): {acc_same:.4f}  "
          f"({n_same_correct}/{len(same_rows)})")
    print(f"[test_cw_ccw] acc (test_window != train_window): {acc_diff:.4f}  "
          f"({n_diff_correct}/{len(diff_rows)})")

    csv_path = os.path.join(save_dir, 'cw_ccw_test_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[test_cw_ccw] per-trial log → {csv_path}")

    summary_path = os.path.join(save_dir, 'cw_ccw_test_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"weights:             {weights_path}\n")
        f.write(f"train_window:        {tuple(train_window)}\n")
        f.write(f"test_windows:        {test_windows}\n")
        f.write(f"contrast:            {contrast}\n")
        f.write(f"p_noise:             {p_noise}\n")
        f.write(f"n_trials_per_window: {n_trials_per_window}\n")
        f.write(f"\nper-window accuracy:\n")
        for win_idx, s in summary.items():
            f.write(f"  win {win_idx} {s['window']} matches_train={s['matches_train']}: "
                    f"acc={s['accuracy']:.4f}  ({s['n_correct']}/{s['n_total']})\n")
        f.write(f"\nsplit accuracy:\n")
        f.write(f"  acc (test_window == train_window): {acc_same:.4f}  "
                f"({n_same_correct}/{len(same_rows)})\n")
        f.write(f"  acc (test_window != train_window): {acc_diff:.4f}  "
                f"({n_diff_correct}/{len(diff_rows)})\n")
    print(f"[test_cw_ccw] summary       → {summary_path}")

    return {
        'acc_same_window': acc_same,
        'acc_diff_window': acc_diff,
        'per_window':      summary,
        'log_rows':        log_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CW/CCW orientation discrimination with localized window mask')
    parser.add_argument('--phase', choices=['train', 'test', 'all'], default='all')
    parser.add_argument('--ref_angle', type=int, default=REF_ANGLE)
    parser.add_argument('--rep_id', type=int, default=0)
    parser.add_argument('--epochs_pre', type=int, default=EPOCHS_PRE)
    parser.add_argument('--epochs_post', type=int, default=EPOCHS_POST)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--spatial_loss_alpha', type=float, default=SPATIAL_LOSS_ALPHA)
    parser.add_argument('--prefer', choices=['post', 'pre'], default='post',
                        help="Which checkpoint to test: 'post' picks the best "
                             "VPL_*_model.pth (fallback post_model.pth); "
                             "'pre' picks pre_model.pth.")
    parser.add_argument('--n_trials_per_window', type=int, default=N_TEST_TRIALS_PER_WINDOW)
    parser.add_argument('--no_posttrain', action='store_true',
                        help='Only run pretrain (skip posttrain).')
    args = parser.parse_args()

    if args.phase in ('train', 'all'):
        train_cw_ccw(
            ref_angle=args.ref_angle,
            rep_id=args.rep_id,
            epochs_pre=args.epochs_pre,
            epochs_post=args.epochs_post,
            lr=args.lr,
            spatial_loss_alpha=args.spatial_loss_alpha,
            run_posttrain=not args.no_posttrain,
        )
    if args.phase in ('test', 'all'):
        test_cw_ccw(
            ref_angle=args.ref_angle,
            rep_id=args.rep_id,
            n_trials_per_window=args.n_trials_per_window,
            prefer=args.prefer,
        )


if __name__ == '__main__':
    main()
