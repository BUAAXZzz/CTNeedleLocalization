# -----------------------------------------------------------------------------
# Copyright (c) 2026 Zhuo Xiao
# All rights reserved.
#
# Author: Zhuo Xiao
#
# This source code is licensed under the MIT License.
# You may obtain a copy of the License at:
#   https://opensource.org/licenses/MIT
#
# Description:
#   Implementation of the Greedy Matching and Merging (GMM) algorithm for
#   3D needle reconstruction, as described in:
#
#   "Multi-needle Localization for Pelvic Seed Implant Brachytherapy
#    based on Tip-handle Detection and Matching"
#
#   This file implements the Unbalanced Assignment Problem with Constraints (UAP-C)
#   using HU-based scoring, geometric constraints, and greedy optimization.
#
#   NOTE:
#   - No neural network forward or inference is included.
#   - This module operates purely on 2D detection results and CT volumes.
# -----------------------------------------------------------------------------

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk

from match3d_utils import (
    read_mha_volume,
    build_score_matrix,          # HU-based O(i,j)
    greedy_match,                # GMM with No-Cross
    greedy_match_wo_nocross,     # GMM without No-Cross (ablation)
    merge_duplicate_paths        # HU-weighted centroid merge
)


# ---------------------------
# Configuration
# ---------------------------

@dataclass
class GMMConfig:
    pred2d_name: str = "pred_2d.json"
    n_prior: int = 32

    # Constraints
    disable_nocross: bool = False
    nocross_tol_mm: float = 2.0

    # Output
    out_name: str = "pred_3d.json"


# ---------------------------
# I/O utilities
# ---------------------------

def load_pred2d(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_tip_handle(pred2d: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Split detections into tip set T and handle set H.

    Convention:
      cls == 1 -> tip
      cls == 0 -> handle
    """
    T, H = [], []
    for s in pred2d.get("slices", []):
        for d in s.get("dets", []):
            if int(d.get("cls", -1)) == 1:
                T.append(d)
            elif int(d.get("cls", -1)) == 0:
                H.append(d)
    return T, H


def to_xyz_angle(dets: List[Dict]) -> List[List[float]]:
    """
    Convert raw dict detections to [x, y, z, angle]
    """
    out = []
    for d in dets:
        out.append([
            float(d["x0"]),
            float(d["y0"]),
            float(d["z"]),
            float(d.get("angle", 0.0))
        ])
    return out


# ---------------------------
# Core GMM pipeline
# ---------------------------

def run_gmm_one_case(
    case_dir: str,
    cfg: GMMConfig,
    out_dir: str = None
) -> Dict:

    t_start = time.time()

    ct_path = os.path.join(case_dir, "ct.mha")
    pred2d_path = os.path.join(case_dir, cfg.pred2d_name)

    if not os.path.exists(ct_path):
        raise FileNotFoundError(ct_path)
    if not os.path.exists(pred2d_path):
        raise FileNotFoundError(pred2d_path)

    if out_dir is None:
        out_dir = case_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load CT volume
    ct_img, vol_zyx, spacing_xyz, origin_xyz, direction = read_mha_volume(ct_path)

    # Load detections
    pred2d = load_pred2d(pred2d_path)
    T_raw, H_raw = split_tip_handle(pred2d)

    T = to_xyz_angle(T_raw)
    H = to_xyz_angle(H_raw)

    # ---------------------------
    # Step 1: HU-based scoring
    # ---------------------------
    # O(i,j) = μ(i,j) / σ(i,j)
    score_matrix = build_score_matrix(ct_img, T, H)

    # ---------------------------
    # Step 2: Greedy matching (UAP-C)
    # ---------------------------
    if cfg.disable_nocross:
        pairs = greedy_match_wo_nocross(
            score_matrix, T, H, n_prior=cfg.n_prior
        )
    else:
        pairs = greedy_match(
            score_matrix,
            T,
            H,
            ct_img,
            n_prior=cfg.n_prior,
            nocross_tol_mm=cfg.nocross_tol_mm
        )

    # ---------------------------
    # Step 3: Duplicate path merging (only if |pairs| > N_prior)
    # ---------------------------
    if len(pairs) > cfg.n_prior:
        pairs = merge_duplicate_paths(
            pairs,
            ct_img,
            target_n=cfg.n_prior
        )

    # ---------------------------
    # Serialize output
    # ---------------------------
    out = {
        "case_dir": case_dir,
        "ct_path": ct_path,
        "pred2d_path": pred2d_path,
        "spacing_xyz": list(map(float, spacing_xyz.tolist())),
        "config": asdict(cfg),
        "counts": {
            "tip_raw": len(T_raw),
            "handle_raw": len(H_raw),
            "pairs_final": len(pairs),
        },
        "pairs": [],
        "elapsed_sec": time.time() - t_start
    }

    for p in pairs:
        out["pairs"].append({
            "Tip": list(map(float, p["Tip"])),
            "Handle": list(map(float, p["Pin"])),
            "score": float(p["score"]),
            "tip_idx": int(p["tip_idx"]),
            "handle_idx": int(p["pin_idx"])
        })

    out_path = os.path.join(out_dir, cfg.out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


# ---------------------------
# Batch interface
# ---------------------------

def main():
    ap = argparse.ArgumentParser("GMM-based 3D Needle Matching")
    ap.add_argument("--root", required=True, help="Root directory of cases")
    ap.add_argument("--out_root", default="", help="Optional output root")
    ap.add_argument("--pred2d_name", default="pred_2d.json")
    ap.add_argument("--n_prior", type=int, default=32)
    ap.add_argument("--disable_nocross", action="store_true")
    ap.add_argument("--nocross_tol_mm", type=float, default=2.0)
    ap.add_argument("--out_name", default="pred_3d.json")
    args = ap.parse_args()

    cfg = GMMConfig(
        pred2d_name=args.pred2d_name,
        n_prior=args.n_prior,
        disable_nocross=args.disable_nocross,
        nocross_tol_mm=args.nocross_tol_mm,
        out_name=args.out_name
    )

    cases = sorted([
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
    ])

    ok, err = 0, 0
    for c in cases:
        case_dir = os.path.join(args.root, c)
        out_dir = None
        if args.out_root:
            out_dir = os.path.join(args.out_root, c)

        try:
            res = run_gmm_one_case(case_dir, cfg, out_dir)
            ok += 1
            print(f"[OK] {c}: pairs={res['counts']['pairs_final']}  "
                  f"time={res['elapsed_sec']:.3f}s")
        except Exception as e:
            err += 1
            print(f"[ERR] {c}: {repr(e)}")

    print(f"Finished. OK={ok}, ERR={err}")


if __name__ == "__main__":
    main()
