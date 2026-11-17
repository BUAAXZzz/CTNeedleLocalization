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
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk

from match3d_utils import (
    PairingConfig, merge_detections_across_slices, compute_pair_score, NEG_INF,
    read_mha_volume,
    build_score_matrix,          # HU-based O(i,j)
    greedy_match,                # GMM with No-Cross
)


# ---------------------------
# Configuration
# ---------------------------

@dataclass
class GMMConfig(PairingConfig):
    pred2d_name: str = "pred_2d.json"
    n_prior: int = 32
    ct_name: str = ""  # optional override; otherwise detect CT filename
    prior_source: str = "manual"

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

    pred2d_path = os.path.join(case_dir, cfg.pred2d_name)

    if not os.path.exists(pred2d_path):
        raise FileNotFoundError(pred2d_path)
    pred2d = load_pred2d(pred2d_path)
    if cfg.ct_name:
        candidates = [os.path.join(case_dir, cfg.ct_name)]
    else:
        candidates = [os.path.join(case_dir, name)
                      for name in ("ct.nii.gz", "ct.nii", "ct.mha")]
        if pred2d.get("ct_path"):
            candidates.append(os.path.join(case_dir, pred2d["ct_path"]))
    ct_path = next((p for p in candidates if os.path.isfile(p)), None)
    if ct_path is None:
        raise FileNotFoundError("No CT found; tried: " + ", ".join(candidates))
    ct_path = os.path.abspath(ct_path)

    if out_dir is None:
        out_dir = case_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load CT volume
    ct_img, vol_zyx, spacing_xyz, origin_xyz, direction = read_mha_volume(ct_path)

    # Load detections
    T_raw, H_raw = split_tip_handle(pred2d)

    def consolidate(raw):
        selected, raw_indices = [], []
        for index, det in enumerate(raw):
            if float(det.get('score', 0)) < cfg.min_detection_score:
                continue
            d = dict(det)
            xyz = ct_img.TransformContinuousIndexToPhysicalPoint(
                (float(d['x0']), float(d['y0']), float(d['z'])))
            d.update(zip(('x_mm', 'y_mm', 'z_mm'), xyz))
            selected.append(d)
            raw_indices.append(index)
        points, info = merge_detections_across_slices(
            selected, cfg.merge_distance_mm, cfg.merge_z_gap,
            angle_tol_deg=cfg.merge_angle_tol_deg)
        for group in info:
            group['members'] = [raw_indices[i] for i in group['members']]
        return points, info

    T, T_info = consolidate(T_raw)
    H, H_info = consolidate(H_raw)
    diagnostics = {}

    # ---------------------------
    # Step 1: HU-based scoring
    # ---------------------------
    # O(i,j) = μ(i,j) / σ(i,j)
    score_matrix = build_score_matrix(ct_img, T, H, cfg, diagnostics)

    for i, tip in enumerate(T_info):
        for j, handle in enumerate(H_info):
            confidence = float(np.sqrt(tip['score_max'] * handle['score_max']))
            if confidence < cfg.min_pair_confidence:
                if score_matrix[i, j] > NEG_INF:
                    diagnostics['accepted'] -= 1
                    diagnostics['low_pair_confidence'] = diagnostics.get('low_pair_confidence', 0) + 1
                score_matrix[i, j] = NEG_INF

    # ---------------------------
    # Step 2: Greedy matching (UAP-C)
    # ---------------------------
    pairs = greedy_match(score_matrix, T, H, ct_img, n_prior=cfg.n_prior,
                         enable_nocross=not cfg.disable_nocross,
                         nocross_tol_mm=cfg.nocross_tol_mm)
    # Averaging matched endpoints can change geometry; revalidate final pairs.
    valid_pairs = []
    for pair in pairs:
        score = compute_pair_score(ct_img, pair['Tip'], pair['Pin'], spacing_xyz,
                                   cfg, vol_zyx)
        if score <= NEG_INF:
            diagnostics['post_merge_rejected'] = diagnostics.get('post_merge_rejected', 0) + 1
            continue
        pair['score'] = score
        valid_pairs.append(pair)
    pairs = valid_pairs
    used_t = {i for p in pairs for i in p.get('tip_indices', [p['tip_idx']])}
    used_h = {i for p in pairs for i in p.get('pin_indices', [p['pin_idx']])}
    def unmatched(info, used, matrix):
        return [dict(cluster_index=i, raw_indices=g['members'],
                     reason='no_eligible_candidate' if not np.any(matrix[i] > NEG_INF)
                            else 'assignment_conflict_or_count_limit')
                for i, g in enumerate(info) if i not in used]

    # ---------------------------
    # Serialize output
    # ---------------------------
    out = {
        "case_dir": case_dir,
        "ct_path": ct_path,
        "pred2d_path": pred2d_path,
        "spacing_xyz": list(map(float, spacing_xyz.tolist())),
        "origin_xyz": list(map(float, origin_xyz.tolist())),
        "direction": direction.tolist(),
        "coordinate_system": "CT continuous voxel indices; x,y,z,angle_rad",
        "config": asdict(cfg),
        "candidate_diagnostics": diagnostics,
        "endpoint_clusters": {"tip": T_info, "handle": H_info},
        "unmatched": {
            "tip": unmatched(T_info, used_t, score_matrix),
            "handle": unmatched(H_info, used_h, score_matrix.T),
            "low_confidence_tip_raw_indices": [i for i, d in enumerate(T_raw) if float(d.get('score', 0)) < cfg.min_detection_score],
            "low_confidence_handle_raw_indices": [i for i, d in enumerate(H_raw) if float(d.get('score', 0)) < cfg.min_detection_score],
        },
        "counts": {
            "tip_raw": len(T_raw),
            "handle_raw": len(H_raw),
            "tip_merged": len(T),
            "handle_merged": len(H),
            "pairs_final": len(pairs),
        },
        "pairs": [],
        "elapsed_sec": time.time() - t_start
    }

    for p in pairs:
        # Preserve published indices as indices into raw class-specific detections.
        ti = p.get('tip_indices', [p['tip_idx']])
        hi = p.get('pin_indices', [p['pin_idx']])
        p['tip_indices'] = sorted({j for i in ti for j in T_info[i]['members']})
        p['pin_indices'] = sorted({j for i in hi for j in H_info[i]['members']})
        p['tip_idx'] = T_info[p['tip_idx']]['members'][0]
        p['pin_idx'] = H_info[p['pin_idx']]['members'][0]
        out["pairs"].append({
            "Tip": list(map(float, p["Tip"])),
            "Handle": list(map(float, p["Pin"])),
            "score": float(p["score"]),
            "tip_idx": int(p["tip_idx"]),
            "handle_idx": int(p["pin_idx"]),
            "tip_indices": list(map(int, p.get("tip_indices", [p["tip_idx"]]))),
            "handle_indices": list(map(int, p.get("pin_indices", [p["pin_idx"]])))
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
    ap.add_argument("--ct_name", default="", help="CT filename override (default: auto-detect)")
    ap.add_argument("--priors_json", default="", help="JSON mapping case directory names to needle counts")
    ap.add_argument("--prior_source", default="manual", help="Description of where needle counts came from")
    ap.add_argument("--disable_nocross", action="store_true")
    ap.add_argument("--nocross_tol_mm", type=float, default=2.0)
    ap.add_argument("--out_name", default="pred_3d.json")
    ap.add_argument('--case_config_json', default='', help='Per-case overrides of PairingConfig fields; lengths in mm')
    for name, default in vars(PairingConfig()).items():
        ap.add_argument('--' + name, type=float if default is None else type(default), default=default)
    args = ap.parse_args()

    cfg = GMMConfig(
        pred2d_name=args.pred2d_name,
        n_prior=args.n_prior,
        ct_name=args.ct_name,
        prior_source=args.prior_source,
        disable_nocross=args.disable_nocross,
        nocross_tol_mm=args.nocross_tol_mm,
        **{name: getattr(args, name) for name in vars(PairingConfig())},
        out_name=args.out_name
    )

    cases = sorted([
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
    ])
    priors = None
    if args.priors_json:
        with open(args.priors_json, encoding="utf-8") as f:
            priors = json.load(f)
        for case in cases:
            count = priors.get(case)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"Missing or invalid positive needle count for {case}")

    overrides = {}
    if args.case_config_json:
        with open(args.case_config_json, encoding='utf-8') as f:
            overrides = json.load(f)
        if set(overrides) - set(cases):
            raise ValueError('Unknown cases in case_config_json')
        for case, values in overrides.items():
            if set(values) - set(vars(PairingConfig())):
                raise ValueError(f'Unknown pairing fields for {case}')
            replace(cfg, **values)  # validate before processing any cases

    ok, err = 0, 0
    for c in cases:
        case_dir = os.path.join(args.root, c)
        out_dir = None
        if args.out_root:
            out_dir = os.path.join(args.out_root, c)

        try:
            case_cfg = replace(cfg, n_prior=priors[c]) if priors is not None else cfg
            case_cfg = replace(case_cfg, **overrides.get(c, {}))
            res = run_gmm_one_case(case_dir, case_cfg, out_dir)
            ok += 1
            print(f"[OK] {c}: pairs={res['counts']['pairs_final']}  "
                  f"time={res['elapsed_sec']:.3f}s")
        except Exception as e:
            err += 1
            print(f"[ERR] {c}: {repr(e)}")

    print(f"Finished. OK={ok}, ERR={err}")
    if err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
