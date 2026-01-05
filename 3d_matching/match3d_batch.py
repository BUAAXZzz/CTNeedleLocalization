# -----------------------------------------------------------------------------
# Copyright (c) 2026 Zhuo Xiao
# All rights reserved.
#
# This source code is licensed under the MIT License.
# You may obtain a copy of the License at:
#   https://opensource.org/licenses/MIT
#
# Description:
#   This file implements the 3D needle matching and association logic
#   (tip–handle matching, cross-slice merging, and greedy assignment),
#   with all network forward/inference components removed.
#
# Notes:
#   - This code is intended for research and academic use.
#   - If you use this code in published work, please cite the corresponding paper.
# -----------------------------------------------------------------------------

# match3d_batch.py
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from match3d_utils import (
    read_mha_volume,
    merge_detections_across_slices,
    build_score_matrix,
    greedy_match,
    greedy_match_wo_nocross,
)


@dataclass
class Match3DConfig:
    # Input
    pred2d_name: str = "pred_2d.json"

    # 3D merge config
    enable_3d_merge: bool = True
    merge_cross_slice_tol_mm: float = 6.0
    merge_max_z_gap_slices: int = 2
    merge_use_angle: bool = True
    merge_angle_tol_deg: float = 5.0

    # Matching config
    n_prior: int = 32
    disable_nocross: bool = False
    nocross_tol_mm: float = 2.0

    # Output
    out_name: str = "pred_3d.json"


def _safe_makedirs(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_pred2d(pred2d_path: str) -> Dict[str, Any]:
    with open(pred2d_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_tip_pin_from_pred2d(pred2d: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Expect pred2d["slices"] list. Each slice has "dets".
    Each det should include: x0,y0,z,angle,cls,score (x_mm/y_mm/z_mm optional).
    By default:
      cls == 1 -> tip
      cls == 0 -> pin/handle
    """
    all_tip: List[Dict[str, Any]] = []
    all_pin: List[Dict[str, Any]] = []

    slices = pred2d.get("slices", [])
    for s in slices:
        dets = s.get("dets", [])
        for d in dets:
            cls = int(d.get("cls", -1))
            if cls == 1:
                all_tip.append(d)
            elif cls == 0:
                all_pin.append(d)

    return all_tip, all_pin


def _run_match_one_case(case_dir: str, cfg: Match3DConfig, out_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    case_dir must contain:
      - ct.mha
      - pred_2d.json (or cfg.pred2d_name)
    """
    t0 = time.time()

    ct_path = os.path.join(case_dir, "ct.mha")
    pred2d_path = os.path.join(case_dir, cfg.pred2d_name)
    if not os.path.exists(pred2d_path):
        raise FileNotFoundError(f"pred2d not found: {pred2d_path}")
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"ct.mha not found: {ct_path}")

    if out_dir is None:
        out_dir = case_dir
    _safe_makedirs(out_dir)

    ct_img, vol_zyx, spacing_xyz, origin_xyz, direction_3x3 = read_mha_volume(ct_path)
    pred2d = _load_pred2d(pred2d_path)
    all_tip_dets, all_pin_dets = _collect_tip_pin_from_pred2d(pred2d)

    if cfg.enable_3d_merge:
        tip_data, tip_merge_info = merge_detections_across_slices(
            all_tip_dets,
            cross_slice_tol_mm=cfg.merge_cross_slice_tol_mm,
            max_z_gap_slices=cfg.merge_max_z_gap_slices,
            use_angle=cfg.merge_use_angle,
            angle_tol_deg=cfg.merge_angle_tol_deg,
        )
        pin_data, pin_merge_info = merge_detections_across_slices(
            all_pin_dets,
            cross_slice_tol_mm=cfg.merge_cross_slice_tol_mm,
            max_z_gap_slices=cfg.merge_max_z_gap_slices,
            use_angle=cfg.merge_use_angle,
            angle_tol_deg=cfg.merge_angle_tol_deg,
        )
    else:
        # Use raw 2D dets as 3D points: [x0, y0, z, angle]
        tip_data = [[float(d["x0"]), float(d["y0"]), float(d["z"]), float(d.get("angle", 0.0))] for d in all_tip_dets]
        pin_data = [[float(d["x0"]), float(d["y0"]), float(d["z"]), float(d.get("angle", 0.0))] for d in all_pin_dets]
        tip_merge_info, pin_merge_info = [], []

    # Build score matrix and match
    score_mat = build_score_matrix(ct_img, tip_data, pin_data)

    if cfg.disable_nocross:
        pairs = greedy_match_wo_nocross(score_mat, tip_data, pin_data, n_prior=cfg.n_prior)
    else:
        pairs = greedy_match(
            score_mat,
            tip_data,
            pin_data,
            ct_img,
            n_prior=cfg.n_prior,
            enable_nocross=True,
            nocross_tol_mm=cfg.nocross_tol_mm,
        )

    # Serialize results
    out = {
        "case_dir": case_dir,
        "ct_path": ct_path,
        "pred2d_path": pred2d_path,
        "spacing_xyz": list(map(float, spacing_xyz.tolist())),
        "shape_zyx": [int(vol_zyx.shape[0]), int(vol_zyx.shape[1]), int(vol_zyx.shape[2])],
        "config": asdict(cfg),
        "counts": {
            "tip_raw": int(len(all_tip_dets)),
            "pin_raw": int(len(all_pin_dets)),
            "tip_merged": int(len(tip_data)),
            "pin_merged": int(len(pin_data)),
            "pairs": int(len(pairs)),
        },
        "tip_merge_info": tip_merge_info,
        "pin_merge_info": pin_merge_info,
        "pairs": [],
        "elapsed_sec": float(time.time() - t0),
    }

    # Keep only necessary fields (Tip/Pin are numpy arrays)
    for p in pairs:
        tip = np.asarray(p["Tip"], dtype=float).reshape(-1).tolist()
        pin = np.asarray(p["Pin"], dtype=float).reshape(-1).tolist()
        out["pairs"].append({
            "Tip": tip,          # [x,y,z,angle]
            "Pin": pin,          # [x,y,z,angle]
            "score": float(p.get("score", 0.0)),
            "tip_idx": int(p.get("tip_idx", -1)),
            "pin_idx": int(p.get("pin_idx", -1)),
        })

    out_path = os.path.join(out_dir, cfg.out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root dir containing multiple case subfolders.")
    ap.add_argument("--out_root", default="", help="If set, write each case output to out_root/<case>/pred_3d.json")
    ap.add_argument("--pred2d_name", default="pred_2d.json", help="2D prediction json filename under each case dir.")
    ap.add_argument("--n_prior", type=int, default=32)
    ap.add_argument("--enable_3d_merge", action="store_true")
    ap.add_argument("--disable_3d_merge", action="store_true")
    ap.add_argument("--disable_nocross", action="store_true")
    ap.add_argument("--nocross_tol_mm", type=float, default=2.0)

    ap.add_argument("--merge_cross_slice_tol_mm", type=float, default=6.0)
    ap.add_argument("--merge_max_z_gap_slices", type=int, default=2)
    ap.add_argument("--merge_use_angle", action="store_true")
    ap.add_argument("--merge_angle_tol_deg", type=float, default=5.0)

    ap.add_argument("--out_name", default="pred_3d.json")
    args = ap.parse_args()

    cfg = Match3DConfig(
        pred2d_name=args.pred2d_name,
        enable_3d_merge=(args.enable_3d_merge and (not args.disable_3d_merge)),
        merge_cross_slice_tol_mm=args.merge_cross_slice_tol_mm,
        merge_max_z_gap_slices=args.merge_max_z_gap_slices,
        merge_use_angle=args.merge_use_angle,
        merge_angle_tol_deg=args.merge_angle_tol_deg,
        n_prior=args.n_prior,
        disable_nocross=args.disable_nocross,
        nocross_tol_mm=args.nocross_tol_mm,
        out_name=args.out_name,
    )

    root = args.root
    case_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

    ok, bad = 0, 0
    for case in case_names:
        case_dir = os.path.join(root, case)
        out_dir = None
        if args.out_root:
            out_dir = os.path.join(args.out_root, case)
            os.makedirs(out_dir, exist_ok=True)

        try:
            res = _run_match_one_case(case_dir, cfg, out_dir=out_dir)
            ok += 1
            print(f"[OK] {case}: pairs={res['counts']['pairs']}  time={res['elapsed_sec']:.3f}s")
        except Exception as e:
            bad += 1
            print(f"[ERR] {case}: {repr(e)}")

    print(f"Done. OK={ok}, ERR={bad}")


if __name__ == "__main__":
    main()
