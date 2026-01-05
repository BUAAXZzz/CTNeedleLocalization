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

# match3d_utils.py
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk


# -----------------------------
# Constants (tune if needed)
# -----------------------------
NEEDLE_LENGTH_PRIOR_MM = 210.0
NEEDLE_LENGTH_TOL_MM = 50.0
ANGLE_DIFF_TOL_DEG = 20.0
MERGE_DIST_TOL_MM = 2.5
NEG_INF = -1e9


# -----------------------------
# Union-Find for cross-slice merging
# -----------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# -----------------------------
# CT I/O
# -----------------------------
def read_mha_volume(mha_path: str):
    img = sitk.ReadImage(mha_path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)      # (sx, sy, sz)
    origin = np.array(img.GetOrigin(), dtype=np.float64)        # (ox, oy, oz)
    direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    return img, arr, spacing, origin, direction


# -----------------------------
# Geometry helpers
# -----------------------------
def calculate_physical_distance_mm(p1_xyz, p2_xyz, spacing_xyz) -> float:
    p1 = np.asarray(p1_xyz, dtype=np.float64).reshape(-1)[:3]
    p2 = np.asarray(p2_xyz, dtype=np.float64).reshape(-1)[:3]
    sp = np.asarray(spacing_xyz, dtype=np.float64).reshape(-1)[:3]
    d = (p2 - p1) * sp
    return float(np.sqrt(np.dot(d, d)))


def cal_angle_diff_deg(angle_tip_rad: float, angle_handle_rad: float) -> float:
    """
    Keep the same behavior as your original code:
    angle_tip is shifted by pi before comparison.
    """
    a_tip = float(angle_tip_rad) - math.pi
    a_hdl = float(angle_handle_rad)
    diff = abs(a_tip - a_hdl)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return float(diff * 180.0 / math.pi)


def _segment_segment_distance(p1, p2, q1, q2, eps=1e-12) -> float:
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    D = a * c - b * b

    if a < eps and c < eps:
        return float(np.linalg.norm(p1 - q1))
    if a < eps:
        t = np.clip(e / c if c > eps else 0.0, 0.0, 1.0)
        proj = q1 + t * v
        return float(np.linalg.norm(p1 - proj))
    if c < eps:
        s = np.clip(-d / a if a > eps else 0.0, 0.0, 1.0)
        proj = p1 + s * u
        return float(np.linalg.norm(proj - q1))

    if D < eps:
        s = 0.0
        t = np.clip(e / c, 0.0, 1.0)
    else:
        s = (b * e - c * d) / D
        t = (a * e - b * d) / D
        s = np.clip(s, 0.0, 1.0)
        t = np.clip(t, 0.0, 1.0)
        t = np.clip((b * s + e) / c, 0.0, 1.0)
        s = np.clip((b * t - d) / a, 0.0, 1.0)

    cp_p = p1 + s * u
    cp_q = q1 + t * v
    return float(np.linalg.norm(cp_p - cp_q))


def line_too_close_mm(spacing_xyz, p1_start, p1_end, p2_start, p2_end, tol_mm=2.0) -> bool:
    sp = np.asarray(spacing_xyz, dtype=np.float64).reshape(-1)[:3]

    a1 = np.asarray(p1_start, dtype=np.float64).reshape(-1)[:3] * sp
    b1 = np.asarray(p1_end, dtype=np.float64).reshape(-1)[:3] * sp
    a2 = np.asarray(p2_start, dtype=np.float64).reshape(-1)[:3] * sp
    b2 = np.asarray(p2_end, dtype=np.float64).reshape(-1)[:3] * sp

    min1 = np.minimum(a1, b1) - tol_mm
    max1 = np.maximum(a1, b1) + tol_mm
    min2 = np.minimum(a2, b2) - tol_mm
    max2 = np.maximum(a2, b2) + tol_mm
    if np.any(max1 < min2) or np.any(max2 < min1):
        return False

    d = _segment_segment_distance(a1, b1, a2, b2)
    return d <= float(tol_mm)


def check_intersections(spacing_xyz, pairs: List[Dict[str, Any]], tol_mm=2.0) -> bool:
    if not pairs or len(pairs) < 2:
        return False

    segs = []
    for p in pairs:
        if "Tip" not in p or "Pin" not in p:
            continue
        tip = np.asarray(p["Tip"], dtype=np.float64).reshape(-1)[:3]
        pin = np.asarray(p["Pin"], dtype=np.float64).reshape(-1)[:3]
        segs.append((tip, pin))

    if len(segs) < 2:
        return False

    for (t1, p1), (t2, p2) in combinations(segs, 2):
        if line_too_close_mm(spacing_xyz, t1, p1, t2, p2, tol_mm=tol_mm):
            return True
    return False


# -----------------------------
# Cross-slice merging for points
# -----------------------------
def merge_detections_across_slices(
    dets: List[Dict[str, Any]],
    cross_slice_tol_mm: float = 6.0,
    max_z_gap_slices: int = 2,
    use_angle: bool = True,
    angle_tol_deg: float = 5.0,
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """
    Merge same-class detections across adjacent slices into 3D points.
    Each det is expected to contain at least: x0, y0, z, angle, score.
    """
    if len(dets) == 0:
        return [], []

    n = len(dets)
    uf = UnionFind(n)

    z = np.array([int(d["z"]) for d in dets], dtype=np.int32)
    x = np.array([float(d["x0"]) for d in dets], dtype=np.float64)
    y = np.array([float(d["y0"]) for d in dets], dtype=np.float64)
    ang = np.array([float(d.get("angle", 0.0)) for d in dets], dtype=np.float64)
    sc = np.array([float(d.get("score", 0.0)) for d in dets], dtype=np.float64)

    # We need spacing to compute real mm distance; store in dets if available,
    # otherwise treat voxel spacing as (1,1,1) and rely on CT spacing in caller.
    # Here we only do voxel distance on (x,y,z) and caller should pass a mm-based threshold if needed.
    # To preserve your original behavior, we do mm distance using det["x_mm","y_mm","z_mm"] if present.
    has_mm = ("x_mm" in dets[0]) and ("y_mm" in dets[0]) and ("z_mm" in dets[0])
    if has_mm:
        xm = np.array([float(d["x_mm"]) for d in dets], dtype=np.float64)
        ym = np.array([float(d["y_mm"]) for d in dets], dtype=np.float64)
        zm = np.array([float(d["z_mm"]) for d in dets], dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            dz = abs(int(z[i]) - int(z[j]))
            if dz == 0:
                continue
            if dz > int(max_z_gap_slices):
                continue

            if has_mm:
                dist_mm = float(np.sqrt((xm[i] - xm[j]) ** 2 + (ym[i] - ym[j]) ** 2 + (zm[i] - zm[j]) ** 2))
            else:
                dist_mm = float(np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2))

            if dist_mm > float(cross_slice_tol_mm):
                continue

            if use_angle:
                ddeg = cal_angle_diff_deg(ang[i], ang[j])
                if ddeg > float(angle_tol_deg):
                    continue

            uf.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    merged_xyzang: List[List[float]] = []
    merged_info: List[Dict[str, Any]] = []

    for _, idxs in groups.items():
        idxs_sorted = sorted(idxs, key=lambda k: sc[k], reverse=True)
        # Weighted average by score (avoid 0-sum)
        w = np.maximum(sc[idxs_sorted], 1e-6)
        w = w / w.sum()

        mx = float((x[idxs_sorted] * w).sum())
        my = float((y[idxs_sorted] * w).sum())
        mz = float((z[idxs_sorted] * w).sum())
        mang = float((ang[idxs_sorted] * w).sum())

        merged_xyzang.append([mx, my, mz, mang])
        merged_info.append({
            "members": [int(i) for i in idxs_sorted],
            "num_members": int(len(idxs_sorted)),
            "score_max": float(sc[idxs_sorted[0]]),
        })

    return merged_xyzang, merged_info


# -----------------------------
# HU-based scoring
# -----------------------------
def sample_line_hu_stats(ct_img: sitk.Image, p1_xyz, p2_xyz) -> Tuple[float, float, int]:
    arr = sitk.GetArrayFromImage(ct_img)  # (z, y, x)
    p1 = np.asarray(p1_xyz[:3], dtype=float)
    p2 = np.asarray(p2_xyz[:3], dtype=float)

    voxel_dist = float(np.linalg.norm(p2 - p1))
    num_samples = max(int(voxel_dist) + 1, 2)

    pts = np.linspace(p1, p2, num=num_samples)
    pts = np.round(pts).astype(int)

    z_max, y_max, x_max = arr.shape
    xs = np.clip(pts[:, 0], 0, x_max - 1)
    ys = np.clip(pts[:, 1], 0, y_max - 1)
    zs = np.clip(pts[:, 2], 0, z_max - 1)

    vals = arr[zs, ys, xs].astype(np.float32)
    if vals.size == 0:
        return 0.0, 0.0, 0

    mean_hu = float(vals.mean())
    std_hu = float(vals.std(ddof=0) + 1e-6)
    return mean_hu, std_hu, int(vals.size)


def compute_pair_score(
    ct_img: sitk.Image,
    tip_xyzang: np.ndarray,
    pin_xyzang: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> float:
    tip = np.asarray(tip_xyzang, dtype=float)
    pin = np.asarray(pin_xyzang, dtype=float)

    length_mm = calculate_physical_distance_mm(tip[:3], pin[:3], spacing_xyz)
    if abs(length_mm - NEEDLE_LENGTH_PRIOR_MM) > NEEDLE_LENGTH_TOL_MM:
        return NEG_INF

    angle_diff = cal_angle_diff_deg(tip[3], pin[3])
    if angle_diff > ANGLE_DIFF_TOL_DEG:
        return NEG_INF

    mean_hu, std_hu, n_samples = sample_line_hu_stats(ct_img, tip[:3], pin[:3])
    if n_samples <= 0:
        return NEG_INF

    score = mean_hu / std_hu
    score -= 0.1 * abs(float(tip[2] - pin[2]))  # penalize z mismatch
    score -= 0.01 * float(length_mm)            # mild length penalty
    return float(score)


def build_score_matrix(ct_img: sitk.Image, tip_data: List[List[float]], pin_data: List[List[float]]) -> np.ndarray:
    spacing_xyz = ct_img.GetSpacing()
    tip_data = np.asarray(tip_data, dtype=float)
    pin_data = np.asarray(pin_data, dtype=float)

    m, n = int(len(tip_data)), int(len(pin_data))
    score_mat = np.full((m, n), NEG_INF, dtype=float)

    for i in range(m):
        for j in range(n):
            score_mat[i, j] = compute_pair_score(ct_img, tip_data[i], pin_data[j], spacing_xyz)

    return score_mat


# -----------------------------
# Greedy matching + optional merge duplicates
# -----------------------------
def merge_duplicate_needles(pairs: List[Dict[str, Any]], spacing_xyz, target_n: Optional[int] = None) -> List[Dict[str, Any]]:
    def needle_dist(p1: Dict[str, Any], p2: Dict[str, Any]) -> float:
        d_tip = calculate_physical_distance_mm(p1["Tip"][:3], p2["Tip"][:3], spacing_xyz)
        d_pin = calculate_physical_distance_mm(p1["Pin"][:3], p2["Pin"][:3], spacing_xyz)
        return max(d_tip, d_pin)

    pairs = sorted(pairs, key=lambda x: float(x["score"]), reverse=True)
    merged: List[Dict[str, Any]] = []

    while pairs:
        base = pairs.pop(0)
        cluster = [base]
        remain = []
        for p in pairs:
            if needle_dist(base, p) < MERGE_DIST_TOL_MM:
                cluster.append(p)
            else:
                remain.append(p)

        tips = np.stack([c["Tip"] for c in cluster], axis=0)
        pins = np.stack([c["Pin"] for c in cluster], axis=0)
        scores = [float(c["score"]) for c in cluster]

        merged.append({
            "Tip": tips.mean(axis=0),
            "Pin": pins.mean(axis=0),
            "score": float(max(scores)),
        })

        pairs = remain
        if target_n is not None and len(merged) >= int(target_n):
            break

    return merged


def greedy_match(
    score_mat: np.ndarray,
    tip_data: List[List[float]],
    pin_data: List[List[float]],
    ct_img: sitk.Image,
    n_prior: int,
    enable_nocross: bool = True,
    nocross_tol_mm: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Greedy matching with optional NoCross constraint.
    """
    spacing_xyz = ct_img.GetSpacing()
    tip_data = np.asarray(tip_data, dtype=float)
    pin_data = np.asarray(pin_data, dtype=float)

    score = score_mat.copy()
    m, n = score.shape

    matched_pairs: List[Dict[str, Any]] = []
    used_tip = set()
    used_pin = set()

    while True:
        max_idx = np.unravel_index(int(np.argmax(score)), score.shape)
        i, j = int(max_idx[0]), int(max_idx[1])
        max_val = float(score[i, j])

        if max_val <= NEG_INF:
            break

        if i in used_tip or j in used_pin:
            score[i, j] = NEG_INF
            continue

        candidate = {
            "Tip": tip_data[i].copy(),
            "Pin": pin_data[j].copy(),
            "score": float(max_val),
            "tip_idx": i,
            "pin_idx": j,
        }

        if enable_nocross:
            tmp = matched_pairs + [candidate]
            if check_intersections(spacing_xyz, tmp, tol_mm=float(nocross_tol_mm)):
                score[i, j] = NEG_INF
                continue

        matched_pairs.append(candidate)
        used_tip.add(i)
        used_pin.add(j)

        score[i, :] = NEG_INF
        score[:, j] = NEG_INF

    # Duplicate merge (same behavior as your original flow)
    if len(matched_pairs) > int(n_prior):
        matched_pairs = merge_duplicate_needles(matched_pairs, spacing_xyz, target_n=int(n_prior))
    else:
        matched_pairs = merge_duplicate_needles(matched_pairs, spacing_xyz, target_n=None)

    return matched_pairs


def greedy_match_wo_nocross(score_mat: np.ndarray, tip_data, pin_data, n_prior: int) -> List[Dict[str, Any]]:
    """
    Pure greedy 1-1 matching without NoCross check.
    """
    score = score_mat.copy()
    tip_data = np.asarray(tip_data, dtype=float)
    pin_data = np.asarray(pin_data, dtype=float)

    matched_pairs: List[Dict[str, Any]] = []
    used_tip = set()
    used_pin = set()

    while True:
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        max_val = float(score[i, j])
        if max_val <= NEG_INF:
            break

        if int(i) in used_tip or int(j) in used_pin:
            score[i, j] = NEG_INF
            continue

        matched_pairs.append({
            "Tip": tip_data[int(i)].copy(),
            "Pin": pin_data[int(j)].copy(),
            "score": float(max_val),
            "tip_idx": int(i),
            "pin_idx": int(j),
        })
        used_tip.add(int(i))
        used_pin.add(int(j))
        score[int(i), :] = NEG_INF
        score[:, int(j)] = NEG_INF

        if len(matched_pairs) >= int(n_prior):
            break

    return matched_pairs
