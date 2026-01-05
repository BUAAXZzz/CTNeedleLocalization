# CTNeedleLocalization

Official Code of  
**“Multi-needle Localization for Pelvic Seed Implant Brachytherapy based on Tip-handle Detection and Matching”**

---

## 📝 Citation

If you find this work or code useful for your research, please cite our paper:

```bibtex
@misc{xiao2025multineedlelocalizationpelvicseed,
      title={Multi-needle Localization for Pelvic Seed Implant Brachytherapy based on Tip-handle Detection and Matching}, 
      author={Zhuo Xiao and Fugen Zhou and Jingjing Wang and Chongyu He and Bo Liu and Haitao Sun and Zhe Ji and Yuliang Jiang and Junjie Wang and Qiuwen Wu},
      year={2025},
      eprint={2509.17931},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.17931}
}
```

---

## 📌 Overview

This repository provides a complete pipeline for **automatic multi-needle localization in intraoperative CT** for pelvic seed implant brachytherapy.

The framework consists of two stages:

1. **2D Detection**  
   Slice-wise detection of needle tips and handles.

2. **3D Matching and Reconstruction**  
   Greedy matching and merging of tip–handle pairs to reconstruct full 3D needle trajectories.

---

## 1️⃣ 2D Detection

### Reference and Acknowledgement

The 2D detection module in this project is implemented with reference to the following open-source projects:

- **CircleNet**  
  https://github.com/hrlblab/CircleNet

- **CenterNet**  
  https://github.com/xingyizhou/CenterNet

We sincerely thank the authors for making their work publicly available.

> The usage (training scripts, inference pipeline, configuration style, and workflow) of the 2D detection module is consistent with the original CircleNet / CenterNet repositories, with task-specific adaptations for needle tip and handle detection.

---

### Installation

Please follow the official CircleNet installation guide:

https://github.com/hrlblab/CircleNet/blob/master/docs/INSTALL2023.md

The environment configuration described in that document has been verified to work with this project.

---

## 2️⃣ 3D Matching and Needle Reconstruction

The 3D matching module is an original contribution of this work and implements the **Greedy Matching and Merging (GMM)** strategy described in the paper.  
This module contains **no neural network forward or inference code** and operates purely on geometric reasoning and CT intensity statistics.

Relevant files:

- `match3d_utils.py`
- `match3d_batch.py`

---

### Problem Formulation

After detecting 2D needle tips and handles across all axial slices, the reconstruction of full 3D needle trajectories is formulated as an **Unbalanced Assignment Problem with Constraints (UAP-C)**.

Let:
- **T = {t₁, …, tₘ}** be the set of detected needle tips  
- **H = {h₁, …, hₙ}** be the set of detected needle handles  

Each potential tip–handle pair (tᵢ, hⱼ) defines a candidate 3D needle path.

The objective is to select a subset of tip–handle pairs that maximizes the overall confidence score while satisfying physical and geometric constraints.

---

### Tip–Handle Pair Scoring

For each candidate tip–handle pair, a score is computed based on CT intensity statistics along the straight line segment connecting them.

The score is defined as:

```
O(i, j) = μ(i, j) / σ(i, j)
```

where μ(i, j) and σ(i, j) denote the mean and standard deviation of Hounsfield Unit (HU) values sampled along the candidate needle path.

This formulation favors paths with consistently high CT intensities, corresponding to metallic needle shafts.

---

### Constraints

Each candidate pairing must satisfy the following constraints:

1. **One-to-one matching**  
   Each tip and each handle can be matched at most once.

2. **Needle length constraint**  
   The 3D distance between the tip and handle must be close to a predefined needle length.

3. **Angle consistency constraint**  
   The difference between the detected tip angle and handle angle must be within a tolerance.

4. **Non-intersection (No-Cross) constraint**  
   No two reconstructed needle paths are allowed to intersect in 3D space.

5. **Needle count constraint**  
   The total number of reconstructed needles must equal the known prior number of implanted needles (**N_prior**).

---

### Greedy Matching Strategy

The UAP-C is solved using a greedy strategy:

1. A score matrix is constructed for all valid tip–handle pairs.
2. Pairs violating length or angle constraints are excluded.
3. The highest-scoring remaining pair that does not violate the No-Cross constraint is selected iteratively.
4. The process continues until no valid pairs remain.

If the number of matched pairs is less than or equal to **N_prior**, the result is accepted directly.

---

### Duplicate Path Merging

If the number of matched pairs exceeds **N_prior**, duplicate needle paths are assumed to exist and a merging process is triggered.

Duplicate paths are identified based on:
- Low matching scores
- Spatial proximity between tips and handles

For duplicate paths:
- Tip and handle positions are merged separately.
- The merged position is computed as a **HU-weighted centroid**, where the weights are derived from average non-zero HU values within the detected needle regions.

The merging process is performed iteratively until the number of remaining needle paths equals **N_prior**.

---

### Input Format

`match3d_batch.py` operates on a case-level directory structure:

```
case_xxx/
├── ct.mha              # Original CT volume
└── pred_2d.json        # 2D detection results
```

---

### Usage

```bash
python match3d_batch.py   --root /path/to/cases_root   --n_prior <number_of_needles>
```

---

### Output

For each case, the module generates:

```
pred_3d.json
```

The output contains reconstructed 3D needle trajectories, matched tip–handle pairs, and final matching scores.

---

## 📄 License

This project is released under the **MIT License**.
