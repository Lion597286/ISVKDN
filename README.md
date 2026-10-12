# ISVKDN

**Inertial Sensor-to-Vision Knowledge Distillation Network for Human Action Recognition**

ISVKDN transfers knowledge from an inertial-sensor teacher to an RGB-video student. Accelerometer and gyroscope signals are represented as Gramian Angular Field (GAF) images for teacher training. The student learns through classification, soft-label distillation, semantic feature matching, and semantic preservation. At inference, only RGB video is required.

This repository provides the PyTorch implementation and controlled experiments on UESTC-MMEA-CL.

## Method

<p align="center">
  <img src="assets/figures/framework.png" alt="Overview of the ISVKDN teacher-student framework" width="100%">
</p>

- **SMMRF:** Similarity Matrix-based Multi-Relational Fusion integrates accelerometer and gyroscope features through concatenation, gating, and feature crossing.
- **SGSM:** Sparse Graph-guided Semantic Mapping aligns teacher and student features using action semantics and locally linear reconstruction graphs.
- **Video student:** A multiscale Temporal Relation Network (TRN) with a BN-Inception backbone samples eight RGB frames per video.

<table>
  <tr>
    <td align="center" width="50%"><b>SMMRF</b></td>
    <td align="center" width="50%"><b>SGSM</b></td>
  </tr>
  <tr>
    <td><img src="assets/figures/smmrf.png" alt="SMMRF fusion module from the manuscript" width="100%"></td>
    <td><img src="assets/figures/sgsm.png" alt="SGSM semantic mapping module from the manuscript" width="100%"></td>
  </tr>
</table>

## Environment

The development environment uses Python 3.8.19, PyTorch 1.13.0, torchvision 0.14.0, and CUDA 11.7.

```bash
conda create -n isvkdn python=3.8.19 -y
conda activate isvkdn
python -m pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

Run commands from the repository root with a CUDA-capable GPU. Pretrained VGG16-BN and BN-Inception weights are loaded through PyTorch's model cache.

## Data preparation

Prepare the videos and corresponding accelerometer and gyroscope GAF images in the following structure:

```text
GAF_ROOT/
├── Acc_train/   Acc_val/   Acc_test/
└── Gyro_train/  Gyro_val/  Gyro_test/

VIDEO_ROOT/
└── video_train/ video_val/ video_test/
```

Each GAF split contains numbered class folders, such as `a01` for accelerometer and `g01` for gyroscope. Paired samples share an identifier:

```text
Acc_train/a01/1_example_GASF_a.png
Gyro_train/g01/1_example_GASF_g.png
video_train/1_example.mp4
```

Identifiers and class labels must agree across all modalities. Classes are numbered 1-32; class 1 corresponds to row 0 of the semantic embeddings. The prepared train/validation/test splits contain 4,553/653/1,316 samples. Videos are decoded directly, so pre-extracted frames are unnecessary.

The `data/` directory contains BERT (`32 × 768`) and GloVe class embeddings. Ordered action names are in [`UESTC-MMEA-CL.txt`](UESTC-MMEA-CL.txt).

## Controlled experiments

In [`run_stage2_experiments.py`](run_stage2_experiments.py), set `GAF_ROOT` and `VIDEO_ROOT` to your data directories, and set `PREFERRED_PYTHON = Path(sys.executable)` to use the active environment. Choose the experiments and random seeds:

```python
SELECTED_EXPERIMENTS = ["A0", "A1", "A2", "A3"]
SEEDS = [2025]
RUN_MODE = "train"
```

```bash
python run_stage2_experiments.py
```

| Group | Configurations |
| --- | --- |
| A0-A3 | Video baseline, soft-label distillation, SGSM feature matching, and full ISVKDN |
| B0-B3 | Mean fusion, concatenation, concatenation with gating, and full SMMRF |
| C0/C1/C2/C4 | Disable SGSM feature matching, replace it with raw feature matching, shuffle class semantics, or disable graph propagation |

A3 is the full-SGSM reference (C3). For a consistent B0-B3 comparison, use the same seeds and set `REUSE_EXISTING_FULL_TEACHER = False`. The supplied experiment selection is `C0`; update it to the configurations you intend to run. Set `RUN_MODE = "smoke"` for a single-batch check.

The teacher and student implementations are in `model/`, semantic mapping in `uestc_semantic_mapping.py`, and paired data loading in `uestc_paired_dataset.py`. The BN-Inception support code retains its upstream license in [`tf_model_zoo/LICENSE`](tf_model_zoo/LICENSE).
