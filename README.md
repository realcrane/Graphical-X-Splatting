# Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty

**Doğa Yılmaz¹ · Jialin Zhu² · Deshan Gong³ · He Wang¹**

¹ University College London · ² Baidu Inc. · ³ The University of Hong Kong

[**arXiv**](https://arxiv.org/abs/2601.19843) · **SIGGRAPH** (coming soon) · [**BibTeX**](#citation)

> This codebase has been tested on **Ubuntu 22.04 LTS and 24.04 LTS** with
> **NVIDIA RTX 3090** and **RTX 4090** GPUs, as well as on an **NVIDIA GH200–based HPC cluster**.

---

## 1. Environment setup

These steps take you to a working install. Run everything from the repository root.

### 1.1 Clone

```bash
git clone https://github.com/realcrane/Graphical-X-Splatting.git
cd Graphical-X-Splatting
```

### 1.2 Create the conda environment

```bash
conda create -n graphixs python=3.10 -y
conda activate graphixs
```

### 1.3 Install PyTorch and a matching CUDA toolkit

Install PyTorch (we used version 2.7.0 with CUDA 12.8).

> **Note:** The prebuilt RoMa `local_corr` op (`fused_local_corr`, see
> [Section 1.6](#16-preprocessing-prerequisites-only-needed-for-section-2)) and the CUDA
> extensions in step 1.5 are ABI-tied to this PyTorch build. A newer `torch` (e.g. 2.11) makes RoMa crash with `std::bad_alloc` and breaks the extensions with `undefined symbol: ...decref_pyobjectEv`.

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

To build the CUDA extensions in Step 1.5, you need a CUDA toolkit (with `nvcc`) that matches your PyTorch CUDA version. The simplest approach is to install it directly into your active Conda environment from the NVIDIA channel:

```bash
conda install -c nvidia/label/cuda-12.8.0 -c conda-forge cuda-toolkit "cuda-version=12.8" -y
```

> **Note:** any *later* `conda install` into this env (e.g. COLMAP in
> [Section 1.6](#16-preprocessing-prerequisites-only-needed-for-section-2)) can silently remove or downgrade `python`/CUDA while solving. Always carry the pins along, e.g. `conda install -c conda-forge <pkg> "cuda-version=12.8" "python=3.10" -y`.

Once installed, `nvcc` will point to the environment's build (`$CONDA_PREFIX/bin/nvcc`). To ensure the extensions in Step 1.5 compile using this specific toolchain rather than your system's default CUDA, export `CUDA_HOME=$CONDA_PREFIX`.

<details>
<summary><b>No matching conda CUDA build? Install from the NVIDIA runfile + auto-switch per env</b></summary>

**1. Install the toolkit via the runfile installer.** Download the matching version from the [CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive) and run its `.run` installer. In the installer options **uncheck everything except the CUDA Toolkit** (do not install the bundled driver if you already have one). Check the toolkit's C++ compiler requirements and make sure your `gcc`/`g++` is compatible first.

When the installer asks to repoint the `/usr/local/cuda` symlink to the version you just
installed, choose **No**, keep your existing system default. (If you accidentally said yes, or it didn't ask, remove and recreate the symlink to the default you want.) The runfile installs to a versioned path like `/usr/local/cuda-12.8`.

**2. Add per-env activate/deactivate scripts** so the env uses the runfile CUDA and restores the system one on exit. Replace `<ENV>` with your env name and adjust the versioned paths.

`~/miniconda3/envs/<ENV>/etc/conda/activate.d/activate.sh`:

```bash
ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ORIGINAL_PATH=$PATH
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

`~/miniconda3/envs/<ENV>/etc/conda/deactivate.d/deactivate.sh`:

```bash
export PATH=$ORIGINAL_PATH
export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH
unset ORIGINAL_LD_LIBRARY_PATH
unset ORIGINAL_PATH
```

Re-activate the env (`conda deactivate && conda activate <ENV>`) and confirm `nvcc --version` reports the version you intend to build against. Then set `CUDA_HOME=/usr/local/cuda-12.8` before step 1.5.

</details>

### 1.4 Install Python dependencies

```bash
pip install "numpy>=2" pillow tqdm rich matplotlib prompt_toolkit pytorch-msssim scipy opencv-python plyfile tensorboard
```

### 1.5 Build the CUDA extensions

Point the build at the CUDA toolchain from step 1.3, then compile:

```bash
export CUDA_HOME=$CONDA_PREFIX          # or /usr/local/cuda-12.8 if you used the runfile install
pip install --no-build-isolation --no-cache-dir submodules/diff-graphixs-rasterization
pip install --no-build-isolation --no-cache-dir submodules/simple-knn
```

> Use `--no-cache-dir` so the extensions are actually recompiled against the active PyTorch.
> If you ever change the `torch` version, rebuild both with the same command (a plain
> `--force-reinstall` reuses the stale cached wheel and you'll hit `undefined symbol` errors at import time).

### 1.6 Preprocessing prerequisites

Data preprocessing additionally requires COLMAP and ffmpeg, plus RoMa for dense matching:

```bash
conda install -c conda-forge "colmap=3.11.1" ffmpeg "cuda-version=12.8" "python=3.10" -y

pip install git+https://github.com/Parskatt/RoMa.git

pip install fused_local_corr==0.2.3
```

---

## 2. Data preprocessing

All preprocessing tools live in `preprocessing/`. They turn raw multi-camera videos into the camera poses and initial point cloud that training (`train.py`) consumes. This stage needs the COLMAP / ffmpeg / `romatch` dependencies from [Section 1.6](#16-preprocessing-prerequisites-only-needed-for-section-2).

The pipeline has two stages, both **required**:

1. **Camera poses (COLMAP).** Recover per-camera intrinsics/extrinsics and undistorted frames from the raw videos.
2. **Initial 4D point cloud (RoMa).** Using the poses from stage 1, run RoMa dense matching to produce per-frame point clouds, then merge them into a temporal **4D** point cloud (`points4D.ply`, with per-point time + velocity). This is the init the dynamic model expects (`--init_type sfm_4D`).

### Script reference

| Script | Purpose |
|---|---|
| `colmap_from_video_n3dv.py` | N3DV/LLFF capture → frames + COLMAP poses + dense `points3D.ply` |
| `colmap_from_video_immersive.py` | Google Immersive capture → fisheye-undistorted frames + per-frame COLMAP (+ `.npy` flow maps) |
| `pcd_from_images_n3dv.py` | RoMa: per-frame point clouds from N3DV `camXX_YYYY.png` images |
| `pcd_from_images_immersive.py` | RoMa: per-frame point clouds from Immersive `camera_XXXX.png` images |
| `pcd_merge_4d.py` | Merge per-frame PLYs into one temporal `points4D.ply` (adds time + velocity) |

---

### N3DV (Neural 3D Video)

**Stage 0 — download the datasets**

N3DV (Neural 3D Video) is released under **CC-BY-NC 4.0** on the official [release page](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0). Download the scene zip files you need from there and unzip them into your data folder, organized as follows:

```
data/
└── N3DV/
    ├── coffee_martini/
    │   ├── cam00.mp4
    │   ├── cam01.mp4
    │   ├── …
    │   ├── camNN.mp4
    │   └── poses_bounds.npy
    ├── cook_spinach/
    ├── cut_roasted_beef/
    ├── flame_salmon/
    ├── flame_steak/
    └── sear_steak/
```

Each scene folder holds the per-camera videos, named `cam00.mp4`, `cam01.mp4`, … `camNN.mp4`, and a `poses_bounds.npy`, which is what the code expects. If your downloaded files are named differently, rename them to match this layout.

**Stage 1 — COLMAP poses**:

```bash
python preprocessing/colmap_from_video_n3dv.py data/N3DV/<scene>/
```

This extracts frames to `data/N3DV/<scene>/images/` as `camXX_YYYY.png`, runs COLMAP, and writes `cameras.txt`, `images.txt`, `points3D.txt`, and the dense **`points3D.ply`** into the scene folder.

**Stage 2 — RoMa 4D point cloud**:

```bash
python preprocessing/pcd_from_images_n3dv.py \
    --colmap_dir data/N3DV/<scene>/ \
    --image_dir  data/N3DV/<scene>/images/ \
    --output_dir data/N3DV/<scene>/point_clouds/

python preprocessing/pcd_merge_4d.py \
    --input_dir data/N3DV/<scene>/point_clouds \
    --output    data/N3DV/<scene>/points4D.ply
```

Useful `pcd_from_images_n3dv.py` knobs: `--frames`/`--max_frames` (subset),
`--max_views` (cameras per frame), `--min_certainty` / `--match_threshold` (match
filtering), `--max_points`, `--outlier_threshold`. `pcd_merge_4d.py` knobs:
`--max_points`, `--max_points_per_frame`, `--region_radius` (NN match distance),
`--frame_stride`, `--motion_weight`.

To run stage 2 for **all six N3DV scenes** at once (after stage 1 has been done for each), use the batch script:

```bash
bash shell_scripts/generate_4d_init.sh
```

---

### Google Immersive (GI)

**Stage 0 — download the datasets**

The GI (Google Immersive) scenes are hosted by Google at `https://storage.googleapis.com/deepview_video_raw_data/<scene>.zip`. The official dataset has 15 scenes, but we use a subset of five — `01_Welder`, `02_Flames`, `05_Horse`, `06_Goats`, and `10_Alexa`. Download these scene zip files and unzip them into your data folder, organized as follows:

```
data/
└── google_immersive/
    ├── 01_Welder/
    │   ├── camera_0001.mp4
    │   ├── camera_0002.mp4
    │   ├── …
    │   ├── camera_NNNN.mp4
    │   └── models.json
    ├── 02_Flames/
    ├── 05_Horse/
    ├── 06_Goats/
    └── 10_Alexa/      # downloaded as 10_Alexa_Meade_Face_Paint_1 — rename the folder to 10_Alexa
```

Each scene folder holds the per-camera videos, named `camera_0001.mp4`, `camera_0002.mp4`, … `camera_NNNN.mp4`, and a `models.json` (fisheye intrinsics/extrinsics), which is what the code expects. The Alexa scene is published under the longer name `10_Alexa_Meade_Face_Paint_1`, so rename its folder to `10_Alexa` after unzipping; the other four already match. These raw `<scene>/` folders are the inputs to Stage 1, which produces the `<scene>_dist/` used for training.

**Stage 1 — per-frame COLMAP poses:**

```bash
python preprocessing/colmap_from_video_immersive.py \
    --videopath data/google_immersive/<scene> \
    --startframe 0 --endframe 30
```

This creates `data/google_immersive/<scene>_dist/`. It holds one `colmap_<i>/`
per frame, each with `manual/` (COLMAP poses from `models.json`), fisheye-undistorted
`images/` (`camera_XXXX.png`), and a triangulated COLMAP model in `sparse/0/` plus
per-camera `.npy` distortion flow maps at the `<scene>_dist/` root (`camera_XXXX.npy`,
used at render time by `utils/fisheye_utils.py`). `--startframe`/`--endframe` select the
frame range.

**Stage 2 — RoMa 4D point cloud**:

```bash
python preprocessing/pcd_from_images_immersive.py \
    --base_dir   data/google_immersive/<scene>_dist \
    --output_dir data/google_immersive/<scene>_dist/point_clouds

python preprocessing/pcd_merge_4d.py \
    --input_dir data/google_immersive/<scene>_dist/point_clouds \
    --output    data/google_immersive/<scene>_dist/points4D.ply
```

`pcd_from_images_immersive.py` reads `colmap_<i>/manual/{cameras.txt,images.txt}` and
`colmap_<i>/images/camera_XXXX.png`, and accepts the same matching/filtering knobs as the N3DV variant (`--frames`, `--max_views`, `--min_certainty`, `--match_threshold`,
`--max_points`, …).

---

## 3. Usage

The example below uses the Google Immersive `Flames` scene with the standard Student's-t config. Substitute your own scene / config as needed.

Configs live in `configs/` and follow the naming `graphi{ts,gs}_<setting>_<dataset>.json`, where `ts` = Student's-t (GraphiTS) and `gs` = approximate Gaussian (GraphiGS), and `<dataset>` is `n3dv` or `gi` (Google Immersive).

### 3.1 Train

```bash
python train.py \
    -s data/google_immersive/02_Flames_dist \
    -m output/flames_graphits_standard \
    --config configs/graphits_standard_gi.json
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `-s, --source_path` | — | Input data folder |
| `-m, --model_path` | — | Output directory |
| `--config` | none | JSON config overriding argument defaults |
| `--iterations` | `30000` | Total training iterations |
| `--test_iterations` | `10000 20000 30000` | Iterations to evaluate on |
| `--checkpoint_iterations` | `10000 20000 30000` | Iterations to checkpoint |
| `--cap_max` | from config | Max number of primitives |
| `--start_checkpoint` | none | Resume from a `.pth` checkpoint |

Outputs under `-m`: `chkpnt<iter>.pth`, `chkpnt_best_test.pth`, `chkpnt_best_train.pth`,
a `cfg_args.json` snapshot, and image previews under `imgs/`.

### 3.2 Render

```bash
python render.py \
    -m output/flames_graphits_standard \
    --config output/flames_graphits_standard/cfg_args.json \
    --iteration 30000 \
    --skip_train
```

Renders to `<model_path>/{train,test}/ours_<iter>/{renders,gt,grids}/`. Useful extras:
`--render-videos` (assemble MP4s, needs ffmpeg), `--benchmark` (FPS / GPU-memory benchmark instead of saving images), `--skip_test`.

### 3.3 Metrics

```bash
python metrics.py -m output/flames_graphits_standard --skip_train
```

Prints SSIM / PSNR / LPIPS (VGG & AlexNet) / MS-SSIM / DSSIM and writes
`test_results.json` (and `train_results.json` unless `--skip_train`) into the model folder. `-m` accepts multiple model paths.

### 3.4 Full train → render → metrics

```bash
python train.py   -s data/google_immersive/02_Flames_dist -m output/flames_graphits_standard --config configs/graphits_standard_gi.json

python render.py  -m output/flames_graphits_standard --config output/flames_graphits_standard/cfg_args.json --iteration 30000 --skip_train

python metrics.py -m output/flames_graphits_standard --skip_train
```

### 3.5 Launcher (interactive TUI)

`launcher.py` wraps training in a guided terminal UI. Its extra dependencies are `rich` and `prompt_toolkit` (both installed in [Section 1.4](#14-install-python-dependencies)).

```bash
python launcher.py
```

The wizard walks through five steps — data source → config file → output folder → execution environment (GPU selection) → extra CLI arguments — then shows a summary and launches a live training dashboard. It can optionally run rendering and metrics automatically after training.

> For running many experiments at once, see the batch scripts under `shell_scripts/`.
> For local runs, see shell_scripts/local_commands_{n3dv,immersive}.sh. For runs on HPC via SLURM, see shell_scripts/slurm_commands_{n3dv,immersive}.sh.

---

## Acknowledgements

This work was supported in part by the Dr. Rabin Ezra Scholarship (Charity No. 1116049), awarded to Doğa Yılmaz, and the UK Research and Innovation AIRR Innovator Award (0261-5654-9320-1).

Our code is built on top of the [vanilla 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [SSS](https://github.com/realcrane/3D-student-splatting-and-scooping) code bases. We thank the authors for open-sourcing their implementation.

---

## Citation

```bibtex
@inproceedings{yilmaz2026graphical,
  author = {Doğa Yılmaz and Jialin Zhu and Deshan Gong and He Wang},
  title = {Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty},
  booktitle = {ACM SIGGRAPH},
  year = {2026}
}
```

---

## Contact

For any code related questions, please reach out to Doğa Yılmaz at **doga.yilmaz[at]ucl.ac.uk**.
