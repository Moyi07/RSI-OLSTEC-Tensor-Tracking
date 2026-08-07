# Reproducibility Checklist and Run Instructions

This file lists the minimal steps required to reproduce the experiments in this repository.

## 1. Requirements

- MATLAB R2024b, which was used for the reported experiments.
- Statistics and Machine Learning Toolbox, required by `Exp_S2.m` and `Exp_R1.m` for boxplot generation.
- Tensor Toolbox for MATLAB 2.6 and Poblano Toolbox 1.1 for the CP-WOPT baseline used in the three `Exp_S3_*.m` scripts and in `Exp_R1.m`, `Exp_R2.m`, and `Exp_R3.m`.
- The released third-party benchmark files under `benchmark/`.

Tensor Toolbox 2.6 and Poblano Toolbox 1.1 are not redistributed because the licenses bundled with these legacy releases restrict redistribution. Obtain the required releases lawfully from their publishers or an authorized archive and place them at:

```text
tool/tensor_toolbox_2.6/
tool/poblano_toolbox_1.1/
```

Do not substitute a different toolbox release when reproducing the reported numerical values without first validating compatibility.

## 2. Data

The WAAM-ViD reference data are not redistributed. Download the official dataset from:

[https://doi.org/10.57996/cran.ceres-2763](https://doi.org/10.57996/cran.ceres-2763)

Place the selected video sequence at:

```text
dataset/video/250312-110206-video_1.mp4
```

Place the official metadata file at:

```text
dataset/WAMVID_metadata.csv
```

See `dataset/README.md` for the dataset citation, license, file identifiers, and integrity information.

## 3. Initial Setup

In MATLAB, run:

```matlab
run('run_me_first.m');
```

This adds the repository, auxiliary functions, benchmark folders, and any locally installed toolboxes under `tool/` to the MATLAB path.

## 4. Experiment Scripts

Synthetic experiments:

```matlab
run('Exp_S1.m');
run('Exp_S2.m');
run('Exp_S3_LN_GaussianOnly.m');
run('Exp_S3_LE_SpatterOnly.m');
run('Exp_S3_LNE_GaussianSpatter.m');
run('Exp_S4.m');
run('Exp_S5.m');
run('Exp_S6.m');
```

Real-video experiments:

```matlab
run('Exp_R1.m');
run('Exp_R2.m');
run('Exp_R3.m');
```

Supplementary video export:

```matlab
run('Export_R2_Supplementary_Video.m');
```

## 5. Outputs

Generated figures, statistics, and supplementary videos are written under
`result/`. The three S3 observation models use separate condition-specific
subdirectories. Formal CSV files and non-image-bearing final statistics MAT
files are retained for verification. Checkpoint files, MATLAB command-window
captures, temporary files, and intermediate video exports are not part of the
formal result release. The editable real-video figures
`Fig_Running_Final.fig`, `Fig_MultiFrame_Matrix_R2.fig`, and
`Fig_MultiFrame_Matrix_R3.fig`, together with `R3_stats.mat`, are generated
locally but are not redistributed because they contain machine-readable
WAAM-derived frames. Users can regenerate them after obtaining the reference
data. When dynamic visualization and result export are enabled in R2,
`Fig_Running_Final.fig` is saved after the last video frame has been displayed.

## 6. Notes

- Use the checked-in parameter values in `Exp_*.m` for full reproduction of the reported experiments.
- If a real-video script reports missing data, confirm that the required WAAM-ViD video and metadata files have been placed at the exact paths listed above.
- If a CP-WOPT run reports missing toolbox functions, confirm that Tensor Toolbox 2.6 and Poblano Toolbox 1.1 are installed at the specified local paths, then run `run_me_first.m` again.
- Synthetic Experiments S1--S6 do not require the WAAM-ViD files.
