# RSI-OLSTEC Tensor Tracking

This repository contains the MATLAB implementation used for the paper:

**Robust Side-Information-Guided Online Tensor Tracking for Dynamic Image Decoupling in Non-Stationary Manufacturing Processes**

The code implements RSI-OLSTEC and the benchmark scripts for the synthetic and WAAM-ViD experiments reported in the manuscript.

## Repository Structure

- `rsi_olstec.m`: proposed robust side-information-guided online tensor tracking algorithm.
- `Exp_S1.m`, `Exp_S2.m`, and `Exp_S4.m`--`Exp_S6.m`: synthetic benchmark
  and analysis experiments.
- `Exp_S3_*.m`: the three independently executable observation models of
  Experiment S3.
- `Exp_R1.m`--`Exp_R3.m`: real-video WAAM benchmark experiments.
- `Export_R2_Supplementary_Video.m`: supplementary video export script for the real-video tracking benchmark.
- `auxiliary/`: data loading, synthetic data generation, metric computation, checkpointing, and validation helpers.
- `benchmark/`: baseline algorithm wrappers and third-party benchmark implementations.
- `tool/`: local placement directory for externally obtained MATLAB toolboxes.
- `dataset/`: WAAM-ViD download and local data-placement instructions. Raw dataset files are not redistributed.

## Data

The raw WAAM-ViD files are not redistributed in this repository. Download the reference dataset from the official Cranfield University record:

[https://doi.org/10.57996/cran.ceres-2763](https://doi.org/10.57996/cran.ceres-2763)

The real-video experiments require the following local files:

```text
dataset/video/250312-110206-video_1.mp4
dataset/WAMVID_metadata.csv
```

See `dataset/README.md` for the dataset citation, license, expected file identifiers, and placement instructions.

## Requirements

- MATLAB R2024b, which was used to generate and verify the reported results.
- Statistics and Machine Learning Toolbox, required by `Exp_S2.m` and `Exp_R1.m` for boxplot generation.
- Tensor Toolbox for MATLAB 2.6 and Poblano Toolbox 1.1, required by the CP-WOPT baseline in the three `Exp_S3_*.m` scripts and in `Exp_R1.m`, `Exp_R2.m`, and `Exp_R3.m`. These legacy releases are not redistributed because their bundled licenses restrict redistribution. Install them locally as `tool/tensor_toolbox_2.6/` and `tool/poblano_toolbox_1.1/`.

## Quick Start

In MATLAB, run:

```matlab
run('run_me_first.m');
```

Then run the desired experiment, for example:

```matlab
run('Exp_S2.m');
run('Exp_S5.m');
run('Exp_S6.m');
run('Exp_R2.m');
```

Each script writes generated figures and statistics under `result/`. The three
S3 observation models use separate condition-specific subdirectories. Formal
CSV results for Experiments S1--S6 and R1--R3 and the non-image-bearing final
statistics MAT files are retained for verification. Checkpoint files, MATLAB
command-window captures, temporary files, and intermediate video exports are
excluded. The editable real-video figures `Fig_Running_Final.fig`,
`Fig_MultiFrame_Matrix_R2.fig`, and `Fig_MultiFrame_Matrix_R3.fig`, together
with `R3_stats.mat`, are generated locally but are not redistributed because
they contain machine-readable WAAM-derived frames. They can be regenerated
after obtaining the reference data. For R2, `Fig_Running_Final.fig` records
the final frame of the dynamic visual comparison and is saved automatically
when dynamic visualization and result export are enabled.

## Reproducibility

Detailed run instructions are provided in `REPRODUCE.md`. Experiment scripts
store their numerical settings with the generated statistics.

## Third-Party Code

This repository includes third-party baseline implementations and wrappers. Separately licensed components retain their original notices, while the restricted Tensor Toolbox 2.6 and Poblano Toolbox 1.1 releases must be obtained independently. See `THIRD_PARTY_NOTICE.md` for component-level attribution and licensing information.

## License

The original RSI-OLSTEC code and modifications are released under the MIT License in `LICENSE`. Third-party software and WAAM-ViD reference data remain subject to their respective licenses and are not relicensed by the repository-level MIT License.
