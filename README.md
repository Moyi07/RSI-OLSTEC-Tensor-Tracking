# RSI-OLSTEC: Robust Side-Informed Online Low-rank Tensor Subspace Tracking

[![MATLAB](https://img.shields.io/badge/Language-MATLAB-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📖 Introduction

**RSI-OLSTEC** is a robust and physics-guided extension of the online low-rank tensor subspace tracking framework. It is specifically designed to handle highly corrupted, incomplete streaming data (such as process monitoring videos in Wire Arc Additive Manufacturing, WAAM). 

Traditional online tensor completion methods often suffer from severe performance degradation when facing impulsive noise (e.g., manufacturing spatter) or abrupt physical state mutations. To address these challenges, RSI-OLSTEC introduces two core mechanisms:

1. **Robust Huber Penalty:** Replaces the standard L2-norm with a Huber loss function during the Recursive Least Squares (RLS) updates. This significantly suppresses the impact of sparse, high-magnitude impulsive outliers (spatter noise) without discarding valuable background information.
2. **Physics-Guided Adaptive Forgetting Factor:** Integrates 1D external physical sensor data (Side Information, e.g., real-time melt pool width or current/voltage fluctuations) to dynamically adjust the tracking factor ($\lambda$). This allows the algorithm to rapidly adapt to sudden topological changes (mutations) while maintaining steady-state stability.

## 🌟 Key Features

* **Real-Time Processing:** Processes streaming tensor data (video frames) sequentially with low computational overhead.
* **Impulsive Noise Robustness:** Effectively filters out sparse, high-intensity noise anomalies.
* **Side-Information Integration:** Bridges physical sensing data with purely mathematical tensor decomposition.
* **Matrix-Inversion-Free Updates:** Optimized robust RLS update steps to ensure numerical stability and speed.

## 📊 Dataset

The real-world evaluations and experiments for this algorithm utilize the public **WAAM-ViD** (Wire Arc Additive Manufacturing Video) dataset. 

You can access and download the original dataset here:  
🔗 [WAAM-ViD Repository by IFRA-Cranfield](https://github.com/IFRA-Cranfield/WAAM-ViD)

## 📜 Acknowledgement & Attribution

The core architecture of this codebase builds upon the highly efficient **OLSTEC** (Online Low-rank Tensor Subspace Tracking from Incomplete Data) algorithm developed by Hiroyuki Kasai. 

We sincerely thank the original author for making the baseline code open-source. If you find our modified RSI-OLSTEC framework useful, please consider citing both our work and the original OLSTEC paper:

**Original OLSTEC Paper:**
> H. Kasai, "Online low-rank tensor subspace tracking from incomplete data by CP decomposition using recursive least squares," *2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Shanghai, China, 2016, pp. 2519-2523.  
> [DOI: 10.1109/ICASSP.2016.7472131](https://doi.org/10.1109/ICASSP.2016.7472131) | [Original GitHub Repository](https://github.com/hiroyuki-kasai/OLSTEC)

**(Please insert the citation for your RSI-OLSTEC paper here once published)**
