# Third-Party Code Notice

This repository combines the original RSI-OLSTEC implementation and experiment scripts with third-party baseline code used for comparison. The repository-level MIT License applies to the original RSI-OLSTEC code and modifications. It does not replace the license of any separately licensed component identified below.

## Included Components

| Component | Repository location | Attribution and source | License information |
| --- | --- | --- | --- |
| OLSTEC and OLSTEC benchmark wrappers | `benchmark/OLSTEC/` and Kasai-authored `*_mod.m` wrappers | Hiroyuki Kasai, [OLSTEC](https://github.com/hiroyuki-kasai/OLSTEC) | MIT License; the original 2017 copyright notice is retained in the root `LICENSE`. Separately licensed code, including GRASTA, remains an exception. |
| GRASTA core routines | `benchmark/grasta.1.2.0/grasta_stream.m` and `benchmark/grasta.1.2.0/admm_srp.m` | Jun He, Laura Balzano, and Arthur Szlam; GRASTA library | GNU Lesser General Public License, version 3 or any later version. See `benchmark/grasta.1.2.0/LICENSE.txt`, `LGPL.txt`, and `GPL.txt`. |
| GRASTA OLSTEC wrapper | `benchmark/grasta.1.2.0/grasta_mod.m` | Created for the OLSTEC package by Hiroyuki Kasai | OLSTEC repository-level MIT License; the wrapper calls the separately licensed GRASTA routines above. |
| GROUSE | `benchmark/Grouse/` | Core code by Ben Recht and Laura Balzano; the wrapper header also attributes Laura Balzano, Benjamin Recht, and Robert Nowak and records modification for OLSTEC by Hiroyuki Kasai | Included through the OLSTEC distribution under its repository-level MIT notice; original attribution headers are retained. |
| PETRELS | `benchmark/petrels/` | PETRELS implementation attributed to Yuejie Chi and modified from GROUSE; OLSTEC wrapper created by Hiroyuki Kasai | Included through the OLSTEC distribution under its repository-level MIT notice; original attribution headers are retained. |
| TeCPSGD | `benchmark/TeCPSGD/` | OLSTEC-package implementation by Hiroyuki Kasai, based on the method of Mardani, Mateos, and Giannakis | OLSTEC repository-level MIT License. |
| CP-WOPT wrapper | `benchmark/CP-WOPT/cp_wopt_mod.m` | OLSTEC-package wrapper by Hiroyuki Kasai; CP-WOPT was introduced by Acar, Dunlavy, Kolda, and Morten Mørup | OLSTEC repository-level MIT License for the wrapper. The Tensor Toolbox and Poblano dependencies have separate terms and are not included. |

Only `admm_srp.m`, `grasta_stream.m`, `grasta_mod.m`, and the corresponding GRASTA license texts are released from the upstream GRASTA directory because these are the files required by the reported experiments. Upstream demonstrations, compiled files, ancillary utilities, and the incomplete `grasta_video_demo_kasai.m` demonstration are not included.

## External Toolboxes Not Redistributed

The reported CP-WOPT runs used the following legacy releases:

- Tensor Toolbox for MATLAB 2.6, authored by Brett W. Bader and Tamara G. Kolda at Sandia National Laboratories.
- Poblano Toolbox 1.1, authored by Daniel M. Dunlavy, Tamara G. Kolda, and Evrim Acar at Sandia National Laboratories.

The license files bundled with these releases grant research and evaluation use but instruct other users to obtain the software from its distribution source. Consequently, `tool/tensor_toolbox_2.6/` and `tool/poblano_toolbox_1.1/` are local installation directories and are not redistributed by this repository. Official project information is available from the [Tensor Toolbox website](https://www.tensortoolbox.org/) and the [Poblano Toolbox website](https://www.sandia.gov/ccr/software/poblano-toolbox/). A different release should not be assumed to reproduce the reported CP-WOPT values without compatibility testing.

## Reference Data

The WAAM-ViD video and metadata used by the real-video experiments are third-party reference data and are not included. They remain subject to the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International license specified by the official dataset record. See `dataset/README.md` for citation and download instructions.
