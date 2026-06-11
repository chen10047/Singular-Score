Supplementary Materials

## Overview
This material contains the implementation of our proposed zero-cost proxy for neural architecture search (NAS) and results on standard benchmark.
The files `zc_transbench101_macro.json` and `zc_transbench101_micro.json` are sourced from the NAS-Bench-Suite-Zero(Krishnakumar et al., 2022).

For other search spaces, additional data and benchmarks are available from the following sources:

- **zero-cost-pt**:  "Zero‑cost operation scoring in differentiable architecture search" (Xiang et al., 2023) and "MECO: Zero‑shot NAS with one data and single forward pass via minimum eigenvalue of correlation" (Jiang et al., 2023).

- **TransNas‑Bench‑101**: "TransNas‑Bench‑101: Improving transferability and generalizability of cross‑task neural architecture search" (Krishnakumar et al., 2022).

- **MedMNIST v2**: "Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification"(Yang et al., 2023) and "MedNAS: Multiscale training‑free neural architecture search for medical image analysis" (Wang et al., 2024).

### NAS-Bench-201  
- Reference: "NAS-BENCH-201: Extending the Scope of Reproducible Neural Architecture Search"
- Consistent across CIFAR-10, CIFAR-100, and ImageNet16-120
- The implementation is based on **AZ-NAS**: "Assembling Zero‑cost proxies for Network Architecture Search" (Lee & Ham, 2024).

## Compared Zero-Cost Proxies

Our method is compared against:

1. **SNIP** (Lee et al. 2019, ABDELFATTAH ET AL., 2021)
2. **Gradnorm** (Abdelfattah et al. 2021)
3. **Grasp** (Wang et al. 2020, ABDELFATTAH ET AL., 2021)
4. **Synflow** (Tanaka et al. 2020, ABDELFATTAH ET AL., 2021)
5. **NASWOT** (Mellor et al. 2021)
6. **TE-NAS** (Chen et al. 2021)
7. **Zen-NAS** (Lin et al. 2021)
8. **Entropic** (Cavagnero et al. 2023)
9. **Zico** (Li et al. 2023)
10. **AZE(Expressivity in AZ-NAS)** (Lee et al. 2024)

## Implementation

### Our Proposed Method
- File: compute_singular_score.py
- Results in logs

## Usage

Evaluate on NAS-Bench-201:
    bash eval.sh