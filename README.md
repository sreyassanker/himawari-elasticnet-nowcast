# Sparse Elastic Net Linear Regression for Geostationary Infrared Brightness Temperature Nowcasting: A Multi-Year Benchmark Using Himawari-9 AHI

[![Journal](https://img.shields.io/badge/Journal-Atmospheric%20Research%20(Under%20Review)-blue.svg)](https://www.sciencedirect.com/journal/atmospheric-research)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: NOAA AWS](https://img.shields.io/badge/Data-NOAA%20Himawari--9%20S3-orange.svg)](https://registry.opendata.aws/noaa-himawari/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-brightgreen.svg)](results/benchmark_metrics.json)

<div align="justify">

**Sreyas S S**<sup>a,b</sup>, **Saied Pirasteh**<sup>a,c,*</sup>, **Dhanaraj K**<sup>b</sup>, **Mansour Almazroui**<sup>e,f</sup>, **Mary-Jane Morongwa Bopape**<sup>d</sup>  

<sup>a</sup> Institute of Artificial Intelligence, Shaoxing University, 508 West Huancheng Road, Yuecheng District, Zhejiang Province 312000, China  
<sup>b</sup> Amity Institute of Remote Sensing & Geoinformatics, Amity University, Noida, Uttar Pradesh 201313, India  
<sup>c</sup> Department of Geotechnics and Geomatics, Saveetha School of Engineering, Saveetha Institute of Medical and Technical Sciences, Chennai, Tamil Nadu, India  
<sup>d</sup> Unisa Center for Climate Systems Science and Applications, Department of Environmental Sciences, College of Agriculture and Environmental Sciences, University of South Africa (UNISA), Florida Science Campus, Johannesburg 1709, South Africa  
<sup>e</sup> Center of Excellence for Climate Change Research (CECCR) / Department of Meteorology, King Abdulaziz University, Jeddah 21589, Saudi Arabia  
<sup>f</sup> Climatic Research Unit, School of Environmental Sciences, University of East Anglia, Norwich NR4 7TJ, United Kingdom  
<sup>*</sup> **Corresponding author:** Saied Pirasteh, spirasteh71@gmail.com

---

## Executive Summary

Short-term forecasting (nowcasting, 20–90 min) of cloud-top infrared (IR) brightness temperature (BT) from geostationary satellites is essential for operational convective tracking, cloud microphysical monitoring, and aviation safety warnings. While modern deep-learning models (e.g., 3D-CNNs, U-Nets, diffusion models) are increasingly applied to satellite nowcasting, their baseline performance is rarely benchmarked against simple, transparent, and physically interpretable linear formulations across multi-year observational scales.

This repository provides the complete, fully reproducible codebase, data pipelines, model artifacts, and figure generation scripts for our multi-year benchmark submitted to ***Atmospheric Research*** (Elsevier). 

Using an exact coordinate descent solver over precomputed Gram matrices, we establish **72 sparse regularized linear equations** (Elastic Net: 85% $L_1$ sparsity penalty, 15% $L_2$ grouping penalty) predicting brightness temperatures across all nine thermal IR bands (B08–B16, 6.2–13.3 µm) of the Advanced Himawari Imager (AHI) aboard Himawari-9 across 8 forecast lead times (20–90 min at 10-minute intervals).

```
State Vector: x(t) = [T_B08(t), T_B09(t), ..., T_B16(t)]^T  ∈  R^9
Target Scalar: y(t + Δt) = T_b(t + Δt)  for band b ∈ {B08..B16}, Δt ∈ {20..90 min}
```

---

## Key Highlights & Performance

- **Multi-Year Training Scale:** Fitted on **373 full days** spanning all four consecutive years of **2023–2026** (~50.3 million valid observation pairs per lead time) sampled across South Asia (5°–35°N, 65°–95°E).
- **Cross-Domain Spatial Transfer:** Evaluated without retraining across three independent geographic validation regimes (368 full days each across 2023–2026, totaling ~150M independent test observations): mid-latitude frontal dynamics (**East Asia**), deep equatorial convection (**Maritime Continent**), and subtropical arid systems (**Australia**).
- **Physical Regularization & Grouping Stability:** The strictly convex Elastic Net $L_2$ penalty mathematically guarantees grouping stability among collinear window channels (e.g., B13/B14 and B14/B15), completely resolving channel-swapping instability while enforcing the 95% OLS $R^2$ retention rule. Equations retain an average of **3.17 nonzero predictors** out of 9.
- **Ultra-Low Operational Latency:** On commodity single-core CPU hardware (3.2 GHz), evaluating all 9 bands for a $1{,}000 \times 1{,}000$ regional grid requires **14.3 ms**; evaluating an entire full-disk scene (~30.25 million pixels) takes **0.43 s**, enabling real-time operational deployment without dedicated GPU acceleration.

### Benchmark Results Summary

| Evaluation Regime | Dataset Scope | Lead: 20 min (R² / RMSE) | Lead: 60 min (R² / RMSE) | Lead: 90 min (R² / RMSE) |
|---|---|:---:|:---:|:---:|
| **In-Sample Multi-Year Training (South Asia)** | 2023–2026 (373 days, 50.3M pairs) | **0.918** / 4.86 K | **0.790** / 7.80 K | **0.718** / 9.03 K |
| **Out-of-Domain: East Asia** | 2023–2026 (368 days, ~50M pairs) | **0.895** / 4.98 K | **0.758** / 7.52 K | **0.680** / 8.63 K |
| **Out-of-Domain: Australia** | 2023–2026 (368 days, ~50M pairs) | **0.906** / 4.68 K | **0.762** / 7.43 K | **0.669** / 8.73 K |
| **Out-of-Domain: Maritime Continent** | 2023–2026 (368 days, ~50M pairs) | **0.874** / 6.69 K | **0.677** / 10.68 K | **0.571** / 12.29 K |

*All values derive strictly from `results/benchmark_metrics.json` and `results/validation_results.json`.*

---

## Himawari-9 AHI Multispectral Infrared Channels

The Advanced Himawari Imager (AHI) carries 9 thermal infrared channels (Bands 8–16) utilized in this study:

| Band | Central λ (µm) | Spectral Region | Primary Atmospheric / Surface Target | Operational Predictability |
|:---:|:---:|:---:|:---|:---:|
| **B08** | 6.2 | Upper-level water vapor | Upper-tropospheric jet dynamics, cirrus | **Highest** (R² = 0.947 @ 20m; pure self-persistence) |
| **B09** | 6.9 | Mid-level water vapor | Middle-tropospheric moisture transport | High (R² = 0.933 @ 20m) |
| **B10** | 7.3 | Lower-level water vapor | Low-to-mid tropospheric moisture boundaries | Moderate-High (R² = 0.919 @ 20m) |
| **B11** | 8.6 | Cloud-phase window | Cloud thermodynamic phase, microphysics | Moderate (R² = 0.907 @ 20m; fastest decay across leads) |
| **B12** | 9.6 | Ozone absorption | Stratosphere-troposphere exchange, tropopause | High in-sample (R² = 0.918 @ 20m) |
| **B13** | 10.4 | Clean surface IR window | Surface temperature, deep convective cloud tops | Sounder-assisted handoff at longer leads (R² = 0.909 @ 20m) |
| **B14** | 11.2 | Standard IR window | Cloud-top height, convective development | High cross-coupling with B13 and B15 (R² = 0.911 @ 20m) |
| **B15** | 12.4 | "Dirty" IR window | Low-level moisture attenuation, split-window | Key predictor for surface window channels (R² = 0.912 @ 20m) |
| **B16** | 13.3 | CO₂ absorption | Cloud-top pressure, vertical atmospheric sounding | Secondary sounder constraint (R² = 0.908 @ 20m) |

---

## Mathematical Formulation

The multispectral brightness temperature evolution is modeled by regularized coordinate descent minimizing the penalized squared-error loss in standardized predictor space:

$$
\min_{\boldsymbol{\beta}} \left[ \frac{1}{2N} \left\| \tilde{\mathbf{y}} - \tilde{\mathbf{X}} \boldsymbol{\beta} \right\|_2^2 + \lambda \left( \alpha \|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2} \|\boldsymbol{\beta}\|_2^2 \right) \right]
$$

where:
- $\tilde{\mathbf{X}} \in \mathbb{R}^{N \times 9}$ represents the standardized nine-band infrared predictors at time $t$ ($\mu=0, \sigma=1$).
- $\tilde{\mathbf{y}} \in \mathbb{R}^N$ represents the standardized scalar brightness temperature for band $b$ at forecast lead $t + \Delta t$.
- $\alpha = 0.85$ sets the mixing proportion between $L_1$ sparsity (LASSO) and $L_2$ shrinkage (Ridge), corresponding to `L1_RATIO = 0.85` in the codebase.
- The regularization parameter $\lambda$ (corresponding to `alpha` in the code's parameter grid) is selected to maximize sparsity subject to the **95% OLS $R^2$ retention rule**:

$$
\boldsymbol{\beta}^* = \arg\min_{\boldsymbol{\beta}(\lambda)} \|\boldsymbol{\beta}(\lambda)\|_0 \quad \text{subject to} \quad R^2(\boldsymbol{\beta}(\lambda)) \ge 0.95 \cdot R^2_{\text{OLS}}
$$

### Mapping to Physical Kelvin Space
To enable direct zero-overhead operational evaluation, the standardized sparse coefficients $\boldsymbol{\beta}^*$ are transformed back into physical Kelvin units:

$$
T_b(t + \Delta t) = \beta_0 + \sum_{j \in \mathcal{S}} \beta_j T_j(t)
$$

where the physical slopes $\beta_j$ and intercept $\beta_0$ are computed analytically:

$$
\beta_j = \beta_j^* \left( \frac{\sigma_{y}}{\sigma_{x_j}} \right), \quad \beta_0 = \mu_y - \sum_{j \in \mathcal{S}} \beta_j \mu_{x_j}
$$

and $\mathcal{S} = \{j \mid \beta_j^* \ne 0\}$ denotes the sparse active set of selected predictor channels (retaining an average of 3.17 bands out of 9).

Because the sample size across the multi-year archive is large ($N \approx 5.03 \times 10^7$ observations per lead), exact coordinate descent is executed directly over the precomputed standardized Gram matrix $\mathbf{G} = \tilde{\mathbf{X}}^T \tilde{\mathbf{X}} \in \mathbb{R}^{9 \times 9}$ and cross-product correlation vector $\tilde{\mathbf{X}}^T \tilde{\mathbf{y}} \in \mathbb{R}^9$, converging to the global minimum in **< 0.1 ms per equation**.

---

## Repository Architecture

```
himawari_nowcast/
├── LICENSE                                # MIT Open-Source License
├── README.md                              # Master technical & operational documentation
├── requirements.txt                       # Python environment dependencies
│
├── src/                                   # Core analysis & modeling pipeline (run from repo root)
│   ├── hsd_reader.py                      # Low-level JMA binary HSD decoder & Planck converter
│   ├── scan_training_days.py              # Atmospheric convective activity scanner for candidate dates
│   ├── precompute_pixels.py               # Balanced stratified pixel grid generator (seed 42, 1000 pixels)
│   ├── extract_training_days.py           # Feature cube extractor downloading cubes from NOAA AWS S3
│   ├── extract_validation_days.py         # Geographic validation cube extractor (East Asia, Maritime, Australia)
│   ├── train_nowcast.py                   # Multi-year Elastic Net solver over precomputed Gram matrices
│   ├── validate_nowcast.py                # Multi-domain evaluation benchmark across held-out domains
│   ├── evaluate_retention_sensitivity.py  # Sensitivity benchmark across OLS retention thresholds (90%, 95%, 99%)
│   └── compute_benchmark_metrics.py      # Statistical metric aggregator for text and tables
│
├── data/                                  # Satellite data directories (*.npz excluded from git via .gitignore)
│   ├── pixel_locations.npz                # Precomputed coordinates & Himawari segment assignments for 1000 pixels
│   ├── training/                          # Extracted South Asia training cubes (373 full days: 2023–2026)
│   └── validation/                        # Extracted geographic validation cubes (1,104 files: 3 domains × 368 days)
│
├── results/                               # Numerical results & model artifacts (JSON format, git-tracked)
│   ├── model_coefficients.json            # All 72 fitted equations (intercepts and coefficients in Kelvin)
│   ├── train_diagnostics.json             # Convergence diagnostics, sample counts, and R² retention
│   ├── validation_results.json            # Empirical validation metrics across lead times and regimes
│   ├── retention_sensitivity.json         # Retention threshold sensitivity metrics and predictor counts
│   └── benchmark_metrics.json             # Consolidated benchmark metrics for manuscript tables/text
│
└── figures/                               # Manuscript figures & generation scripts
    ├── Fig2.drawio                        # Editable XML vector diagram for Manuscript Fig. 2 (draw.io)
    ├── fig1_study_area.py                 # Generator for Manuscript Fig. 1
    ├── fig3_r2_decay.py                   # Generator for Manuscript Fig. 3
    ├── fig4_heatmap_all_leads.py          # Generator for Manuscript Fig. 4
    ├── fig5_band_r2.py                    # Generator for Manuscript Fig. 5
    ├── fig6_influence_matrix.py           # Generator for Manuscript Fig. 6
    ├── fig7_r2_validation.py              # Generator for Manuscript Fig. 7
    ├── fig8_rmse_validation.py            # Generator for Manuscript Fig. 8
    ├── fig9_per_band_r2.py                # Generator for Manuscript Fig. 9
    ├── Fig1.png / Fig1.pdf                # Manuscript Fig. 1: Full-disk coverage & study domains
    ├── Fig2.png / Fig2.pdf                # Manuscript Fig. 2: End-to-end methodology overview flowchart
    ├── Fig3.png / Fig3.pdf                # Manuscript Fig. 3: In-sample training R² decay vs lead time
    ├── Fig4.png / Fig4.pdf                # Manuscript Fig. 4: Per-band R² heatmaps across leads (4 domains)
    ├── Fig5.png / Fig5.pdf                # Manuscript Fig. 5: Per-band R² at 20-min & 90-min leads
    ├── Fig6.png / Fig6.pdf                # Manuscript Fig. 6: Cross-band influence matrix at 20-min lead
    ├── Fig7.png / Fig7.pdf                # Manuscript Fig. 7: Mean R² decay across validation domains
    ├── Fig8.png / Fig8.pdf                # Manuscript Fig. 8: RMSE vs lead time by validation domain (9 panels)
    └── Fig9.png / Fig9.pdf                # Manuscript Fig. 9: Per-band R² decay curves (9 panels)
```

---

## Manuscript Figures and Scripts

Each script in `figures/` generates its corresponding figure:

| Manuscript Figure | Figure Subject / Scientific Scope | Generating Script / Source | Output Artifacts |
|:---:|:---|:---|:---|
| **Fig. 1** | Himawari-9 full-disk coverage, segment layout, and geographic study domains | [`figures/fig1_study_area.py`](figures/fig1_study_area.py) | `figures/Fig1.png`, `figures/Fig1.pdf` |
| **Fig. 2** | End-to-end nowcasting methodology & operational workflow | [`figures/Fig2.drawio`](figures/Fig2.drawio) (Diagram export) | `figures/Fig2.png`, `figures/Fig2.pdf` |
| **Fig. 3** | In-sample training R² decay curves across 20–90 min lead times | [`figures/fig3_r2_decay.py`](figures/fig3_r2_decay.py) | `figures/Fig3.png`, `figures/Fig3.pdf` |
| **Fig. 4** | Per-band R² heatmaps across all leads and 4 evaluation domains | [`figures/fig4_heatmap_all_leads.py`](figures/fig4_heatmap_all_leads.py) | `figures/Fig4.png`, `figures/Fig4.pdf` |
| **Fig. 5** | Per-band in-sample R² performance at 20-min and 90-min leads | [`figures/fig5_band_r2.py`](figures/fig5_band_r2.py) | `figures/Fig5.png`, `figures/Fig5.pdf` |
| **Fig. 6** | Cross-band influence matrix and non-zero coefficient coupling at 20-min lead | [`figures/fig6_influence_matrix.py`](figures/fig6_influence_matrix.py) | `figures/Fig6.png`, `figures/Fig6.pdf` |
| **Fig. 7** | Mean R² decay comparisons across geographic validation domains | [`figures/fig7_r2_validation.py`](figures/fig7_r2_validation.py) | `figures/Fig7.png`, `figures/Fig7.pdf` |
| **Fig. 8** | Root-Mean-Square Error (RMSE) vs lead time across validation domains (9 panels) | [`figures/fig8_rmse_validation.py`](figures/fig8_rmse_validation.py) | `figures/Fig8.png`, `figures/Fig8.pdf` |
| **Fig. 9** | Per-band R² decay curves comparing training vs validation domains (9 panels) | [`figures/fig9_per_band_r2.py`](figures/fig9_per_band_r2.py) | `figures/Fig9.png`, `figures/Fig9.pdf` |

---

## Reproduction Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/sreyassanker/himawari-elasticnet-nowcast.git
cd himawari-elasticnet-nowcast

# Create and activate a clean Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Acquisition Pipeline

The Himawari-9 AHI level-1b Himawari Standard Data (HSD) segments are hosted on the public NOAA AWS open-data archive (`s3://noaa-himawari9`) and require no API keys or access requests:

```bash
# Step 1: Screen candidate dates for deep convective signatures
python src/scan_training_days.py

# Step 2: Generate the balanced stratified 1,000-pixel observation coordinate grid
python src/precompute_pixels.py

# Step 3: Extract multi-year training feature cubes from NOAA S3 archive
python src/extract_training_days.py <YYYYMMDD> [<YYYYMMDD> ...]

# Step 4: Extract held-out geographic validation cubes
python src/extract_validation_days.py
```

### 3. Model Training & Evaluation

```bash
# Step 5: Fit 72 Elastic Net equations using Gram-matrix coordinate descent
python src/train_nowcast.py
# Outputs: results/model_coefficients.json, results/train_diagnostics.json

# Step 6: Validate frozen equations across held-out geographic domains (East Asia, Australia, Maritime)
python src/validate_nowcast.py
# Outputs: results/validation_results.json

# Step 7: Compute consolidated statistics and verify consistency against manuscript
python src/compute_benchmark_metrics.py
# Outputs: results/benchmark_metrics.json

# Optional: Run sensitivity analysis across OLS retention thresholds (90%, 95%, 99%)
python src/evaluate_retention_sensitivity.py
# Outputs: results/retention_sensitivity.json
```

### 4. Manuscript Figure Generation

Generate all manuscript figures (600 DPI PNG and vector PDF; Fig. 2 methodology overview at 1200 DPI):

```bash
python figures/fig1_study_area.py           # Generates Fig. 1 (Full-disk coverage & domains)
# Fig. 2 is exported from figures/Fig2.drawio (Methodology overview flowchart)
python figures/fig3_r2_decay.py             # Generates Fig. 3 (In-sample training R² decay)
python figures/fig4_heatmap_all_leads.py    # Generates Fig. 4 (Per-band R² heatmaps across leads)
python figures/fig5_band_r2.py              # Generates Fig. 5 (Per-band R² at 20-min & 90-min leads)
python figures/fig6_influence_matrix.py     # Generates Fig. 6 (Cross-band influence matrix)
python figures/fig7_r2_validation.py        # Generates Fig. 7 (Mean validation R² decay curves)
python figures/fig8_rmse_validation.py      # Generates Fig. 8 (RMSE vs lead time by domain)
python figures/fig9_per_band_r2.py          # Generates Fig. 9 (Per-band R² decay: train vs validation)
```

---

## Data Availability & Remote Sensing Archive

Himawari-9 AHI Level-1b Full-Disk imagery is publicly accessible without restriction under NOAA Open Data terms:

<div align="left">

- **AWS S3 Bucket:** `s3://noaa-himawari9/AHI-L1b-FLDK/`
- **Registry of Open Data on AWS:** [https://registry.opendata.aws/noaa-himawari/](https://registry.opendata.aws/noaa-himawari/)
- **Japan Meteorological Agency (JMA) Meteorological Satellite Center:** [https://www.data.jma.go.jp/mscweb/en/](https://www.data.jma.go.jp/mscweb/en/)

</div>

---

## Citation

If you utilize this benchmark, model formulations, or analysis code in your research, please cite our study:

```bibtex
@article{sreyas2026himawari_elasticnet,
  title   = {Sparse Elastic Net Linear Regression for Geostationary Infrared Brightness Temperature Nowcasting: A Multi-Year Benchmark Using Himawari-9 AHI},
  author  = {Sreyas, S. S. and Pirasteh, Saied and Dhanaraj, K. and Almazroui, Mansour and Bopape, Mary-Jane Morongwa},
  journal = {Atmospheric Research},
  year    = {2026},
  note    = {Under Review}
}
```

---

## License

This software and analysis pipeline are licensed under the [MIT License](LICENSE). 
Copyright (c) 2026 Sreyas S S et al.

</div>
