# Epidemic SBI Project

This repository implements a Simulation-Based Inference (SBI) framework to estimate parameters (β, γ, I₀) for the SIR epidemic model. This project extends Homework 4 by comparing Neural Posterior Estimation (NPE) against Neural Likelihood Estimation (NLE) using BayesFlow.

---

## 📂 Repository Structure

```
epidemic-sbi-project/
├── 01_simulator/
│   └── sir_model.py
├── 02_data/
│   ├── generate_data.py
│   ├── test_generation.py
│   └── sir_dataset.npz
├── 03_methods/
│   ├── train_npe.py
│   ├── train_nle.py
│   ├── test_npe.py
│   └── artifacts/
│       ├── npe_checkpoint/
│       ├── nle_checkpoint/
│       ├── npe_metrics.json
│       ├── nle_metrics.json
│       └── nle_normalization.npz
├── 04_evaluation/
│   ├── metrics.py
│   ├── real_data.py
│   └── results/
│       ├── comparison_metrics.json
│       ├── npe_posterior_recovery.png
│       ├── nle_posterior_recovery.png
│       ├── npe_vs_nle_comparison.png
│       ├── npe_sbc.png
│       ├── nle_sbc.png
│       ├── metrics_summary.png
│       └── real_data/
│           ├── italy_inference_results.json
│           ├── real_data_posteriors.png
│           ├── real_data_predictive.png
│           └── real_data_npe_vs_nle.png
├── requirements.txt
└── README.md
```

---

## 👥 Team

| Phase | Assigned To | Status |
|-------|-------------|--------|
| Simulator | Pratik Goyal | ✅ Done |
| Data Generation | Suryansh Chaturvedi | ✅ Done |
| NPE Training | Mayank Choudhary | ✅ Done |
| NLE Training | Suryansh Chaturvedi | ✅ Done |
| Evaluation (NPE vs NLE) | Pratik Goyal | ✅ Done |
| Real Data Inference (Italy COVID-19) | Pratik Goyal | ✅ Done |

---

## 🚀 Setup

```bash
# Clone the repository
git clone https://github.com/PratikGoyal19/epidemic-sbi-project.git
cd epidemic-sbi-project

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install bayesflow==1.1.6
```

---

## ▶️ Step 1: Generate Dataset

```bash
python 02_data/generate_data.py --n-samples 10000 --out 02_data/sir_dataset.npz
```

**Expected output:**
```
02_data/sir_dataset.npz   (~3 MB)
theta shape: (10000, 3)   [beta, gamma, I0]
x shape:     (10000, 160) [infected counts over 160 days]
```

**Quick test first (100 samples, ~5 seconds):**
```bash
python 02_data/test_generation.py
```

---

## 🧠 Step 2: Train NPE (Neural Posterior Estimation)

```bash
python 03_methods/train_npe.py
```

**Optional overrides:**
```bash
python 03_methods/train_npe.py \
  --data 02_data/sir_dataset.npz \
  --epochs 50 \
  --batch-size 256 \
  --hidden-dim 128 \
  --num-coupling-layers 4 \
  --summary-dim 32
```

**Expected outputs:**
```
03_methods/artifacts/npe_checkpoint/
03_methods/artifacts/npe_metrics.json
Runtime: ~15 minutes
```

---

## 🧠 Step 3: Train NLE (Neural Likelihood Estimation)

```bash
python 03_methods/train_nle.py
```

**Optional overrides:**
```bash
python 03_methods/train_nle.py \
  --data 02_data/sir_dataset.npz \
  --epochs 150 \
  --batch-size 256 \
  --lr 5e-4 \
  --hidden-dim 128 \
  --num-coupling-layers 4
```

**Expected outputs:**
```
03_methods/artifacts/nle_checkpoint/
03_methods/artifacts/nle_metrics.json
03_methods/artifacts/nle_normalization.npz
Runtime: ~4 minutes
```

---

## 📊 Step 4: Evaluate and Compare NPE vs NLE

```bash
python 04_evaluation/metrics.py
```

Compares NPE and NLE on 200 held-out test samples (500 posterior samples each) using:
- **MAE** and **RMSE** on β, γ, and I₀
- **Coverage** of 50% and 90% credible intervals
- **Posterior recovery plots**
- **SBC** (Simulation-Based Calibration) rank histograms

**Key results (200 test samples, 500 posterior samples):**
```
Metric                NPE        NLE
─────────────────────────────────────
MAE beta            0.0068     0.0672   (NPE 10x better)
MAE gamma           0.0013     0.0229   (NPE 18x better)
MAE I0              1.8620     2.2245   (NPE better)
90% Coverage beta    99.0%     94.5%
90% Coverage gamma   95.0%     86.0%
90% Coverage I0      96.0%     85.5%
```

**Outputs saved to** `04_evaluation/results/`

---

## 🦠 Step 5: Real Data Inference (Italy COVID-19 First Wave)

```bash
python 04_evaluation/real_data.py
```

Applies both trained models to Italy's COVID-19 first wave (Feb 23 – Jul 31, 2020) using the Our World in Data dataset (not included in repo — download separately).

**Key results:**
```
Parameter   NPE Mean   NPE 90% CI         NLE Mean   NLE 90% CI
────────────────────────────────────────────────────────────────
beta         0.2718    [0.24, 0.30]        0.1403    [0.11, 0.20]
gamma        0.1713    [0.15, 0.19]        0.0560    [0.03, 0.10]
I0           2.875     [~0, 5.90]          1.130     [1.03, 1.36]
R0           1.59      [1.48, 1.72]        3.36      [1.14, 5.25]
```

NPE gives tight, confident estimates. NLE gives wider, more uncertain posteriors.

**Outputs saved to** `04_evaluation/results/real_data/`

> **Note:** The OWID COVID dataset (`owid-covid-data.csv`) is excluded from the repo due to its size (93 MB). Download it from [Our World in Data](https://ourworldindata.org/covid-cases) and place it in `02_data/`.

---

## 🔬 SIR Model

The SIR model divides the population into three compartments:

```
dS/dt = -β · S · I / N
dI/dt =  β · S · I / N - γ · I
dR/dt =  γ · I
```

All three parameters are inferred jointly:

| Parameter | Range | Description |
|-----------|-------|-------------|
| β (beta) | [0.10, 0.60] | Infection rate |
| γ (gamma) | [0.01, 0.10] | Recovery rate |
| I₀ | [1, 50] | Initial infected individuals |
| R₀ = β/γ | [1.0, 60.0] | Basic reproduction number |
| N | 10,000 | Total population (fixed) |
| T | 160 days | Simulation length |

---

## ⚖️ Rules

1. **PULL FIRST:** `git pull` every time you sit down to code
2. **ISOLATION:** Only edit your assigned folder
3. **DAILY PUSH:** Commit and push daily
4. **CONSULT:** Ask in the group chat before changing `sir_model.py` or `requirements.txt`