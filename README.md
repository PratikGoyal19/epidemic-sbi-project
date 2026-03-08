# Epidemic SBI Project

This repository implements a Simulation-Based Inference (SBI) framework to estimate parameters (β, γ) for the SIR epidemic model. This project extends Homework 4 by comparing Neural Posterior Estimation (NPE) against Neural Likelihood Estimation (NLE) using BayesFlow.

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
│   └── metrics.py
├── requirements.txt
└── README.md
```

---

## 👥 Team

| Phase | Assigned To | Status |
|-------|-------------|--------|
| Simulator | Pratik Goyal | ✅ Done |
| Data Generation | Suryansh Chaturvedi | ✅ Done |
| NPE Training | Pratik Goyal | ✅ Done |
| NLE Training | Pratik Goyal | ✅ Done |
| Evaluation (NPE vs NLE) | Mayank Choudhary | ✅ Done |

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
theta shape: (10000, 2)   [beta, gamma]
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
  --num-coupling-layers 4 \
  --normalize-x
```

**Expected outputs:**
```
03_methods/artifacts/nle_checkpoint/
03_methods/artifacts/nle_metrics.json
03_methods/artifacts/nle_normalization.npz
```

---

## 📊 Step 4: Evaluate and Compare NPE vs NLE

```bash
python 04_evaluation/metrics.py
```

This script compares NPE and NLE using:
- **MAE** (Mean Absolute Error) on β and γ
- **Coverage** of 90% credible intervals
- **Posterior recovery plots**
- **SBC** (Simulation-Based Calibration)

---

## 🔬 SIR Model

The SIR model divides the population into three compartments:

```
dS/dt = -β · S · I / N
dI/dt =  β · S · I / N - γ · I
dR/dt =  γ · I
```

| Parameter | Range | Description |
|-----------|-------|-------------|
| β (beta) | [0.10, 0.60] | Infection rate |
| γ (gamma) | [0.01, 0.10] | Recovery rate |
| R₀ = β/γ | [1.0, 60.0] | Basic reproduction number |
| N | 10,000 | Total population |
| T | 160 days | Simulation length |

---

## ⚖️ Rules

1. **PULL FIRST:** `git pull` every time you sit down to code
2. **ISOLATION:** Only edit your assigned folder
3. **DAILY PUSH:** Commit and push daily
4. **CONSULT:** Ask in the group chat before changing `sir_model.py` or `requirements.txt`
