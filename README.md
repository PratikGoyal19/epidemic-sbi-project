# epidemic-sbi-project

This repository implements a **Simulation-Based Inference (SBI)** framework to estimate parameters () for the SIR epidemic model. Per instructor feedback, this project extends Homework 4 by comparing **Neural Posterior Estimation (NPE)** against other amortized methods (e.g., NRE or NPSE) using **BayesFlow**.

## 🚀 Step 2: Setup Python Environment

Ensure your local environment is consistent across the team:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
pip install bayesflow==1.1.6

```

## 🛠 Step 3: Configure Git

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"

```

---

## 📋 Team Assignments & Workflow

The core simulator (`01_simulator/sir_model.py`) is **complete** and pushed by Pratik.

| File Path | Assigned To | Key Tasks |
| --- | --- | --- |
| `02_data/generate_data.py` | **Teammate 2** | Generate 10,000 samples. Parameters: , . Save as `training_data.npz`. |
| `03_training/` | **Shared/Lead** | **New Requirement:** Train both NPE (Invertible Neural Networks) and a secondary method (e.g., NRE) for comparison. |
| `04_evaluation/metrics.py` | **Teammate 3** | Implement MAE, Credible Interval Coverage, and **Comparative Plots** (NPE vs Other Method). |

---

## ⚖️ Important Rules

* **SYNC FIRST:** Always run `git pull` before you start work.
* **STAY IN YOUR ZONE:** * Teammate 2: **Only** edit `02_data/`
* Teammate 3: **Only** edit `04_evaluation/`


* **DAILY UPDATES:** Commit and push your progress daily. Small updates prevent massive merge conflicts.
* **METHOD COMPARISON:** Ensure evaluation scripts can handle results from multiple model types.

## 📚 References

* **Simulator Logic:** SIR Model (Homework 4 Extension).
* **Comparison Framework:** Lueckmann et al. (2021) — *Benchmarking Simulation-Based Inference*.
* **Data Visuals:** Chatha et al. (2024).
