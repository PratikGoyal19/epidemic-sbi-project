# epidemic-sbi-project

This repository implements a **Simulation-Based Inference (SBI)** framework to estimate parameters () for the SIR epidemic model. This project extends **Homework 4** by comparing **Neural Posterior Estimation (NPE)** against other amortized methods (e.g., NRE or NPSE) using **BayesFlow**.

## 📂 STEP 1: Repository Structure

Before starting, ensure your local folder looks like this. Pratik has already initialized the repository and pushed the simulator.

```text
epidemic-sbi-project/
├── 01_simulator/
│   └── sir_model.py       <-- DONE (Pratik)
├── 02_data/
│   └── generate_data.py   <-- Assignment: Teammate 2
├── 03_training/
│   └── train_models.py    <-- Assignment: Shared (NPE vs NRE)
├── 04_evaluation/
│   └── metrics.py         <-- Assignment: Teammate 3
├── requirements.txt
└── README.md

```

---

## 🚀 STEP 2: Setup Python Environment

```bash
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install bayesflow==1.1.6

```

## 🛠 STEP 3: Configure Git

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"

```

---

## 📋 Assignments & Extension Goals

The core goal is to move beyond basic NPE and perform a **Method Comparison**.

| Phase | Assigned To | Key Tasks |
| --- | --- | --- |
| **Data Gen** | **Teammate 2** | 10k samples; , . |
| **Training** | **Shared** | Compare **NPE** (Invertible Networks) vs. **NRE** (Ratio Estimation). |
| **Evaluation** | **Teammate 3** | MAE, Coverage, and **SBC** (Simulation-Based Calibration). |

---

## ⚖️ Important Rules

1. **PULL FIRST:** `git pull` every time you sit down to code.
2. **ISOLATION:** Only edit your assigned folder.
3. **DAILY PUSH:** Don't wait until the deadline. Commit daily.
4. **CONSULT:** Ask in the group chat before changing `sir_model.py` or `requirements.txt`.

