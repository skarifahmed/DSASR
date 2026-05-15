# 🚀 DSASR: Dual-Stream Adaptive Sequential Recommendation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)
![Status](https://img.shields.io/badge/Status-Research-orange.svg)

### Adaptive Sequential Recommendation with Dual-Stream Causal Transformers

</div>

---

## 📌 Overview

Official implementation of the paper:

> **Dual-Stream Adaptive Sequential Recommendation (DSASR)**  
> Swastika Ghosh, Rohini Basak, Sourav Mandal, Dilip K. Prasad, Arif Ahmed Sekh

DSASR is a transformer-based sequential recommendation framework that models:

- ⚡ Short-term behavioral intent
- 🧠 Long-term stable preference

through a dual-stream causal transformer architecture with adaptive preference fusion.

---

## ✨ Key Features

- 🔹 Dual-stream causal transformer encoders
- 🔹 Adaptive consistency-gated fusion
- 🔹 Preference drift modeling
- 🔹 Demographic-aware personalization
- 🔹 BCE + BPR hybrid optimization
- 🔹 Sequential recommendation for sparse and dense datasets

---

## 📂 Supported Datasets

- 🎬 MovieLens 100K
- 🎬 MovieLens 1M
- 🛍️ Amazon Beauty

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/DSASR.git

cd DSASR

pip install -r requirements.txt
