# Multi-Modal Chest X-Ray Classification & Report Generation 🩻🤖

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/Architecture-Transformers-orange)
![Medical Imaging](https://img.shields.io/badge/Domain-Medical%20Imaging-green)

## 📌 Project Overview
This repository contains a unified, end-to-end deep learning framework for automated chest X-ray analysis. It integrates **multi-label disease classification** with **natural language report generation** using a decoupled vision-language pipeline.

The system was evaluated on the Indiana University Chest X-Ray Collection and specifically addresses extreme class imbalances (47:1) and the sequential bottleneck of traditional RNNs in medical text generation.

## 🔬 Methodology & Architecture

The architecture consists of a two-stage decoupled pipeline:

1. **Vision Module (Disease Classification):**
   * **Model:** `DenseNet-169` (pretrained on ImageNet, 169 layers, 1,664-dim features).
   * **Optimization:** Implemented a custom **Weighted Focal Loss** with square-root dampening to mitigate severe class imbalance. 
   * **Training Strategy:** Progressive layer unfreezing (Epochs 1-5 frozen features; Epochs 6-20 unfreezing DenseBlock4).
   * **Output:** Predicts presence of 4 critical pathologies: Opacity, Cardiomegaly, Pulmonary Atelectasis, and Pleural Effusion.

2. **Language Module (Report Generation):**
   * **Models Compared:** LSTM with Bahdanau Attention vs. Multi-Head Transformer Decoder.
   * **Mechanism:** Conditioned on the predicted disease labels from the Vision Module to ensure clinical interpretability.
   * **Decoding:** Utilized Beam Search (width=3) to optimize long-range sequence dependencies.

## 📊 Dataset
* **Source:** Indiana University Chest X-Ray Collection.
* **Pre-processing:** Systematic filtering isolated 3,847 frontal-view (PA/AP) radiographs.
* **Challenges:** Extreme data imbalance (75.2% normal vs. 1.6% pleural effusion).

## 🏆 Key Results

Our ablation studies proved that Transformer self-attention mechanisms heavily outperform recurrent architectures (LSTM) in capturing long-range clinical dependencies.

### Classification Performance (DenseNet-169)
| Metric | Score |
| :--- | :--- |
| **Micro F1-Score** | **0.7403** |
| Normal F1 | 0.89 |
| Precision (Micro) | 0.73 |
| Recall (Micro) | 0.75 |

### Generation Performance (Transformer vs LSTM)
| Architecture | Decoding Strategy | BLEU-4 | ROUGE-L | Inference Time |
| :--- | :--- | :--- | :--- | :--- |
| LSTM + Attention | Beam Search (w=3) | 0.0184 | 0.2654 | 0.31s |
| **Transformer (4-Head)** | **Beam Search (w=3)** | **0.0230** | **0.2990** | **0.24s** |

* **Transformer Improvements:** Achieved a **+25.0% improvement in BLEU-4** and **+12.7% in ROUGE-L** over the LSTM baseline.
* **Efficiency:** Accelerated training convergence by 14% and inference speed by **22.6%**, resolving the O(n) path length bottleneck of LSTMs.

## 🛠️ Usage & Installation

```bash
# Clone the repository
git clone https://github.com/LamiaT/chest-xray-multimodal-generation.git
cd chest-xray-multimodal-generation

# Install dependencies
pip install -r requirements.txt
