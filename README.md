# Robustness Enhancement in Image Classification via Data Augmentation

This repository contains the complete experimental framework, source code, data summaries, and academic deliverables for our graduate Machine Learning project (**DASC 5304** at the University of Texas at Arlington). 

Our research systematically investigates the empirical trade-offs between clean test accuracy and out-of-distribution robustness to natural corruptions by evaluating classical, mix-based, and generative data augmentation strategies.

---

## 👥 Authors
* **Pritesh Das** – Department of Computer Science and Engineering, The University of Texas at Arlington
* **Jagriti Koirala** – Department of Computer Science and Engineering, The University of Texas at Arlington

---

## 📝 Project Abstract
Deep convolutional neural networks achieve excellent classification performance on pristine, clean datasets, but they frequently experience significant accuracy degradation when subjected to distribution shifts and natural corruptions (e.g., noise, blur, environmental elements). In this paper, we study the effect of different augmentation techniques on both clean test accuracy and corruption robustness. Using a **ResNet-18** backbone, we benchmark four strategies across the **CIFAR-10**, **CIFAR-100**, and **SVHN** datasets, evaluating out-of-distribution resilience on the **CIFAR-10-C** benchmark across 19 corruption types and 5 distinct severities.

Our experimental timeline consisted of distinct evolutionary phases:
1. **Midway Phase:** Conducted short training cycles (5–10 epochs) with a strong Mixup configuration ($\alpha = 1.0$), which severely penalized clean test accuracy and offered negligible robustness gains.
2. **Final Phase:** Expanded the training cycle (15–20 epochs) utilizing **Cosine Annealing** learning rate scheduling and a tuned, milder Mixup setting ($\alpha = 0.2$). 

Ultimately, our findings revealed a persistent empirical trend: the basic standard augmentation baseline remained incredibly resilient, holding its ground or outperforming advanced mix-based and generative approaches across all three datasets.

---

## 📊 Core Empirical Findings

### 1. Clean Test Accuracy Summary (%)
Across all three evaluation datasets, standard baseline data augmentation (random cropping and horizontal flipping) maintained the highest classification performance. Tuning Mixup from a heavy mixing parameter ($\alpha = 1.0$) to a milder setting ($\alpha = 0.2$) successfully closed the severe performance gaps observed during our midway trials, but it did not surpass the baseline.

| Dataset | Baseline (Standard) | Mixup ($\alpha = 0.2$) | CutMix ($\alpha = 1.0$) | Baseline + GAN Synthetic Data |
| :--- | :---: | :---: | :---: | :---: |
| **CIFAR-10** | **82.94%** | 82.41% | 79.28% | 82.79% |
| **CIFAR-100**| **53.83%** | 53.34% | — | — |
| **SVHN** | **93.82%** | 93.71% | — | — |

### 2. Robustness on CIFAR-10-C
Robustness was measured using Mean Corruption Accuracy (**mCA**) computed uniformly across all 19 corruption categories spanning 5 severity scales.

* **Baseline Model mCA:** **72.57%**
* **Mixup ($\alpha = 0.2$) mCA:** **72.51%** (Recovered from midway penalties, matching baseline performance but showing no absolute robustness enhancement).
* **GAN-Augmented Model mCA:** **72.65%** (A statistically neutral $+0.07\%$ gain over the baseline).
* **CutMix ($\alpha = 1.0$) mCA:** **67.14%** (Experienced a severe drop of **$-5.43\text{ pp}$**).

### 3. Key Insights & Takeaways
* **The Failure of CutMix on Pixel Corruptions:** CutMix applies patch-based masking. While this prevents the network from over-focusing on localized features, it degrades spatial consistency. As a result, models trained on CutMix generalize poorly to pixel-level perturbations such as Gaussian noise, atmospheric fog, or camera blur.
* **Neutral Impacts of GAN Synthesis:** Adding 5,000 synthetic class-conditional images generated via a Generative Adversarial Network yielded effectively neutral returns. The lack of a substantial boost is attributed to the limited visual fidelity of unconditional/pseudo-labeled low-resolution generation, which introduces subtle boundary label noise into training mini-batches.
* **The Importance of Optimization and Scheduling:** Moving to a longer epoch structure combined with a Cosine Annealing learning rate schedule proved to be more critical to absolute validation performance than the choice of an advanced data augmentation algorithm.

---

## 📂 Repository Folder Structure

```text
Robustness Enhancement in Image Classification/
│
├── .gitignore                     # Filters out local Python caches and Jupyter checkpoints
├── README.md                      # Complete project overview, findings, and documentation
│
├── code/                          # Source scripts and development notebooks
│   ├── 01_First_Midway_Progress_Code.ipynb
│   ├── 02_Second_Midway_Progress_Code.ipynb
│   ├── 03_Final_Project_Code.ipynb
│   └── 03_Final_Project_Code.py   # Final end-to-end executable execution pipeline
│
├── results/                       # Tracked data logs and evaluation metrics
│   ├── cifar10c_full_results.csv  # Granular performance logs per corruption and severity
│   ├── cifar10c_mca_summary_with_cutmix.csv
│   ├── final_clean_accuracy_results.csv
│   ├── gan_augmented_cifar10_result.csv
│   └── training_histories_all_models.csv
│
├── progress-reports/              # Project tracking milestones
│   ├── 01_First Midway Progress Report.pdf
│   └── 02_ML Second Midway Progress Report.pdf
│
├── paper/                         # Formal academic write-ups
│   ├── Final Paper 1st Draft.pdf
│   └── Final_Paper.pdf            # IEEE Two-Column Conference formatted paper
│
├── presentation/                  # Visual assets and presentation files
│   ├── Final Presentation.pdf
│   └── Final Presentation.pptx    # Complete 10-slide visual slide deck
│
└── proposal/                      # Project initiation documentation
    └── Project Proposal.pdf       # Original scoping and objective proposal
```

## ⚙️ Dependencies and Requirements

The execution environment relies on a standard `PyTorch` computer vision pipeline. The code automatically detects and allocates operations to an active CUDA-capable GPU (tested on an NVIDIA Tesla T4 engine via Google Colab).

To run the pipeline locally, ensure you have the following libraries installed:

```text
torch
torchvision
pandas
matplotlib
numpy
```
