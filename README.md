<div align="center">

# EMG-EdgeAI

**NinaPro DB5 sEMG gesture classification and systematic model optimization toward edge deployment**

![Dataset](https://img.shields.io/badge/Dataset-NinaPro%20DB5-blue)
![Task](https://img.shields.io/badge/Task-sEMG%20Gesture%20Classification-informational)
![Focus](https://img.shields.io/badge/Focus-Quantization%20%26%20Pruning-success)
![Benchmark](https://img.shields.io/badge/Benchmark-CPU%20Inference-lightgrey)
![Hardware](https://img.shields.io/badge/Hardware%20%2F%20Verilog-Future%20Work-orange)

</div>

This project trains a Transformer-based classifier on NinaPro DB5 surface EMG (53 classes), then studies how **quantization** and **pruning** affect accuracy, model size, parameter count, computational cost, CPU inference latency, and memory. The aim is to identify practical compression options for lightweight edge deployment, with hardware acceleration as a longer-term goal.

> [!NOTE]
> All measurements are software/CPU benchmarks. A hardware (Verilog/FPGA) implementation is **future work** and has not been done.

[Highlights](#highlights) · [Dataset and Pipeline](#dataset-and-pipeline) · [Model](#model) · [Trained Model](#trained-model-evaluation) · [Baseline](#fp32-baseline) · [Results](#optimization-results) · [Figures](#results-figures) · [Key Findings](#key-findings) · [Limitations](#limitations) · [Acknowledgment](#acknowledgment)

---

## Highlights

| Configuration | Result | Note |
| :--- | :--- | :--- |
| **FP16** | 7.49 MB (~50% smaller than FP32) | Macro-F1 0.7507 vs 0.7511; no CPU latency gain (slower in this benchmark) |
| **Dynamic INT8** | 3.85 MB, 89.56% accuracy | Only 0.14 pp accuracy loss vs FP32; Macro-F1 0.7473 |
| **Structured 50% + INT8** | 2.66 MB, 85.28% accuracy | Smallest model in this study; ~4.4 pp accuracy loss vs FP32 |
| **Aggressive 90% pruning** | Macro-F1 ≈ 0.06–0.07 | Severe degradation in this setup (accuracy ≈ 65.5%) |

*pp = percentage points.*

---

## Dataset and Pipeline

| Item | Setting |
| :--- | :--- |
| Dataset | NinaPro DB5 |
| Subjects / Exercises | 10 subjects, Exercises E1–E3 |
| Signal | 16 EMG channels, 200 Hz |
| Classes | 52 movements + rest = 53 |
| Filtering | 20–90 Hz 4th-order Butterworth band-pass, 50 Hz notch (Q = 30), zero-phase |
| Windowing | 200 samples (1 s), stride 50 samples |
| Split (by repetition) | Train: [1, 3, 4, 6] · Validation: [2] · Test: [5] |
| Subjects | All subjects pooled |
| Normalization | Per-channel z-score, statistics from training repetitions |

**Evaluation protocol.** Data from all 10 subjects are pooled, and the train/validation/test split is made by movement repetition. Every subject therefore contributes windows to all three splits, and test windows come from subjects whose other repetitions were seen during training. The reported scores measure generalization to unseen repetitions of known subjects. They are **not** a cross-subject (e.g., leave-one-subject-out) generalization test.

```text
NinaPro DB5 (10 subjects, 16-ch sEMG, 200 Hz)
        │
        ▼
Filtering & Normalization
        │
        ▼
Windowing (200 samples, stride 50)
        │
        ▼
Transformer Model (adapted from TinyMyo)
        │
        ▼
Training & Test Evaluation
        │
        ▼
FP32 Baseline
        │
        ▼
┌────────────────┬────────────────┬────────────────┐
│  Quantization  │    Pruning     │     Hybrid     │
│  FP16 / INT8   │  Unstructured  │  Pruning+INT8  │
│                │  Structured    │                │
│                │  Gradual       │                │
└────────────────┴────────────────┴────────────────┘
        │
        ▼
Accuracy / Macro-F1 / Size / MACs / Latency
        │
        ▼
Candidate Edge Models  →  Future Hardware / Verilog (not yet implemented)
```

---

## Model

The architecture is **adapted from the TinyMyo architecture of the [BioFoundation](https://github.com/pulp-bio/BioFoundation) project**. It was not designed in this repository (see [Acknowledgment](#acknowledgment)).

| Component | Setting |
| :--- | :--- |
| Input | 16 EMG channels × 200 samples |
| Patch size | 20 samples |
| Embedding dimension | 192 |
| Transformer layers | 8 (Pre-LN) |
| Attention heads | 3 |
| Positional encoding | RoPE |
| Output | 53-class classification head |

**Contributions of this project:** preprocessing, training and evaluation, quantization, pruning, compression, and benchmarking/comparison of the resulting models.

---

## Trained Model Evaluation

After training, the primary model was evaluated on the held-out test repetition (repetition 5) using a standard classification report (values shown to three decimals).

| Average | Precision | Recall | F1-score | Support |
| :--- | ---: | ---: | ---: | ---: |
| Accuracy | — | — | 0.897 | 18,717 |
| Macro avg | 0.778 | 0.731 | 0.750 | 18,717 |
| Weighted avg | 0.893 | 0.897 | 0.893 | 18,717 |

Macro-averaging weights all 53 classes equally, whereas weighted-averaging weights them by support. The gap between the two shows that performance is uneven across classes, which is why macro-F1 is tracked alongside accuracy throughout this project.

> [!NOTE]
> The next section reports the FP32 benchmark of this same trained model, which gives macro-F1 = 0.7511 versus 0.750 here. The two values come from separate evaluation passes (the classification report above and the benchmarking pipeline), so a difference of about 0.001 is not a contradiction between different models. All optimization comparisons use the **FP32 benchmark values** as the reference.

<p align="center">
  <img src="results/confusion_matrix_test.png" alt="Test-set confusion matrix of the trained model" width="60%"><br>
  <b>Figure: Test-set confusion matrix of the trained model (before optimization).</b>
</p>

---

## FP32 Baseline

Benchmark of the same trained model in FP32, used as the reference for every optimized variant below.

| Metric | FP32 |
| :--- | ---: |
| Accuracy | 89.70% |
| Macro-F1 | 0.7511 |
| Parameters | 3,729,269 |
| Model size | 14.95 MB |
| BS=1 mean latency | 5.91 ms |

<p align="center">
  <img src="results/fp32_cm.png" alt="FP32 baseline confusion matrix" width="60%"><br>
  <b>Figure: FP32 baseline confusion matrix.</b>
</p>

---

## Optimization Results

Macro-F1 is reported alongside accuracy because it weights all 53 classes equally. Latency is mean CPU latency at batch size 1 (BS=1).

| Method | Accuracy | Macro-F1 | Model Size | Parameters / Non-zero Params | BS=1 Latency |
| :--- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 89.70% | 0.7511 | 14.95 MB | 3,729,269 | 5.91 ms |
| FP16 | 89.70% | 0.7507 | 7.49 MB | 3,729,269 | 9.48 ms |
| INT8 Dynamic | 89.56% | 0.7473 | 3.85 MB | 3,729,269 | 24.84 ms |
| INT8 PTQ | 71.66% | 0.3351 | 4.25 MB | 3,729,269 | 35.18 ms |
| Unstruct. 50% | 89.11% | 0.7359 | 14.95 MB | 1,959,797 | 5.91 ms |
| Unstruct. 70% | 84.75% | 0.6233 | 14.95 MB | 1,252,021 | 5.98 ms |
| Unstruct. 90% | 65.46% | 0.0659 | 14.95 MB | 544,237 | 7.73 ms |
| Struct. 25% | 89.08% | 0.7378 | 12.58 MB | 3,137,909 | 7.81 ms |
| Struct. 50% | 85.43% | 0.6468 | 10.22 MB | 2,546,549 | 8.61 ms |
| Gradual 70% | 85.03% | 0.6323 | 14.95 MB | 1,252,020 | 5.85 ms |
| Gradual 90% | 65.53% | 0.0632 | 14.95 MB | 544,237 | 5.94 ms |
| Unstruct. 70% + INT8 | 84.61% | 0.6195 | 3.85 MB | 1,250,143 | 24.70 ms |
| Struct. 50% + INT8 | 85.28% | 0.6439 | 2.66 MB | 2,496,012 | 20.97 ms |
| Gradual 70% + INT8 | 84.96% | 0.6306 | 3.85 MB | 1,250,112 | 34.39 ms |

**Reading the parameter column.** The column has a different meaning depending on the method:

- **Quantization rows (FP16, INT8):** total parameter count, unchanged from FP32. Only the numeric precision of the stored weights changes.
- **Unstructured and gradual pruning rows:** the number of **non-zero** parameters. Pruned weights are set to zero, but the dense tensors keep their original shape, so the total parameter count remains 3,729,269 and the stored model size stays at 14.95 MB unless the model is also quantized or stored in a sparse format. The percentage in the method name is the pruning target; since some parameters are not pruned, the resulting global non-zero fraction is slightly higher than the nominal target (e.g., 1,959,797 non-zero parameters, or about 53% of the total, at the 50% target).
- **Structured pruning rows:** the **total parameter count of the smaller dense model**, since whole structures are removed and the tensors themselves shrink. This reduces both parameter count and dense model size. As with unstructured pruning, the reduction in total parameters is smaller than the nominal target (50% structured pruning removes about 32% of the parameters).
- **Hybrid (pruning + INT8) rows:** non-zero parameters of the pruned and quantized model, which can differ slightly from the pre-quantization count.

Gradual pruning raises sparsity progressively over pruning steps rather than in a single step (see [Gradual pruning schedule](#gradual-pruning-schedule)).

**Sparsity and computation.** Unstructured and gradual sparsity do not by themselves reduce computation or latency on a dense runtime. Zero-valued weights are still stored and multiplied by dense kernels, so latency changes only if sparse storage, sparse kernels, or dedicated hardware explicitly exploit the zeros. The latencies of the unstructured, gradual, and hybrid rows were measured with dense execution and therefore do not reflect any potential benefit from sparsity.

---

## Results Figures

### Overall comparison

<table>
  <tr>
    <td align="center" width="50%"><img src="results/accuracy.png" alt="Accuracy comparison" width="100%"><br><sub><b>Accuracy</b></sub></td>
    <td align="center" width="50%"><img src="results/macro_f1.png" alt="Macro-F1 comparison" width="100%"><br><sub><b>Macro-F1</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="results/size.png" alt="Model size comparison" width="100%"><br><sub><b>Model size</b></sub></td>
    <td align="center"><img src="results/latency.png" alt="BS=1 latency comparison" width="100%"><br><sub><b>BS=1 CPU latency</b></sub></td>
  </tr>
</table>

The plot below combines **model size**, **BS=1 latency**, and **accuracy** to show the trade-offs between configurations.

<p align="center">
  <img src="results/emg_tradeoff.png" alt="Size, latency and accuracy trade-off" width="75%"><br>
  <b>Figure: Size / latency / accuracy trade-off.</b>
</p>

### Gradual pruning schedule

In gradual pruning, sparsity is raised step by step toward the target instead of being applied at once. The figure shows sparsity versus pruning step for both the 70% and 90% targets.

<p align="center">
  <img src="results/gradual_pruning.png" alt="Sparsity versus pruning step for gradual 70% and 90% pruning" width="75%"><br>
  <b>Figure: Sparsity vs. pruning step for gradual 70% and 90% pruning.</b>
</p>

### Confusion matrices

Representative confusion matrices are shown below. The repository contains the matrices for the remaining configurations in `results/`.

**Quantization**

<table>
  <tr>
    <td align="center" width="33%"><img src="results/fp32_cm.png" alt="FP32 confusion matrix" width="100%"><br><sub><b>FP32</b></sub></td>
    <td align="center" width="33%"><img src="results/int8_dynamics_cm.png" alt="INT8 dynamic confusion matrix" width="100%"><br><sub><b>INT8 Dynamic</b></sub></td>
    <td align="center" width="33%"><img src="results/int8_ptq_cm.png" alt="INT8 PTQ confusion matrix" width="100%"><br><sub><b>INT8 PTQ</b></sub></td>
  </tr>
</table>

**Pruning and hybrid pruning + INT8**

<table>
  <tr>
    <td align="center" width="33%"><img src="results/unstruct_50_cm.png" alt="Unstructured 50% confusion matrix" width="100%"><br><sub><b>Unstruct. 50%</b></sub></td>
    <td align="center" width="33%"><img src="results/unstruct_70_cm.png" alt="Unstructured 70% confusion matrix" width="100%"><br><sub><b>Unstruct. 70%</b></sub></td>
    <td align="center" width="33%"><img src="results/struct_50_cm.png" alt="Structured 50% confusion matrix" width="100%"><br><sub><b>Struct. 50%</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="results/struct_50_int8_cm.png" alt="Structured 50% plus INT8 confusion matrix" width="100%"><br><sub><b>Struct. 50% + INT8</b></sub></td>
    <td align="center"><img src="results/gradual_70_cm.png" alt="Gradual 70% confusion matrix" width="100%"><br><sub><b>Gradual 70%</b></sub></td>
    <td align="center"><img src="results/gradual_70_int8_cm.png" alt="Gradual 70% plus INT8 confusion matrix" width="100%"><br><sub><b>Gradual 70% + INT8</b></sub></td>
  </tr>
</table>

---

## Key Findings

These conclusions apply to this model, this protocol, and this CPU benchmark.

1. **FP16** roughly halves model size (14.95 → 7.49 MB) with essentially no accuracy or macro-F1 change. It did not improve BS=1 CPU latency here (9.48 ms vs 5.91 ms for FP32).
2. **Dynamic INT8** gave the largest size reduction among the near-lossless options: 3.85 MB (about 74% smaller than FP32) at 89.56% accuracy, a loss of only 0.14 pp.
3. **INT8 PTQ** degraded performance substantially in this setup: 71.66% accuracy and macro-F1 of 0.3351.
4. **Pruning:** Unstructured 50% (89.11%) and Structured 25% (89.08%) stayed close to the baseline accuracy, although macro-F1 already dropped slightly (0.7359 and 0.7378 vs 0.7511). Structured 50% and the 70% settings lost about 4.3–5 pp accuracy, and macro-F1 fell to 0.62–0.65. At 90% sparsity, accuracy fell to about 65.5% and macro-F1 to about 0.06–0.07, so 90% was too aggressive for this setup.
5. **Structured pruning** reduces both parameter count and dense model size. At 50%, the model is 10.22 MB with 85.43% accuracy.
6. **Structured 50% + INT8** reached 2.66 MB at 85.28% accuracy. That is a smaller model at similar accuracy compared with structured pruning alone (10.22 MB, 85.43%), and a smaller and slightly more accurate model than unstructured 70% + INT8 (3.85 MB, 84.61%). Compared with Dynamic INT8, it is smaller (2.66 vs 3.85 MB) but about 4.3 pp less accurate.
7. **CPU latency (BS=1, mean):** No optimized variant was meaningfully faster than FP32 (5.91 ms). Unstructured 50% and 70% (5.91 and 5.98 ms), Gradual 70% and 90% (5.85 and 5.94 ms) were within about 0.1 ms of the baseline, which is too small to interpret without variance estimates. Every other variant was slower: FP16 (9.48 ms), structured pruning (7.81 and 8.61 ms), Unstructured 90% (7.73 ms), and all INT8-based variants (20.97–35.18 ms). Configurations with identical or near-identical non-zero parameter counts also showed different latencies (Unstruct. 90% at 7.73 ms vs Gradual 90% at 5.94 ms; Unstruct. 70% + INT8 at 24.70 ms vs Gradual 70% + INT8 at 34.39 ms), so differences of this size should be treated with caution. Lower precision or fewer non-zero weights therefore did not translate into lower latency in this benchmark, and these timings depend on the CPU and runtime implementation.

**Practical options in this study:** Dynamic INT8 when accuracy matters most, Structured 50% + INT8 when size matters most, and FP16 as a simple ~2× size reduction. None of these improved CPU latency over FP32 in this benchmark.

---

## Computational Details

Beyond accuracy and macro-F1, the experiments also measured the metrics below. The full results are kept in the repository.

| Category | Metrics |
| :--- | :--- |
| Model complexity | Parameter count, non-zero parameters, sparsity, MACs/FLOPs |
| Memory | Model size, activation memory, peak RAM |
| Latency | Mean, P95, and P99 latency; throughput |
| Fidelity to FP32 | Numerical error, top-1 agreement with FP32 |

Energy and power measurements were **not available** in the current benchmark.

---

## Repository Structure

```text
EMG-EdgeAI/
├── models/
├── notebooks/
├── results/
└── README.md
```

| Path | Contents |
| :--- | :--- |
| `models/` | Model-related files for the baseline and optimized variants |
| `notebooks/` | Notebooks for preprocessing, training, optimization, and evaluation |
| `results/` | Result figures (comparison plots, trade-off plot, confusion matrices) and detailed results |
| `README.md` | This document |

---

## Limitations

- The benchmark is software/CPU-based; results may differ on other CPUs, runtimes, or backends.
- Unstructured and gradual sparsity were evaluated on a dense runtime, so their latency and stored size do not show any benefit that sparse kernels or sparse storage might provide.
- INT8 performance is runtime/backend dependent, so the INT8 latencies here should not be generalized.
- Latency differences of a few tenths of a millisecond are not interpreted, and some configurations with identical or near-identical non-zero parameter counts show noticeably different latencies (see Key Findings).
- Results follow the current NinaPro DB5 protocol (pooled subjects, repetition-based split, test on repetition 5). They measure generalization to unseen repetitions of subjects seen in training and should not be read as a cross-subject generalization test.
- Metrics come from a single trained model and a single test repetition; no confidence intervals or repeated runs are reported.
- No hardware/Verilog accelerator has been implemented yet.
- Energy and power were not measured.
- The architecture is adapted from the BioFoundation/TinyMyo work and is not an original design.

---

## Future Work

- Hardware-aware optimization
- Better quantization and calibration
- Deployment-oriented benchmarking, including sparse-aware runtimes or kernels for unstructured sparsity
- Cross-subject (e.g., leave-one-subject-out) evaluation
- FPGA/Verilog implementation and hardware acceleration
- Power and energy measurement
- Further exploration of structured sparsity

---

## Acknowledgment

> The model architecture used in this project is adapted from the TinyMyo architecture developed in the BioFoundation project. The preprocessing, training/evaluation setup, optimization experiments, and benchmarking presented in this repository were developed for this project.

**TinyMyo / BioFoundation**

- Paper: M. Fasulo, G. Spacone, T. M. Ingolfsson, Y. Li, L. Benini, and A. Cossettini, "TinyMyo: a Tiny Foundation Model for Flexible EMG Signal Processing at the Edge," arXiv:2512.15729, 2025. [https://arxiv.org/abs/2512.15729](https://arxiv.org/abs/2512.15729)
- Code: [https://github.com/pulp-bio/BioFoundation](https://github.com/pulp-bio/BioFoundation)

**Dataset:** NinaPro DB5, [NinaPro](https://ninapro.hevs.ch/).

**Dataset reference:** M. Atzori, A. Gijsberts, I. Kuzborskij, S. Elsig, A.-G. M. Mittaz Hager, O. Deriaz, C. Castellini, H. Müller, and B. Caputo, "Characterization of a benchmark database for myoelectric movement classification," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 23, no. 1, pp. 73–83, 2015, doi: [10.1109/TNSRE.2014.2328495](https://doi.org/10.1109/TNSRE.2014.2328495).
