# EGSimCLR — Entropy-Guided Contrastive Learning for Remote Sensing Scene Classification

Official implementation of:

> S. F. Ismael and E. Aptoula, "Entropy-Guided Contrastive Learning for Remote Sensing Scene Classification,"
> *IEEE Geoscience and Remote Sensing Letters*, 2026.
> DOI: `<ADD DOI HERE>`

Contrastive learning in remote sensing suffers from the **false-negative problem**: semantically
similar images drawn into the same batch are treated as negatives and pushed apart. EGSimCLR
estimates a per-pair *collision score* from feature similarity and the normalized prediction
entropy of a lightweight auxiliary head, and uses it to **down-weight pairs that are likely to be
false negatives** rather than removing them.

The method adds one weighting term to the InfoNCE objective and no additional training stage.

---

## Method in brief

For a negative pair `(i, j)` the collision score combines similarity with entropy-based gating:

```
q_ij = s̃_ij · (1 − Ĥ_avg) · exp(−ΔĤ)
w_ij = exp(−λ · q_ij)
```

where `s̃_ij` is normalized cosine similarity, `Ĥ` is entropy normalized by `log K̂`, and `λ`
follows a linear warm-up then cosine decay. The weights `w_ij` scale the negative terms of the
contrastive loss, so high-collision pairs contribute less.

---

## Installation

```bash
git clone https://github.com/Sarmadfismael/EGSimCLR.git
cd EGSimCLR
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10 and PyTorch 2.x on a single NVIDIA RTX 4090 (24 GB).

---

## Data setup

All paths derive from a single environment variable, `GRSL_ROOT`. No code changes are needed:

```bash
export GRSL_ROOT=/path/to/root          # Windows: $env:GRSL_ROOT="C:\path\to\root"
```

Expected layout beneath it:

```
<GRSL_ROOT>/
└── dataset/
    ├── AID/                  # 10,000 images, 30 classes
    ├── EuroSAT/              # 27,000 images, 10 classes
    ├── UC_Merced/            #  2,100 images, 21 classes
    └── MS_Dataset/           # Mosul Scene Dataset — see below
        ├── split_dataset1/   # seed 42
        ├── split_dataset2/   # seed 123
        ├── split_dataset3/   # seed 456
        ├── split_dataset4/   # seed 789
        └── split_dataset5/   # seed 1011
```

AID, EuroSAT and UC Merced are public benchmarks obtainable from their original sources.
The five MSD splits are independently sampled stratified 70/15/15 partitions, each generated
with its own random state (listed above) so that the five partitions differ.

---

## Usage

Pre-train with the entropy-guided objective:

```bash
python train.py --dataset AID
```

Reproduce the main comparison (Table II — five independent 70/15/15 splits):

```bash
python run_table1_proposal_5splits.py
```

Reproduce the baselines:

```bash
python run_baseline_simclr_5splits.py
python run_baseline_moco_5splits.py
python run_baseline_byol_5splits.py
python run_baseline_barlowtwins_5splits.py
python run_baseline_swav_5splits.py
python run_baseline_supervised_5splits.py
```

Cross-dataset transfer (Table IV — EuroSAT → UC Merced):

```bash
python run_table2_crossdataset.py
```

Analyses reported in the paper:

```bash
python evaluate_5split_confusion.py   # pooled confusion matrix (Table III)
python w_ij_final_snapshot.py         # distribution of q_ij and w_ij at convergence
python qualitative_examples.py        # corrected / persistent failure cases
```

---

## Key hyperparameters

| Setting | Value |
|---|---|
| Encoder | ResNet-50, random init |
| Projection head | 2048 → 2048 → 128 |
| Classifier head | 2048 → 512 → K̂ = 32, dropout 0.2 |
| K̂ (pseudo-classes) | 32, fixed for all datasets |
| τ (temperature) | 0.07 |
| λ range | 1.0 → 5.0, warm-up 20 epochs then cosine decay |
| Pre-training | Adam, lr 3e-4, 200 epochs |
| Fine-tuning | Adam, lr 1e-4, wd 1e-4, batch 32, early stopping patience 20 |
| Evaluation | 5 stratified 70/15/15 splits, master seed 42 |

---

## Repository layout

```
├── train.py / test.py              # pre-training and evaluation entry points
├── config.py                       # paths and hyperparameters
├── models/                         # encoders, projection/classifier heads, fine-tuning variants
├── datasets/loader.py              # dataset loading and split generation
├── utils/metrics.py                # macro-averaged accuracy, precision, recall, F1
├── run_baseline_*.py               # baseline reproductions
├── run_table*.py                   # main comparison and cross-dataset transfer
└── results/                        # per-split metrics and confusion matrix (JSON)
```

Trained checkpoints are **not** stored in this repository because of their size.

---

## Mosul Scene Dataset (MSD)

MSD contains 900 high-resolution tiles (100 per class, 500×500 px) across nine urban and
semi-urban classes over Mosul, Iraq (36.34°N, 43.13°E), with capture dates between 2022 and 2024.

> **Release status:** pending. MSD is derived from Google Earth imagery, whose terms restrict
> redistribution. Release details will be added here once the applicable rights have been
> confirmed. For access enquiries in the meantime, please contact the corresponding author.

---

## Reproducibility notes

- All reported numbers use five independently sampled stratified 70/15/15 splits
  (random states 42, 123, 456, 789, 1011), reported as mean ± standard deviation. The model
  initialization seed is held fixed so that the partition is the only source of variation.
- Pre-training batch size follows each method's own configuration and is bounded by the memory of
  a single GPU; the comparison is therefore **not batch-size-controlled**.
- Sensitivity to K̂ was not assessed.

---

## Citation

```bibtex
@article{ismael2026entropy,
  author  = {Ismael, Sarmad F. and Aptoula, Erchan},
  title   = {Entropy-Guided Contrastive Learning for Remote Sensing Scene Classification},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  doi     = {ADD-DOI-HERE}
}
```

---

## License

Code released under the MIT License — see [LICENSE](LICENSE).
The MSD dataset is **not** covered by this license; see the section above.

## Contact

Sarmad F. Ismael — sarmad.ismael@uoninevah.edu.iq — Ninevah University, Mosul, Iraq
Erchan Aptoula — Sabanci University, Istanbul, Türkiye
