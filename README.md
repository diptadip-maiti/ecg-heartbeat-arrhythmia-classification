# ECG Heartbeat Arrhythmia Classification

An end-to-end **ECG heartbeat arrhythmia classification** pipeline built with PyTorch for the NPPE2 T2 26 competition.

The project classifies each heartbeat into four classes using two complementary feature groups:

- **250-sample ECG beat signal** → 1D CNN
- **3 RR-interval features** (`pre_rr`, `post_rr`, `rr_ratio`) → small MLP
- CNN and RR embeddings are fused before the final classifier.

## Project highlights

- PyTorch 1D CNN for raw ECG morphology
- Auxiliary RR-interval feature branch
- Class-weighted Cross-Entropy for imbalanced classes
- Optional focal-loss implementation for experimentation
- Per-beat signal normalization
- StandardScaler fitted only on training RR features
- Stratified 80/20 validation split
- AdamW + ReduceLROnPlateau
- Gradient clipping
- Early stopping
- Reproducible training
- Macro F1 evaluation
- Per-class F1 and confusion matrix
- Test prediction + submission generation
- Training artifacts automatically saved for portfolio presentation

## Architecture

```text
                    ECG beat
                  250 samples
                      │
                      ▼
              ┌───────────────┐
              │    1D CNN     │
              │ 32 → 64 → 128 │
              └───────┬───────┘
                      │
                 ECG embedding
                      │
                      ├──────────────┐
                      │              │
                      │        RR features
                      │        pre_rr/post_rr
                      │         rr_ratio
                      │              │
                      │              ▼
                      │         ┌────────┐
                      │         │  MLP   │
                      │         │  32-D  │
                      │         └───┬────┘
                      │             │
                      └──────┬──────┘
                             ▼
                      Feature fusion
                             │
                             ▼
                       MLP classifier
                             │
                             ▼
                    4-class prediction
```

## Classes

| ID | Class |
|---:|---|
| 0 | Normal |
| 1 | Supraventricular Ectopic |
| 2 | Ventricular Ectopic |
| 3 | Fusion |

## Data

The code expects the competition dataset in this structure:

```text
nppe2_dataset/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Each training row contains:

```text
id
sig_0 ... sig_249
pre_rr
post_rr
rr_ratio
label
```

The dataset itself is **not included** in this repository.

## Quick start

### 1. Clone

```bash
git clone <https://github.com/diptadip-maiti/ecg-heartbeat-arrhythmia-classification>
cd ecg-arrhythmia-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Validate the dataset and train

```bash
python scripts/train.py \
  --data-dir /path/to/nppe2_dataset \
  --output-dir outputs
```

The script performs:

1. CSV loading
2. schema validation
3. missing-value handling for RR features
4. stratified train/validation split
5. per-beat ECG normalization
6. training-only fitting of the RR StandardScaler
7. class-weight calculation
8. CNN + RR feature fusion
9. validation with Macro F1
10. checkpointing
11. early stopping
12. artifact generation

### 4. Generate test predictions

```bash
python scripts/predict.py \
  --data-dir /path/to/nppe2_dataset \
  --checkpoint outputs/best_model.pt \
  --output outputs/submission.csv
```

## Kaggle GPU workflow

The repository is designed to run locally or in a Kaggle notebook.

For Kaggle:

```python
!git clone https://github.com/diptadip-maiti/ecg-heartbeat-arrhythmia-classification.git
%cd ecg-heartbeat-arrhythmia-classification
!pip install -q -r requirements.txt
```

Attach the competition dataset through Kaggle's **Add Input** panel and locate it under `/kaggle/input`.

Then:

```python
!python scripts/train.py \
  --data-dir "/kaggle/input/.../nppe2_dataset" \
  --output-dir outputs
```

The device is selected automatically:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Portfolio artifacts

A real training run generates:

```text
outputs/
├── training_curves.png
├── confusion_matrix.png
├── class_f1.png
├── signal_examples.png
├── metrics.json
├── best_model.pt
└── submission.csv
```


## Reproducibility

The training pipeline fixes:

- Python random seed
- NumPy seed
- PyTorch seed
- CUDA seed
- cuDNN deterministic mode

The validation split is:

```text
80% train / 20% validation
```

with stratification by class label.

## Engineering decisions

### Per-beat normalization

Each 250-point ECG beat can have a different amplitude/offset. The pipeline optionally standardizes each beat independently:

```text
x_normalized = (x - mean(x)) / std(x)
```

This emphasizes waveform morphology rather than absolute amplitude.

### RR-feature scaling

The RR features are standardized using `StandardScaler`, but the scaler is fitted **only on the training split**. Validation and test data use the training statistics.

This avoids validation/test information leaking into preprocessing.

### Class imbalance

Balanced class weights are calculated from the training labels and passed to Cross-Entropy:

```text
CrossEntropyLoss(weight=class_weights)
```

A focal-loss implementation is also included for controlled experiments.

### Optimization

The baseline uses:

- AdamW
- learning rate `1e-3`
- weight decay `1e-4`
- gradient clipping at `5.0`
- ReduceLROnPlateau
- early stopping

## Original notebook

The original Kaggle-style notebook is preserved at:

```text
notebooks/original_kaggle_experiment.ipynb
```

The modular `src/` and `scripts/` implementation is the portfolio-oriented version of the same workflow.


## Talking Points

### Why a 1D CNN?

The ECG input is a sequential waveform, so 1D convolutions can learn local morphology such as peaks, slopes, and waveform patterns while preserving temporal ordering.

### Why add RR features?

RR intervals provide timing information that is complementary to waveform morphology. The model therefore learns from both shape and beat-to-beat timing.

### Why Macro F1?

Macro F1 gives each class equal weight and is therefore useful when class frequencies are imbalanced.

### How did you avoid preprocessing leakage?

The RR scaler is fitted only on the training partition. Validation and test data are transformed using those training-derived statistics.

## Future experiments

- stronger CNN residual blocks
- temporal attention
- label smoothing
- focal loss comparison
- class-specific thresholding
- calibration analysis
- cross-validation
- ensemble of waveform models
- additional morphology features
- experiment tracking with MLflow or Weights & Biases
