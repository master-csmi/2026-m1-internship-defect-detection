# ECG Anomaly Detection: MIT-BIH Arrhythmia Database

M1 internship 2026: Detecting abnormal heartbeats in ECG signals.

## Setup

```bash
pip install -e ".[notebook]"
python -c "from m1_defect_anomaly_2026.data import download_mitbih; download_mitbih()"
```

## Exploratory Data Analysis

Main findings:
- Signal are sampled at 360 Hz and annotations are sample indices placed on the R-peak.
- Beats follow the AAMI EC57 five class grouping (N, S, V, F, Q).
- Severe class imbalance: N is ~85% of all beats and fusion (F) beats number only a few hundred in the whole database.
- 4 paced records are excluded from modelling.
- We adopt the "de Chazal et al. (2004) inter-patient split" (DS1 train/ DS2 test) so results stay comparable to published work.

## Running the tests

```bash
pytest
```


## Running with Docker

```bash
docker build -t ecg-anomaly .
docker compose up
```

Then open the link with the token shown in the terminal.

The ECG data is not inside the image. To download it once:

```bash
docker run --rm -v "$(pwd)/data:/app/data" ecg-anomaly \
  python -c "from m1_defect_anomaly_2026.data import download_mitbih; download_mitbih()"
```

To run the tests in the container:

```bash
docker run --rm ecg-anomaly pytest
```


## Methods and results

Every method is measured the same way: the threshold is chosen on DS1, the score is
measured on DS2, so no patient is ever in both. All the numbers below are on DS2.

| Method | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest (supervised) | 0.837 | 0.633 | 0.721 | 0.948 |
| XGBoost (supervised) | 0.686 | 0.677 | 0.681 | 0.940 |
| Isolation Forest (unsupervised) | 0.343 | 0.566 | 0.427 | 0.836 |
| Spectral features (statistical) | 0.297 | 0.502 | 0.373 | 0.773 |
| CUSUM, k=0.5 (statistical) | 0.410 | 0.335 | 0.369 | 0.680 |
| Z-score, 10s window (statistical) | 0.652 | 0.175 | 0.276 | 0.615 |
| One-Class SVM (unsupervised) | 0.200 | 0.437 | 0.275 | 0.687 |
| Random baseline | 0.110 | 0.996 | 0.198 | 0.500 |
| Z-score, 1s window (statistical) | 0.110 | 0.997 | 0.198 | 0.274 |


Main findings:
- The supervised models win clearly: F1 0.721 against 0.427 for the best unsupervised one.
- What the labels bring is precision, not recall. Isolation Forest already found 57% of the anomalies, the Random Forest finds 63%, but its precision goes from 0.34 to 0.84. The labels teach the model to stop flagging normal beats, not to see more abnormal ones.
- SHAP shows the model looks at rhythm first (heart_rate, rr_deviation, pre_rr, rr_ratio) and at the shape of the beat after. No feature dominates, so the model adds many small clues.
- The limit is the class S. V beats are caught at 0.946 by both models, but S stays at 0.161 (Random Forest) and 0.275 (XGBoost), and F at 0.281 and 0.351. So the overall F1 mostly reflects the V beats. An S beat comes early but keeps a normal shape, so only the rhythm features can see it.
- A 1-second window for the Z-score is below the random baseline in ROC-AUC (0.274 against 0.500): the window is too short, the anomaly becomes its own reference.

## Notebooks

| Notebook | What it does |
|---|---|
| `01_exploration.ipynb` | The data, the classes, the imbalance |
| `02_statistical_baseline.ipynb` | Z-score, CUSUM, spectral features |
| `03_features_unsupervised.ipynb` | 35 features per beat, Isolation Forest, One-Class SVM |
| `04_supervised_ml.ipynb` | Random Forest, XGBoost, SHAP, full comparison |

Results and figures are saved under `results/`.