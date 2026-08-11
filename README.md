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



