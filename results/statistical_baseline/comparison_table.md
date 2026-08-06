# Method comparison (test set=DS2)

| method               |   precision |   recall |    f1 |   roc_auc |
|:---------------------|------------:|---------:|------:|----------:|
| Spectral features    |       0.297 |    0.502 | 0.373 |     0.773 |
| CUSUM (k=0.5)        |       0.41  |    0.335 | 0.369 |     0.68  |
| Z-score (10s window) |       0.652 |    0.175 | 0.276 |     0.615 |
| Random baseline      |       0.11  |    0.996 | 0.198 |     0.5   |
| Z-score (1s window)  |       0.11  |    0.997 | 0.198 |     0.274 |
