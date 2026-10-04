# Data Dictionary

The model uses 30 numeric measurements derived from digitized breast-mass images. Feature names follow the WDBC dataset convention.

| Group | Meaning |
|---|---|
| `*_mean` | Mean measurement |
| `*_se` | Standard error |
| `*_worst` | Largest/worst measurement in the original feature grouping |

Core measurement families include radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

The target column is `diagnosis`:

- `M` = malignant
- `B` = benign

The source CSV also contains `id` and `Unnamed: 32`; these are not model inputs.

**Important:** This dictionary describes the dataset fields; it does not imply clinical interpretation or diagnostic validity of the model.
