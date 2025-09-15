# CNN-Based Network Intrusion Detection (CICIDS-2018)

A reproducible **1D Convolutional Neural Network (CNN) pipeline** for network intrusion detection, implemented on the **CICIDS-2018 dataset**.  
The implementation is fully re-written in a clean, modular structure with an **embedded control-plane** (powered by [Cognize](https://pypi.org/project/cognize/)) for adaptive stability and drift-aware training.

---

## Key Features
- **End-to-End Pipeline**
  - Data ingestion, preprocessing, and class balancing
  - Model definition, training, evaluation, and artifact export
- **CNN Architecture**
  - Conv1D → BatchNorm → MaxPooling → Dense → Dropout → Softmax
  - Optimized for tabular network flow features
- **Embedded Control-Plane**
  - Non-optional runtime controller for stability and robustness
  - Adaptive learning rate cooling on validation plateaus
  - Gradient clipping and bounded logit adjustments during drift
  - Lightweight telemetry (CSV logs, JSON summaries)
- **Reimplementation Guarantee**
  - Code is original and restructured for clarity
  - No reliance on prior Kaggle scripts beyond dataset usage

---

## Repository Structure
```
.
├── src/
│   ├── data_utils.py     # Data loading and preprocessing (CICIDS-2018)
│   ├── model.py          # CNN model with embedded control-plane
│   ├── callbacks.py      # AdaptivePlateau: Cognize-driven training callback
│   ├── train.py          # Training entrypoint (saves models & logs)
│   ├── evaluate.py       # Model evaluation (confusion matrix, metrics, ROC)
│   └── __init__.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation
```bash
# Create a clean environment
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel

# Install dependencies
pip install -r requirements.txt
```

Minimal requirements:
```txt
numpy
pandas
scikit-learn
tensorflow>=2.12
matplotlib
seaborn
cognize>=0.1.8
```

---

## Dataset
The project uses the **CICIDS-2018 dataset**, available from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html).  
Example file used: `02-14-2018.csv`.

Expected structure:
```
data/
 └── 02-14-2018.csv
```

---

## Usage

### 1. Train the Model
```bash
python -m src.train   --data data/02-14-2018.csv   --out_dir results   --epochs 30   --batch_size 256   --sample_frac 1.0
```

Artifacts:
```
results/
  ├── artifacts/
  │   ├── best.keras
  │   ├── final.keras
  │   ├── label_classes.json
  │   └── eval.json
  └── logs/
      └── train.csv
```

---

### 2. Evaluate the Model
```bash
python -m src.evaluate   --data data/02-14-2018.csv   --model results/artifacts/best.keras   --out_dir results/eval   --sample_frac 0.2
```

Outputs:
- `confusion.png` – confusion matrix heatmap  
- `report.json` – per-class precision/recall/F1  
- `roc.png` – multi-class ROC curves  

---

## 📈 Model Summary
- **Input:** `(n_features, 1)` tabular traffic features  
- **Backbone:** 3 × Conv1D + BatchNorm + MaxPooling  
- **Head:** Dense(128) → Dropout → Dense(#classes, softmax)  
- **Loss:** `sparse_categorical_crossentropy`  
- **Optimizer:** Adam with ClipNorm  

---

## Control-Plane (Embedded)
The CNN is tightly coupled with an **AdaptivePlateau controller**:  
- Monitors validation loss and accuracy plateaus  
- Dynamically adjusts learning rate and clip-norm bounds  
- Applies bounded logit biasing during ruptures for smoother convergence  
- Maintains training speed and prevents catastrophic forgetting  

This controller is **not optional** — it is embedded in the architecture.

---

## Citation
If you use this repository or the embedded control-plane concept in research, please cite:

```bibtex
@software{pulikanti_cognize,
  author    = {Pulikanti, Sashi Bharadwaj},
  title     = {Cognize: Programmable cognition for Python systems},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17042859},
  url       = {https://doi.org/10.5281/zenodo.17042859}
}
```

---

## License
- Code: [Apache-2.0](LICENSE)  
- Dataset: Subject to CICIDS-2018 licensing terms from its original maintainers
