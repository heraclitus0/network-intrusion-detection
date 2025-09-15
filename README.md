# Network Intrusion Detection Using CNN

This project implements a Convolutional Neural Network (CNN) for classifying network traffic (CICIDS-2018 dataset) into Benign, FTP-BruteForce, or SSH-BruteForce.

## Overview
- Dataset: CICIDS-2018
- Model: 1D CNN
- Output: Multiclass classification (Benign / FTP-BruteForce / SSH-BruteForce)

## Repository Structure
```
├── data/                # Datasets
├── notebooks/           # Exploratory analysis
├── src/                 # Source code (data, model, training, evaluation)
├── results/             # Logs, plots, checkpoints
├── requirements.txt     # Dependencies
└── README.md            # Project description
```

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py       # Train the model
python src/evaluate.py    # Evaluate the model
```

## Results
- Training accuracy: ~99%
- Validation accuracy: ~92%

## License
Apache 2.0
