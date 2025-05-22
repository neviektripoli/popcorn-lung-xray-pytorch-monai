# Popcorn Lung X-ray Classifier
A PyTorch + MONAI-based classifier to detect Popcorn Lung (Bronchiolitis Obliterans) from chest X-ray images.

### Features
- Binary classification (Popcorn Lung vs. Normal)
- Dataset loader using MONAI's `CacheDataset`
- Pre-trained ResNet18 architecture
- Evaluation metrics: Accuracy, F1-score, ROC-AUC
- Logging and checkpointing support
- Google Colab-ready

### Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Google Colab
You can run this project on [Google Colab](https://colab.research.google.com/) by uploading the files and updating the `DATA_DIR` path in `config.py`.
