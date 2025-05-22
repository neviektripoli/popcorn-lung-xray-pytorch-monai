### README.md

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

---

### requirements.txt

```
torch
torchvision
monai
numpy
matplotlib
pandas
tqdm
scikit-learn
opencv-python
```

---

### config.py

```python
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 2
IMAGE_SIZE = (224, 224)
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
```

---

### model.py

```python
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2):
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

---

### dataset.py

```python
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImage, AddChannel, ScaleIntensity, Resize, ToTensor
import os
import random
from config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE

def load_data():
    train_images = [os.path.join(DATA_DIR, "train", f) for f in os.listdir(os.path.join(DATA_DIR, "train"))]
    val_images = [os.path.join(DATA_DIR, "val", f) for f in os.listdir(os.path.join(DATA_DIR, "val"))]

    train_labels = [0 if "normal" in f.lower() else 1 for f in train_images]
    val_labels = [0 if "normal" in f.lower() else 1 for f in val_images]

    train_data = [{"img": i, "label": l} for i, l in zip(train_images, train_labels)]
    val_data = [{"img": i, "label": l} for i, l in zip(val_images, val_labels)]

    random.shuffle(train_data)
    random.shuffle(val_data)

    transforms = Compose([
        LoadImage(image_only=True),
        AddChannel(),
        ScaleIntensity(),
        Resize(IMAGE_SIZE),
        ToTensor()
    ])

    train_ds = CacheDataset(train_data, transform=transforms)
    val_ds = CacheDataset(val_data, transform=transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader
```

---

### utils.py

```python
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    return acc, f1, auc
```

---

### train.py

```python
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import get_model
from dataset import load_data
from config import *
from utils import save_checkpoint, evaluate

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=NUM_CLASSES).to(device)
    train_loader, val_loader = load_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc, f1, auc = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        save_checkpoint(model, f"{CHECKPOINT_DIR}/model_epoch_{epoch+1}.pt")

if __name__ == '__main__':
    train()
```
