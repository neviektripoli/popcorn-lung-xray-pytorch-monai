from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImage, AddChannel, ScaleIntensity, Resize, ToTensor
import os

from config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE

def load_data():
    train_images = [os.path.join(DATA_DIR, "train", f) for f in os.listdir(os.path.join(DATA_DIR, "train"))]
    val_images = [os.path.join(DATA_DIR, "val", f) for f in os.listdir(os.path.join(DATA_DIR, "val"))]

    train_labels = [0 if "normal" in f.lower() else 1 for f in train_images]
    val_labels = [0 if "normal" in f.lower() else 1 for f in val_images]

    train_data = [{"img": i, "label": l} for i, l in zip(train_images, train_labels)]
    val_data = [{"img": i, "label": l} for i, l in zip(val_images, val_labels)]

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
