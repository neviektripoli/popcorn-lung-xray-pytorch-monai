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
