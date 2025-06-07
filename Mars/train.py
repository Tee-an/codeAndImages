import torch
import os
import torch.nn as nn
from torch.amp import GradScaler,autocast
import matplotlib.pyplot as plt
import UNet

config = {
    "lr":1e-4,
    "batch_size":8,
    "epochs":100,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "save_path":"../models/best_model.pth"
}

if not os.path.exists("../models"):
    os.mkdir("../models",0o774)

def train(fileDir):
    train_loader,val_loader = UNet.loadData(fileDir)
    model = UNet.UNet(n_channels=3).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    best_loss = float('inf')
    train_losses,val_losses = [],[]

    for epoch in range(config['epochs']):
        model.train()
        epoch_train_loss = 0.0
        for images,masks in train_loader:
            images = images.to(config['device'])
            masks = masks.to(config['device'])

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs,masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item() * images.size(0)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images,masks in val_loader:
                images = images.to(config['device'])
                masks = masks.to(config['device'])
                outputs = model(images)
                loss = criterion(outputs,masks)
                epoch_val_loss += loss.item() * images.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(),config['save_path'])
            print(f"√ Epoch {epoch + 1}:保存最佳模型（Val loss:{avg_val_loss:.4f}）")

        print(f"Epoch {epoch + 1} / {config['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} |"
              f"Val Loss: {avg_val_loss:.4f}")

    plt.plot(train_losses,label='Train Loss')
    plt.plot(val_losses,label='Val Loss')
    plt.legend()
    if not os.path.exists("../results"):
        os.mkdir("../results")
    plt.savefig("../results/loss_curve.png")


if __name__ == '__main__':
    fileDir = "../data/train_data"
    train(fileDir)