# TODO: USE stratified kfold

import os
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from data import get_tiles

# from sklearn.model_selection import KFold, train_test_split
from dotenv import load_dotenv

from dataset import NASAHarverstFieldDataset
from model import FieldBoundariesDetector

# from torchsummary import summary
import numpy as np
import wandb
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss

load_dotenv()
wandb.init(project="nasa_harvest_model_rwanda_field_boundary_detection")

torch.manual_seed(4124)
torch.cuda.manual_seed(4124)
np.random.seed(4124)

torch.backends.cudnn.deterministic = True


def train_epoch(
    both_losses, model, optimizer, criterion, criterion2, data_loader, device, scaler
):
    model.train()
    running_loss = 0.0
    running_f1 = 0.0
    for inputs, targets in data_loader:
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            preds = (nn.Sigmoid()(outputs) >= 0.5).float()
            loss = criterion2(outputs, targets)
            # if both_losses:
            #   loss += criterion2(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()  #

        running_loss += loss.item()
        wandb.log(
            {
                "Input": wandb.Image(
                    inputs[-1][np.random.randint(6)]
                    .permute(1, 2, 0)
                    .to("cpu")
                    .numpy()[:, :, 2:]
                ),
                "Output": wandb.Image(preds[-1]),
                "Mask": wandb.Image(targets[-1]),
            }
        )

        running_f1 += f1_score(
            targets.cpu().numpy().ravel(), preds.cpu().numpy().ravel()
        )
        del inputs, targets, outputs
        torch.cuda.empty_cache()

    epoch_loss = running_loss / len(data_loader)
    epoch_f1 = running_f1 / len(data_loader)
    return epoch_loss, epoch_f1


def valid_epoch(both_losses, model, criterion, criterion2, data_loader, device):
    model.eval()
    running_loss = 0.0
    running_f1 = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            outputs = model(inputs)
            preds = (nn.Sigmoid()(outputs) >= 0.5).float()
            loss = criterion2(outputs, targets)
            # if both_losses:
            #   loss += criterion(outputs, targets)
            running_loss += loss.item()

            wandb.log(
                {
                    "Input_val": wandb.Image(
                        inputs[-1][np.random.randint(6)]
                        .permute(1, 2, 0)
                        .to("cpu")
                        .numpy()[:, :, 2:]
                    ),
                    "Output_val": wandb.Image(preds[-1]),
                    "Mask_val": wandb.Image(targets[-1]),
                }
            )
            running_f1 += f1_score(
                targets.cpu().numpy().ravel(), preds.cpu().numpy().ravel()
            )
            del inputs, targets, outputs, preds
            torch.cuda.empty_cache()

    epoch_loss = running_loss / len(data_loader)
    epoch_f1 = running_f1 / len(data_loader)

    return epoch_loss, epoch_f1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archive = "source_train"
    tiles_list = list(get_tiles(archive).items())
    val_indices = np.random.choice(
        len(tiles_list), size=int(0.2 * len(tiles_list)), replace=False
    )
    val_tiles_list = [tiles_list[i] for i in val_indices]
    train_tiles_list = [
        tiles_list[i] for i in range(len(tiles_list)) if i not in val_indices
    ]

    train_dataset = NASAHarverstFieldDataset(archive, train_tiles_list)
    val_dataset = NASAHarverstFieldDataset(
        archive, val_tiles_list, apply_transforms=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=12
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=12)

    num_epochs = 200
    model = FieldBoundariesDetector()
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    # criterion  = JaccardLoss('binary')#, from_logits=False)#, smooth=1)
    # criterion  = DiceLoss('binary')#, from_logits=False)
    criterion.to(device)
    criterion2 = JaccardLoss("binary")
    criterion2.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.8, patience=15, verbose=True
    )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=1000,
    #                                     steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler()
    best_val_f1 = float("-inf")
    both_losses = False
    for e in range(num_epochs):
        train_loss, train_f1 = train_epoch(
            both_losses,
            model,
            optimizer,
            criterion,
            criterion2,
            train_loader,
            device,
            scaler,
        )
        wandb.log({"train_loss": train_loss, "train_f1": train_f1, "epoch": e})
        val_loss, val_f1 = valid_epoch(
            both_losses, model, criterion, criterion2, val_loader, device
        )
        wandb.log({"val_loss": val_loss, "val_f1": val_f1, "epoch": e})
        print(f"Epoch {e}", {"train_loss": train_loss, "train_f1": train_f1})
        print(f"Epoch {e}", {"val_loss": val_loss, "val_f1": val_f1})
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.run.summary["best_val_f1"] = best_val_f1
            wandb.run.summary["best_epoch"] = e
            torch.save(
                {
                    "epoch": e,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                },
                os.path.join(wandb.run.dir, "best_model.pt"),
            )
            wandb.save("best_model.pt")
        if e > 50:
            scheduler.step(val_f1)
        # if e >75: both_losses = True

    # Load best model and train a bit on validation data
    new_train_dataset = NASAHarverstFieldDataset(archive, train_tiles_list)
    new_train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=12
    )
    checkpoint = torch.load(os.path.join(wandb.run.dir, "best_model.pt"))
    model = FieldBoundariesDetector()
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.8, patience=10, verbose=True
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_val_f1 = checkpoint["best_val_f1"]
    epoch = checkpoint["epoch"]
    # for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.00001
    both_losses = True
    best_train_f1 = float("-inf")
    for e in range(100):
        train_loss, train_f1 = train_epoch(
            e, model, optimizer, criterion, criterion2, new_train_loader, device, scaler
        )
        wandb.log(
            {
                "train_loss_with_val": train_loss,
                "train_f1_with_val": train_f1,
                "epoch_after_val": e,
            }
        )
        print(
            {
                "train_loss_with_val": train_loss,
                "train_f1_with_val": train_f1,
                "epoch_after_val": e,
            }
        )
        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
            wandb.run.summary["best_train_f1_with_val"] = best_train_f1
            wandb.run.summary["best_epoch_after_val"] = e
            torch.save(
                {
                    "epoch": e,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_train_f1": best_train_f1,
                },
                os.path.join(wandb.run.dir, "best_model_with_train_and_val_data.pt"),
            )
            wandb.save("best_model_with_train_and_val_data.pt")

    wandb.finish()
