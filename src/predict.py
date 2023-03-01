import numpy as np
import torch
from data import get_tiles
from dataset import NASAHarverstFieldDataset
from torch.utils.data import DataLoader

from model import FieldBoundariesDetector
from dotenv import load_dotenv

import wandb
import torch
import torch.nn as nn
import pandas as pd

load_dotenv()

torch.backends.cudnn.deterministic = True
torch.manual_seed(4124)
torch.cuda.manual_seed(4124)
wandb.init(project="nasa_harvest_model_rwanda_field_boundary_detection")


def predict(model, test_ids, test_loader, device):
    model.eval()
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.float().to(device)
            outputs = model(inputs)
            preds = (nn.Sigmoid()(outputs) >= 0.5).long().squeeze()
            for i in range(preds.shape[0]):
                wandb.log(
                    {
                        "Input_test": wandb.Image(
                            inputs[i][np.random.randint(6)]
                            .permute(1, 2, 0)
                            .to("cpu")
                            .numpy()[:, :, 2:]
                        ),
                        "Output_test": wandb.Image(preds.to("cpu").numpy()[i]),
                    }
                )
            del inputs, outputs
            torch.cuda.empty_cache()
    output_dict = {}
    for i in range(preds.shape[0]):
        for j, val in enumerate(preds[i].flatten()):
            tile_id = f"Tile{test_ids[i]}"
            row_id = j // preds.shape[2]
            col_id = j % preds.shape[2]
            output_dict[f"{tile_id}_{row_id}_{col_id}"] = val.item()
    results = pd.DataFrame(
        list(output_dict.items()), columns=["tile_row_column", "label"]
    )
    results.to_csv("dutiful-glitter-190.csv", index=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_archive = "source_test"
    test_tiles_list = list(get_tiles(test_archive).items())
    test_dataset = NASAHarverstFieldDataset(
        test_archive, test_tiles_list, apply_transforms=False
    )
    test_loader = DataLoader(test_dataset, batch_size=13, shuffle=False, num_workers=2)

    checkpoint = torch.load("best_model_with_train_and_val_data.pt")
    model = FieldBoundariesDetector()
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    best_threshold = 0.5
    test_ids = [tile[0] for tile in test_tiles_list]
    predict(model, test_ids, test_loader, device)
