from torch.utils.data import Dataset
from data import generate_tile_data


class NASAHarverstFieldDataset(Dataset):
    def __init__(self, archive, tiles_list, apply_transforms=True):
        self.archive = archive
        self.tiles_list = tiles_list
        self.apply_transforms = apply_transforms

    def __len__(self):
        return len(self.tiles_list)

    def __getitem__(self, idx):
        return generate_tile_data(
            self.tiles_list[idx][0],
            self.tiles_list[idx][1],
            self.archive,
            self.apply_transforms,
        )
