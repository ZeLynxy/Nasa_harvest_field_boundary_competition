import os
from dotenv import load_dotenv
from radiant_mlhub import Dataset
import shutil
import rasterio as rio
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import torch

import torchvision.transforms as transforms
#from typing import List, Any, Callable, Tuple

DATASET_ID = "nasa_rwanda_field_boundary_competition"
BASE_INPUT_DATA_PATH = "data/input"
ARCHIVES = ["source_train", "source_test", "labels_train"]


ITEMS = {
    archive: f"{BASE_INPUT_DATA_PATH}/{DATASET_ID}/{DATASET_ID}_{archive}"
    for archive in ARCHIVES
}


def download_and_unpack_data_from_ml_hub() -> None:
    load_dotenv()
    dataset = Dataset.fetch(DATASET_ID)
    dataset.download(
        output_dir=f"{BASE_INPUT_DATA_PATH}/{DATASET_ID}", if_exists="skip"
    )
    for archive in ARCHIVES:
        full_path = f"{BASE_INPUT_DATA_PATH}/{DATASET_ID}/{DATASET_ID}_{archive}.tar.gz"
        shutil.unpack_archive(full_path, f"{BASE_INPUT_DATA_PATH}/{DATASET_ID}")


def clean_string(s: str) -> str:
    """
    extract the tile id and timestamp from a source image folder
    e.g extract 'ID_YYYY_MM' from 'nasa_rwanda_field_boundary_competition_source_train_ID_YYYY_MM'
    """
    s = s.replace(f"{DATASET_ID}_source_", "").split("_")[1:]
    return "_".join(s)


def get_tiles(archive):
    archive_tiles = [clean_string(s) for s in next(os.walk(ITEMS.get(archive)))[1]]
    tiles_dict = {}
    for archive_tile in archive_tiles:
        tile_id = archive_tile.split("_")[0]
        tiles_dict.setdefault(tile_id, []).append(archive_tile)
    for tile_id in tiles_dict:
        tiles_dict[tile_id].sort(key=lambda x: int(x.split("_")[2]))
    return tiles_dict


def aggregate_bands_and_transform(tile_id, timestamps, archive="source_train"):
    images = []
    for timestamp in timestamps:
        blue_raster = rio.open(
            f"{ITEMS.get(archive)}/{DATASET_ID}_{archive}_{timestamp}/B01.tif"
        )
        green_raster = rio.open(
            f"{ITEMS.get(archive)}/{DATASET_ID}_{archive}_{timestamp}/B02.tif"
        )
        red_raster = rio.open(
            f"{ITEMS.get(archive)}/{DATASET_ID}_{archive}_{timestamp}/B03.tif"
        )
        nir_raster = rio.open(
            f"{ITEMS.get(archive)}/{DATASET_ID}_{archive}_{timestamp}/B04.tif"
        )

        red_band = red_raster.read(1)
        green_band = green_raster.read(1)
        blue_band = blue_raster.read(1)
        nir_band = nir_raster.read(1)

        # Add Enhanced Vegetation Index (EVI) which is modified version of NDVI that is less
        # sensitive to atmospheric interference and Combine the 5 bands into a single ENRGB image
        evi_band = 2.5 * (
            (nir_band - red_band) / (nir_band + 6 * red_band - 7.5 * blue_band + 1)
        )
        norm_bands = [
            exposure.rescale_intensity(
                band, in_range=(np.min(band), np.max(band)), out_range=(0, 1)
            )
            for band in [blue_band, green_band, red_band, nir_band, evi_band]
        ]
        enrgb = np.dstack(norm_bands[::-1])  # EVI, NIR, R, G, B

        # plt.imshow(enrgb[:, :, 2:])
        # plt.show()

        # plt.imshow(enrgb[:, :, 1:4])
        # plt.show()

        # plt.imshow(enrgb[:, :, 0], cmap="viridis")
        # plt.colorbar()
        # plt.show()

        images.append(enrgb)

    mask = None
    if archive == "source_train":
        mask = rio.open(
            f"{ITEMS.get('labels_train')}/{DATASET_ID}_labels_train_{tile_id}/raster_labels.tif"
        ).read(1)
        # plt.gray()
        # plt.imshow(mask)
        # plt.show()
        # print(mask)
    return np.stack(images, axis=0), mask


def transform(images, mask, apply_transforms):
    base_augmentations = [transforms.Lambda(lambda x: x)]
    input_augmentations = [
        transforms.Lambda(lambda x: x),
    ]
    if apply_transforms:
        base_augmentations = [
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.Lambda(
                        lambda x: torch.rot90(
                            x, k=np.random.randint(-3, 4), dims=[-2, -1]
                        )
                    )
                ],
                p=0.5,
            ),
            transforms.RandomApply([transforms.RandomRotation(45)], p=0.2),
        ]

        input_augmentations = [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=(1, 1.5),
                        contrast=(0.8, 1.3),
                        saturation=0.2,
                        hue=0.2,
                    )
                ],
                p=0.3,
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(kernel_size=3),
                ],
                p=0.35,
            ),
        ]

    # apply same  transformations to images (accross different timestamps) and mask.
    input_channels = np.split(images, images.shape[-1], axis=-1)

    if mask is not None:
        input_channels.append(mask[np.newaxis, ..., np.newaxis])
    images_and_mask = np.concatenate(input_channels, axis=0)

    images_and_mask = torch.from_numpy(images_and_mask).permute(0, 3, 1, 2)

    transformed_images_and_mask = transforms.Compose(base_augmentations)(
        images_and_mask
    )

    if mask is not None:
        transformed_images, transformed_mask = (
            transformed_images_and_mask[:-1, :, :, :],
            transformed_images_and_mask[-1, :, :, :].long(),
        )
        transformed_images = transforms.Compose(input_augmentations)(transformed_images)
    else:
        transformed_images = transforms.Compose(input_augmentations)(
            transformed_images_and_mask
        )
        transformed_mask = None
    transformed_images = transformed_images.view(
        images.shape[0], images.shape[-1], images.shape[1], -1
    )
    # for t in range(transformed_images.shape[0]):
    #   plt.imshow(transformed_images[t].permute(1, 2, 0).numpy()[:, :, 2:])
    #   plt.show()
    #   if mask is not None:
    #     plt.gray()
    #     plt.imshow(transformed_mask.squeeze().numpy())
    #     plt.show()
    return transformed_images, transformed_mask


def generate_tile_data(tile_id, timestamps, archive, apply_transforms=True):
    images, mask = aggregate_bands_and_transform(tile_id, timestamps, archive)
    images, mask = transform(images, mask, apply_transforms)
    if mask:return images, mask
    else:return images


def get_tile_ids(tiles):
    return list(set(tile.split("_")[0] for tile in tiles))


if __name__ == "__main__":
    download_and_unpack_data_from_ml_hub()
    print(get_tile_ids(get_tiles("source_train")))
