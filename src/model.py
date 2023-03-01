import torch.nn as nn
import segmentation_models_pytorch as smp

# import torchvision


class FieldBoundariesDetector(nn.Module):
    def __init__(
        self, timesteps=6, encoder_name="resnet50", in_channels=5, num_classes=1
    ):
        super().__init__()
        self.skip_connection = nn.Sequential(
            nn.Conv3d(
                in_channels,
                6 * in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.AdaptiveAvgPool3d((1, None, None)),
        )

        self.temporal_features_aggregator = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels * 3,
                kernel_size=(timesteps // 3, 1, 1),
                stride=(timesteps // 3, 1, 1),
                padding=0,
            ),
            nn.BatchNorm3d(in_channels * 3),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.Conv3d(
                in_channels * 3,
                in_channels * 6,
                kernel_size=(timesteps // 2, 1, 1),
                stride=(timesteps // 2, 1, 1),
                padding=0,
            ),
            nn.BatchNorm3d(in_channels * 6),
            nn.GELU(),
            nn.Dropout3d(0.2),
            nn.Conv3d(
                in_channels,
                in_channels * 6,
                kernel_size=(timesteps, 1, 1),
                stride=(timesteps, 1, 1),
                padding=0,
            ),
            nn.BatchNorm3d(in_channels * 6),
            nn.GELU(),
        )

        self.unet = smp.Unet(
            encoder_name,
            in_channels=timesteps * in_channels,
            classes=num_classes,
            encoder_depth=5,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, channels, timesteps, height, width]
        skip_x = self.skip_connection(x)  # [batch_size, 6*channels, 1, height, width]
        x = self.temporal_features_aggregator(
            x
        )  # ==> [batch_size, 6*channels, 1, height, width]
        # x = torchvision.ops.drop_block2d(x, p=0.2, block_size = 5)
        x = (x + skip_x).squeeze()
        x = self.unet(x)
        return x
