###################################################################################################
#
# Copyright (C) 2021-2025 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
EfficientNet v2 Model (Adapted for MAX78000)

Original EfficientNetV2 architecture:
- Fused-MBConv blocks in early stages (Conv3x3 fused → Project 1x1)
- MBConv blocks in later stages (Expand 1x1 → Depthwise 3x3 → SE → Project 1x1)
- Expansion ratio: 1 for Fused-MBConv1, 4 for Fused-MBConv4, 4-6 for MBConv

MAX78000 Adaptations:
- No depthwise convolutions (use regular 3x3 with reduced channels)
- No SE attention (not efficiently supported)
- Reduced channel counts to fit 64 processor limit
- Skip connections where feasible (same in/out channels, stride=1)
"""
from torch import nn

import ai8x


class AI85EfficientNetV2(nn.Module):
    """
    EfficientNet v2 Model (Adapted for MAX78000)
    
    Architecture following EfficientNetV2-S structure:
    - Stage 0: Stem Conv3x3
    - Stage 1: Fused-MBConv1 (expansion=1) - just Conv3x3
    - Stage 2: Fused-MBConv4 (expansion=4) - Conv3x3 → Project 1x1
    - Stage 3: Fused-MBConv4 (expansion=4) - Conv3x3 → Project 1x1  
    - Stage 4: MBConv4 (expansion=4) - Expand 1x1 → Conv3x3 → Project 1x1
    - Stage 5: MBConv4 (expansion=4) - Expand 1x1 → Conv3x3 → Project 1x1
    - Head: Conv1x1 → AvgPool → FC
    """
    def __init__(
            self,
            num_classes=100,
            num_channels=3,
            dimensions=(32, 32),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()

        # ========== Stem: 32x32x3 → 32x32x24 ==========
        self.stem = ai8x.FusedConv2dBNReLU(num_channels, 24, 3,
                                           stride=1, padding=1,
                                           batchnorm='Affine',
                                           bias=bias, **kwargs)

        # ========== Stage 1: Fused-MBConv1 (expansion=1) ==========
        # 32x32x24 → 32x32x24 (with residual)
        # Fused-MBConv1: just a Conv3x3 (no expansion, no project)
        self.stage1_conv = ai8x.FusedConv2dBNReLU(24, 24, 3,
                                                   stride=1, padding=1,
                                                   batchnorm='Affine',
                                                   bias=bias, **kwargs)
        self.stage1_add = ai8x.Add()

        # Downsample: 32x32 → 16x16
        self.pool1 = ai8x.MaxPool2d(kernel_size=2, stride=2)

        # ========== Stage 2: Fused-MBConv4 (expansion=4) ==========
        # 16x16x24 → 16x16x48 (no residual - different channels)
        # Fused: Conv3x3 (expand+spatial fused) → Project 1x1
        # expanded = 24 * 4 = 96, but we cap spatial at 48 for memory
        self.stage2_fused = ai8x.FusedConv2dBNReLU(24, 48, 3,
                                                    stride=1, padding=1,
                                                    batchnorm='Affine',
                                                    bias=bias, **kwargs)
        self.stage2_project = ai8x.FusedConv2dBNReLU(48, 48, 1,
                                                      stride=1, padding=0,
                                                      batchnorm='Affine',
                                                      bias=bias, **kwargs)

        # Downsample: 16x16 → 8x8
        self.pool2 = ai8x.MaxPool2d(kernel_size=2, stride=2)

        # ========== Stage 3: Fused-MBConv4 (expansion=4) ==========
        # 8x8x48 → 8x8x64 (no residual - different channels)
        # Fused: Conv3x3 → Project 1x1
        self.stage3_fused = ai8x.FusedConv2dBNReLU(48, 64, 3,
                                                    stride=1, padding=1,
                                                    batchnorm='Affine',
                                                    bias=bias, **kwargs)
        self.stage3_project = ai8x.FusedConv2dBNReLU(64, 64, 1,
                                                      stride=1, padding=0,
                                                      batchnorm='Affine',
                                                      bias=bias, **kwargs)

        # Downsample: 8x8 → 4x4
        self.pool3 = ai8x.MaxPool2d(kernel_size=2, stride=2)

        # ========== Stage 4: MBConv4 (expansion=4) ==========
        # 4x4x64 → 4x4x64 (with residual - same channels)
        # MBConv: Expand 1x1 → Spatial 3x3 → Project 1x1
        # expanded = 64 * 4 = 256, cap at 64 for MAX78000
        self.stage4_expand = ai8x.FusedConv2dBNReLU(64, 64, 1,
                                                     stride=1, padding=0,
                                                     batchnorm='Affine',
                                                     bias=bias, **kwargs)
        self.stage4_spatial = ai8x.FusedConv2dBNReLU(64, 64, 3,
                                                      stride=1, padding=1,
                                                      batchnorm='Affine',
                                                      bias=bias, **kwargs)
        self.stage4_project = ai8x.FusedConv2dBNReLU(64, 64, 1,
                                                      stride=1, padding=0,
                                                      batchnorm='Affine',
                                                      bias=bias, **kwargs)
        self.stage4_add = ai8x.Add()

        # ========== Stage 5: MBConv4 (expansion=4) ==========
        # 4x4x64 → 4x4x128 (no residual - different channels)
        # MBConv: Expand 1x1 → Spatial 3x3 → Project 1x1
        # expanded = 64 * 4 = 256, cap at 128 (use multi-pass)
        self.stage5_expand = ai8x.FusedConv2dBNReLU(64, 128, 1,
                                                     stride=1, padding=0,
                                                     batchnorm='Affine',
                                                     bias=bias, **kwargs)
        self.stage5_spatial = ai8x.FusedConv2dBNReLU(128, 128, 3,
                                                      stride=1, padding=1,
                                                      batchnorm='Affine',
                                                      bias=bias, **kwargs)
        self.stage5_project = ai8x.FusedConv2dBNReLU(128, 128, 1,
                                                      stride=1, padding=0,
                                                      batchnorm='Affine',
                                                      bias=bias, **kwargs)

        # ========== Head ==========
        # Global average pooling and classifier
        self.avg_pool = ai8x.AvgPool2d((4, 4))
        self.fc = ai8x.Linear(128, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Stem
        x = self.stem(x)              # 32x32x3 → 32x32x24

        # Stage 1: Fused-MBConv1 with residual
        identity = x
        x = self.stage1_conv(x)       # 32x32x24 → 32x32x24
        x = self.stage1_add(x, identity)  # Residual connection

        x = self.pool1(x)             # 32x32x24 → 16x16x24

        # Stage 2: Fused-MBConv4
        x = self.stage2_fused(x)      # 16x16x24 → 16x16x48
        x = self.stage2_project(x)    # 16x16x48 → 16x16x48

        x = self.pool2(x)             # 16x16x48 → 8x8x48

        # Stage 3: Fused-MBConv4
        x = self.stage3_fused(x)      # 8x8x48 → 8x8x64
        x = self.stage3_project(x)    # 8x8x64 → 8x8x64

        x = self.pool3(x)             # 8x8x64 → 4x4x64

        # Stage 4: MBConv4 with residual
        identity = x
        x = self.stage4_expand(x)     # 4x4x64 → 4x4x64
        x = self.stage4_spatial(x)    # 4x4x64 → 4x4x64
        x = self.stage4_project(x)    # 4x4x64 → 4x4x64
        x = self.stage4_add(x, identity)  # Residual connection

        # Stage 5: MBConv4 (no residual - channel change)
        x = self.stage5_expand(x)     # 4x4x64 → 4x4x128
        x = self.stage5_spatial(x)    # 4x4x128 → 4x4x128
        x = self.stage5_project(x)    # 4x4x128 → 4x4x128

        # Head
        x = self.avg_pool(x)          # 4x4x128 → 1x1x128
        x = x.view(x.size(0), -1)
        x = self.fc(x)                # 128 → num_classes

        return x


def ai85effnetv2(pretrained=False, **kwargs):
    """
    Constructs an EfficientNet v2 model for MAX78000
    
    More faithful to original EfficientNetV2:
    - Fused-MBConv blocks in early stages (Conv3x3 → Project 1x1)
    - MBConv blocks in later stages (Expand 1x1 → Conv3x3 → Project 1x1)
    - Residual connections where in_channels == out_channels
    - Proper block structure following EfficientNetV2-S
    
    MAX78000 adaptations:
    - No depthwise convolutions (regular conv with reduced channels)
    - No SE attention
    - Capped expansion channels to fit 64 processors (128 max with multi-pass)
    """
    assert not pretrained
    return AI85EfficientNetV2(**kwargs)


models = [
    {
        'name': 'ai85effnetv2',
        'min_input': 1,
        'dim': 2,
    },
]