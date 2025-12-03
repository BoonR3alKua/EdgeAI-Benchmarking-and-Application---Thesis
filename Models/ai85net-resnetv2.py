"""
ResNetV2 implementation for MAX78000
- Uses ai8x.Add() for residual connections
- Uses FusedConv2dBNReLU for proper BatchNorm
- Channel progression: 16 -> 32 -> 64 -> 128
- Architecture: [1, 1, 2, 2] blocks per stage
- Two variants: 32x32 (CIFAR-10/100) and 64x64 (ASL)
"""
import torch.nn as nn
import ai8x
import torch._dynamo
torch._dynamo.config.suppress_errors = True

class AI85ResNetV2(nn.Module):
    """
    ResNetV2 for MAX78000 - Optimized for CIFAR-10/100
    Channel progression: 16 -> 32 -> 64 -> 128 (128 only at small spatial size)
    Input size: 32x32
    Architecture: [1, 1, 2, 2] blocks per stage (4 stages, 6 blocks total)
    
    Spatial flow:
      conv1:    32x32 -> 32x32 (no pool, 16ch)
      stage1:   32x32 -> 32x32 (16->32ch, 1 block)
      stage2:   32x32 -> 16x16 (32->64ch, 1 block with pool)
      stage3:   16x16 -> 8x8   (64->64ch, 2 blocks, first with pool)
      stage4:   8x8   -> 4x4   (64->128ch, 2 blocks, first with pool)
      final:    4x4   -> 1x1   (avg pool)
      fc:       128 -> num_classes
    """
    def __init__(
            self,
            num_classes=10,
            num_channels=3,
            dimensions=(32, 32),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()

        # Initial conv (3 -> 16 channels, 32x32, no pool - already small input)
        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, stride=1,
                                          padding=1, bias=bias, **kwargs)

        # Stage 1: 16 -> 32 channels, 32x32 (1 block, no pooling)
        self.layer1_0_conv = ai8x.FusedConv2dBNReLU(16, 32, 3, stride=1, padding=1, bias=bias,
                                                     batchnorm='NoAffine', **kwargs)
        self.layer1_0_shortcut = ai8x.FusedConv2dBNReLU(16, 32, 1, stride=1, padding=0, bias=bias,
                                                         batchnorm='NoAffine', **kwargs)
        self.layer1_0_add = ai8x.Add()

        # Stage 2: 32 -> 64 channels, 32x32 -> 16x16 (1 block with pooling)
        self.layer2_0_conv = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, pool_size=2, pool_stride=2,
                                                            stride=1, padding=1, bias=bias,
                                                            batchnorm='NoAffine', **kwargs)
        self.layer2_0_shortcut = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 1, pool_size=2, pool_stride=2,
                                                                stride=1, padding=0, bias=bias,
                                                                batchnorm='NoAffine', **kwargs)
        self.layer2_0_add = ai8x.Add()

        # Stage 3: 64 -> 64 channels, 16x16 -> 8x8 (2 blocks, first with pooling)
        # Block 3.0: with pooling
        self.layer3_0_conv = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                            stride=1, padding=1, bias=bias,
                                                            batchnorm='NoAffine', **kwargs)
        self.layer3_0_shortcut = ai8x.FusedMaxPoolConv2d(64, 64, 1, pool_size=2, pool_stride=2,
                                                          stride=1, padding=0, bias=bias, **kwargs)
        self.layer3_0_add = ai8x.Add()
        # Block 3.1: identity shortcut (no pooling, same channels)
        self.layer3_1_conv = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1, bias=bias,
                                                     batchnorm='NoAffine', **kwargs)
        self.layer3_1_add = ai8x.Add()

        # Stage 4: 64 -> 128 channels, 8x8 -> 4x4 (2 blocks, first with pooling)
        # Block 4.0: with pooling, channel expansion
        self.layer4_0_conv = ai8x.FusedMaxPoolConv2dBNReLU(64, 128, 3, pool_size=2, pool_stride=2,
                                                            stride=1, padding=1, bias=bias,
                                                            batchnorm='NoAffine', **kwargs)
        self.layer4_0_shortcut = ai8x.FusedMaxPoolConv2dBNReLU(64, 128, 1, pool_size=2, pool_stride=2,
                                                                stride=1, padding=0, bias=bias,
                                                                batchnorm='NoAffine', **kwargs)
        self.layer4_0_add = ai8x.Add()
        # Block 4.1: identity shortcut (no pooling, same channels)
        self.layer4_1_conv = ai8x.FusedConv2dBNReLU(128, 128, 3, stride=1, padding=1, bias=bias,
                                                     batchnorm='NoAffine', **kwargs)
        self.layer4_1_add = ai8x.Add()

        # Global Average Pool: 4x4 -> 1x1
        self.avg_pool = ai8x.AvgPool2d(4, stride=1)

        # Fully connected layer (128 -> num_classes)
        self.fc = ai8x.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=bias,
                              wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Stem: 3x32x32 -> 16x32x32 (no pool for 32x32 input)
        x = self.conv1(x)

        # Stage 1: 16x32x32 -> 32x32x32 (1 block)
        x_res = self.layer1_0_shortcut(x)
        x = self.layer1_0_conv(x)
        x = self.layer1_0_add(x, x_res)

        # Stage 2: 32x32x32 -> 64x16x16 (1 block with pool)
        x_res = self.layer2_0_shortcut(x)
        x = self.layer2_0_conv(x)
        x = self.layer2_0_add(x, x_res)

        # Stage 3: 64x16x16 -> 64x8x8 (2 blocks)
        # Block 3.0: with pool
        x_res = self.layer3_0_shortcut(x)
        x = self.layer3_0_conv(x)
        x = self.layer3_0_add(x, x_res)
        # Block 3.1: identity
        x_res = x
        x = self.layer3_1_conv(x)
        x = self.layer3_1_add(x, x_res)

        # Stage 4: 64x8x8 -> 128x4x4 (2 blocks)
        # Block 4.0: with pool
        x_res = self.layer4_0_shortcut(x)
        x = self.layer4_0_conv(x)
        x = self.layer4_0_add(x, x_res)
        # Block 4.1: identity
        x_res = x
        x = self.layer4_1_conv(x)
        x = self.layer4_1_add(x, x_res)

        # Global Average Pool: 128x4x4 -> 128x1x1
        x = self.avg_pool(x)

        # Classification: 128x1x1 -> num_classes
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x


class AI85ResNetV2x64(nn.Module):
    """
    ResNetV2 for MAX78000 - Optimized for ASL Alphabet Dataset
    Channel progression: 16 -> 32 -> 64 -> 128 (128 only at small spatial size)
    Input size: 64x64
    Architecture: [1, 1, 2, 2] blocks per stage (4 stages, 6 blocks total)
    Pool in first layer like original ResNet stem
    
    Spatial flow:
      conv1:    64x64 -> 32x32 (pool here, 16ch)
      stage1:   32x32 -> 32x32 (16->32ch, 1 block)
      stage2:   32x32 -> 16x16 (32->64ch, 1 block with pool)
      stage3:   16x16 -> 8x8   (64->64ch, 2 blocks, first with pool)
      stage4:   8x8   -> 4x4   (64->128ch, 2 blocks, first with pool)
      final:    4x4   -> 1x1   (avg pool)
      fc:       128 -> num_classes
    """
    def __init__(
            self,
            num_classes=27,  # ASL: A-Z + unknown (del/space/nothing merged)
            num_channels=3,
            dimensions=(64, 64),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()

        # Initial conv with pool (3 -> 16 channels, 64x64 -> 32x32)
        self.conv1 = ai8x.FusedMaxPoolConv2dReLU(num_channels, 16, 3, pool_size=2, pool_stride=2,
                                                  stride=1, padding=1, bias=bias, **kwargs)

        # Stage 1: 16 -> 32 channels, 32x32 (1 block, no pooling)
        self.layer1_0_conv = ai8x.FusedConv2dBNReLU(16, 32, 3, stride=1, padding=1, bias=bias,
                                                     batchnorm='NoAffine', **kwargs)
        self.layer1_0_shortcut = ai8x.FusedConv2dBNReLU(16, 32, 1, stride=1, padding=0, bias=bias,
                                                         batchnorm='NoAffine', **kwargs)
        self.layer1_0_add = ai8x.Add()

        # Stage 2: 32 -> 64 channels, 32x32 -> 16x16 (1 block with pooling)
        self.layer2_0_conv = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, pool_size=2, pool_stride=2,
                                                            stride=1, padding=1, bias=bias,
                                                            batchnorm='NoAffine', **kwargs)
        self.layer2_0_shortcut = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 1, pool_size=2, pool_stride=2,
                                                                stride=1, padding=0, bias=bias,
                                                                batchnorm='NoAffine', **kwargs)
        self.layer2_0_add = ai8x.Add()

        # Stage 3: 64 -> 64 channels, 16x16 -> 8x8 (2 blocks, first with pooling)
        # Block 3.0: with pooling
        self.layer3_0_conv = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                            stride=1, padding=1, bias=bias,
                                                            batchnorm='NoAffine', **kwargs)
        self.layer3_0_shortcut = ai8x.FusedMaxPoolConv2d(64, 64, 1, pool_size=2, pool_stride=2,
                                                          stride=1, padding=0, bias=bias, **kwargs)
        self.layer3_0_add = ai8x.Add()
        # Block 3.1: identity shortcut (no pooling, same channels)
        self.layer3_1_conv = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1, bias=bias,
                                                     batchnorm='NoAffine', **kwargs)
        self.layer3_1_add = ai8x.Add()

        # Stage 4: 64 -> 128 channels, 8x8 -> 4x4 (2 blocks, first with pooling)
        # Block 4.0: with pooling, channel expansion
        self.layer4_0_conv = ai8x.FusedMaxPoolConv2dBNReLU(64, 128, 3, pool_size=2, pool_stride=2,
                                                            stride=1, padding=1, bias=bias,
                                                            batchnorm='NoAffine', **kwargs)
        self.layer4_0_shortcut = ai8x.FusedMaxPoolConv2dBNReLU(64, 128, 1, pool_size=2, pool_stride=2,
                                                                stride=1, padding=0, bias=bias,
                                                                batchnorm='NoAffine', **kwargs)
        self.layer4_0_add = ai8x.Add()
        # Block 4.1: identity shortcut (no pooling, same channels)
        self.layer4_1_conv = ai8x.FusedConv2dBNReLU(128, 128, 3, stride=1, padding=1, bias=bias,
                                                     batchnorm='NoAffine', **kwargs)
        self.layer4_1_add = ai8x.Add()

        # Global Average Pool: 4x4 -> 1x1
        self.avg_pool = ai8x.AvgPool2d(4, stride=1)

        # Fully connected layer (128 -> num_classes)
        self.fc = ai8x.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=bias,
                              wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Stem: 3x64x64 -> 16x32x32 (pool in first layer)
        x = self.conv1(x)

        # Stage 1: 16x32x32 -> 32x32x32 (1 block)
        x_res = self.layer1_0_shortcut(x)
        x = self.layer1_0_conv(x)
        x = self.layer1_0_add(x, x_res)

        # Stage 2: 32x32x32 -> 64x16x16 (1 block with pool)
        x_res = self.layer2_0_shortcut(x)
        x = self.layer2_0_conv(x)
        x = self.layer2_0_add(x, x_res)

        # Stage 3: 64x16x16 -> 64x8x8 (2 blocks)
        # Block 3.0: with pool
        x_res = self.layer3_0_shortcut(x)
        x = self.layer3_0_conv(x)
        x = self.layer3_0_add(x, x_res)
        # Block 3.1: identity
        x_res = x
        x = self.layer3_1_conv(x)
        x = self.layer3_1_add(x, x_res)

        # Stage 4: 64x8x8 -> 128x4x4 (2 blocks)
        # Block 4.0: with pool
        x_res = self.layer4_0_shortcut(x)
        x = self.layer4_0_conv(x)
        x = self.layer4_0_add(x, x_res)
        # Block 4.1: identity
        x_res = x
        x = self.layer4_1_conv(x)
        x = self.layer4_1_add(x, x_res)

        # Global Average Pool: 128x4x4 -> 128x1x1
        x = self.avg_pool(x)

        # Classification: 128x1x1 -> num_classes
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x


def ai85resnetv2_18(pretrained=False, **kwargs):
    """
    Constructs a ResNetV2 model for MAX78000 (CIFAR-10/100).
    Channel progression: 16 -> 32 -> 64 -> 128
    Input size: 32x32
    Architecture: [1, 1, 2, 2] blocks per stage
    """
    assert not pretrained
    return AI85ResNetV2(**kwargs)


def ai85resnetv2_18x64(pretrained=False, **kwargs):
    """
    Constructs a ResNetV2x64 model for MAX78000 (ASL Alphabet Dataset).
    Channel progression: 16 -> 32 -> 64 -> 128
    Input size: 64x64
    Architecture: [1, 1, 2, 2] blocks per stage
    Pool in first layer like original ResNet stem
    """
    assert not pretrained
    return AI85ResNetV2x64(**kwargs)


models = [
    {
        'name': 'ai85resnetv2_18',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85resnetv2_18x64',
        'min_input': 1,
        'dim': 2,
    },
]