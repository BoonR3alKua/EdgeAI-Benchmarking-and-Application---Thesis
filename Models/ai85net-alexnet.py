"""
AlexNet implementation for MAX78000
Faithful adaptation of the original AlexNet architecture for 64×64 inputs
Uses ai8x fused modules following MAX78000 best practices

Original AlexNet architecture (1998):
- Conv layers: 96 → 256 → 384 → 384 → 256
- 3 MaxPool layers (after Conv1, Conv2, Conv5)
- 3 FC layers: 4096 → 4096 → 1000

MAX78000 adaptation for 64×64:
- Conv layers: 64 → 192 → 384 → 256 → 256
- 3 MaxPool layers (integrated with Conv1, Conv2, Conv5)
- 3 FC layers: 256 → 512 → 256 → num_classes
"""
from torch import nn
import ai8x


class AI85AlexNet(nn.Module):
    """
    AlexNet adapted for MAX78000 with 64×64 inputs
    
    Architecture follows original AlexNet design:
    - 5 convolutional layers with ReLU activation
    - MaxPooling after layers 1, 2, and 5
    - Global Average Pooling
    - 3 fully connected layers with ReLU (final layer uses wide mode)
    
    Spatial dimension flow (for 64×64 input):
    Input:  64×64×3
    Conv1:  64×64 → MaxPool(2×2/2) → 32×32×64
    Conv2:  32×32 → MaxPool(2×2/2) → 16×16×192
    Conv3:  16×16 (no pool) → 16×16×384
    Conv4:  16×16 (no pool) → 16×16×256
    Conv5:  16×16 → MaxPool(2×2/2) → 8×8×256
    AvgPool: 8×8 → 1×1×256 (global average pooling)
    FC1:    256 → 512 (with ReLU)
    FC2:    512 → 256 (with ReLU)
    FC3:    256 → num_classes (wide mode for output)
    
    Note: For 32×32 input (regular CIFAR-10), output before avgpool will be 4×4
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(64, 64), bias=False, **kwargs):
        super().__init__()
        
        # Extract padding for first layer (respecting MAX78000 constraint: 0-2)
        first_layer_padding = min(kwargs.pop('padding', 1), 2)

        # Layer 1: Conv 3×3×3→64 + ReLU + MaxPool(2×2/2)
        # Original AlexNet: Conv 11×11, stride 4 + MaxPool(3×3/2)
        self.conv1 = ai8x.FusedMaxPoolConv2dReLU(
            num_channels, 64, 3,
            pool_size=2, pool_stride=2,
            stride=1, padding=first_layer_padding,
            bias=bias, **kwargs
        )
        
        # Layer 2: Conv 3×3×64→192 + ReLU + MaxPool(2×2/2)
        # Original AlexNet: Conv 5×5×96→256 + MaxPool(3×3/2)
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(
            64, 192, 3,
            pool_size=2, pool_stride=2,
            stride=1, padding=1,
            bias=bias, **kwargs
        )
        
        # Layer 3: Conv 3×3×192→384 + ReLU
        # Original AlexNet: Conv 3×3×256→384
        self.conv3 = ai8x.FusedConv2dReLU(
            192, 384, 3,
            stride=1, padding=1,
            bias=bias, **kwargs
        )
        
        # Layer 4: Conv 3×3×384→256 + ReLU
        # Original AlexNet: Conv 3×3×384→384
        self.conv4 = ai8x.FusedConv2dReLU(
            384, 256, 3,
            stride=1, padding=1,
            bias=bias, **kwargs
        )
        
        # Layer 5: Conv 3×3×256→256 + ReLU + MaxPool(2×2/2)
        # Original AlexNet: Conv 3×3×384→256 + MaxPool(3×3/2)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(
            256, 256, 3,
            pool_size=2, pool_stride=2,
            stride=1, padding=1,
            bias=bias, **kwargs
        )
        
        # Global Average Pooling: Automatically adjust based on input dimensions
        # For 64×64 input: 8×8 → 1×1
        # For 32×32 input: 4×4 → 1×1
        # Calculate the spatial size after all convolutions and pooling
        # 64×64 → pool → 32×32 → pool → 16×16 → 16×16 → 16×16 → pool → 8×8
        # 32×32 → pool → 16×16 → pool → 8×8 → 8×8 → 8×8 → pool → 4×4
        avgpool_size = dimensions[0] // 8  # 64//8=8, 32//8=4
        self.avgpool = ai8x.AvgPool2d(avgpool_size, stride=1)
        
        # Fully Connected Layer 1: 256 → 512 + ReLU
        # Original AlexNet: 9216 → 4096
        self.fc1 = ai8x.FusedLinearReLU(256, 512, bias=bias, **kwargs)
        
        # Fully Connected Layer 2: 512 → 256 + ReLU
        # Original AlexNet: 4096 → 4096
        self.fc2 = ai8x.FusedLinearReLU(512, 256, bias=bias, **kwargs)
        
        # Fully Connected Layer 3: 256 → num_classes (wide mode for output)
        # Original AlexNet: 4096 → 1000
        self.fc3 = ai8x.Linear(256, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):
        """Forward propagation"""
        # Convolutional feature extraction
        x = self.conv1(x)  # 64×64 → 32×32×64 (or 32×32 → 16×16×64 for CIFAR-10)
        x = self.conv2(x)  # 32×32 → 16×16×192 (or 16×16 → 8×8×192 for CIFAR-10)
        x = self.conv3(x)  # 16×16 → 16×16×384 (or 8×8 → 8×8×384 for CIFAR-10)
        x = self.conv4(x)  # 16×16 → 16×16×256 (or 8×8 → 8×8×256 for CIFAR-10)
        x = self.conv5(x)  # 16×16 → 8×8×256 (or 8×8 → 4×4×256 for CIFAR-10)
        
        # Global average pooling
        x = self.avgpool(x)  # 8×8 → 1×1×256 (or 4×4 → 1×1×256 for CIFAR-10)
        
        # Flatten
        x = x.view(x.size(0), -1)  # 1×1×256 → 256
        
        # Fully connected classification layers
        x = self.fc1(x)  # 256 → 512
        x = self.fc2(x)  # 512 → 256
        x = self.fc3(x)  # 256 → num_classes
        
        return x


class AI85AlexNetSmall(nn.Module):
    """
    Smaller AlexNet variant for MAX78000 - ALL CHANNELS ≤ 64
    RECOMMENDED for resource-constrained deployment (no streaming required)
    
    Follows original AlexNet structure with reduced channel counts:
    
    Spatial dimension flow:
    Input:  64×64
    Conv1:  64×64 → MaxPool → 32×32  (3→32 channels)
    Conv2:  32×32 → MaxPool → 16×16  (32→64 channels)
    Conv3:  16×16 (no pool)   16×16  (64→64 channels)
    Conv4:  16×16 (no pool)   16×16  (64→64 channels)
    Conv5:  16×16 → MaxPool → 8×8    (64→64 channels)
    AvgPool: 8×8 → 1×1               (64 channels)
    FC1:    64 → 256 (with ReLU)
    FC2:    256 → 128 (with ReLU)
    FC3:    128 → num_classes (wide mode)
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(64, 64), bias=False, **kwargs):
        super().__init__()
        
        # Extract padding for first layer (respecting MAX78000 constraint: 0-2)
        first_layer_padding = min(kwargs.pop('padding', 1), 2)

        # Layer 1: Conv 3×3×3→32 + ReLU + MaxPool(2×2/2)
        self.conv1 = ai8x.FusedMaxPoolConv2dReLU(
            num_channels, 32, 3,
            pool_size=2, pool_stride=2,
            stride=1, padding=first_layer_padding,
            bias=bias, **kwargs
        )
        
        # Layer 2: Conv 3×3×32→64 + ReLU + MaxPool(2×2/2)
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(
            32, 64, 3,
            pool_size=2, pool_stride=2,
            stride=1, padding=1,
            bias=bias, **kwargs
        )
        
        # Layer 3: Conv 3×3×64→64 + ReLU
        self.conv3 = ai8x.FusedConv2dReLU(
            64, 64, 3,
            stride=1, padding=1,
            bias=bias, **kwargs
        )
        
        # Layer 4: Conv 3×3×64→64 + ReLU
        self.conv4 = ai8x.FusedConv2dReLU(
            64, 64, 3,
            stride=1, padding=1,
            bias=bias, **kwargs
        )
        
        # Layer 5: Conv 3×3×64→64 + ReLU + MaxPool(2×2/2)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(
            64, 64, 3,
            pool_size=2, pool_stride=2,
            stride=1, padding=1,
            bias=bias, **kwargs
        )
        
        # Global Average Pooling: Automatically adjust based on input dimensions
        # For 64×64 input: 8×8 → 1×1
        # For 32×32 input: 4×4 → 1×1
        avgpool_size = dimensions[0] // 8  # 64//8=8, 32//8=4
        self.avgpool = ai8x.AvgPool2d(avgpool_size, stride=1)
        
        # Fully Connected Layer 1: 64 → 256 + ReLU
        self.fc1 = ai8x.FusedLinearReLU(64, 256, bias=bias, **kwargs)
        
        # Fully Connected Layer 2: 256 → 128 + ReLU
        self.fc2 = ai8x.FusedLinearReLU(256, 128, bias=bias, **kwargs)
        
        # Fully Connected Layer 3: 128 → num_classes (wide mode for output)
        self.fc3 = ai8x.Linear(128, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):
        """Forward propagation"""
        # Convolutional feature extraction
        x = self.conv1(x)  # 64×64 → 32×32×32
        x = self.conv2(x)  # 32×32 → 16×16×64
        x = self.conv3(x)  # 16×16 → 16×16×64
        x = self.conv4(x)  # 16×16 → 16×16×64
        x = self.conv5(x)  # 16×16 → 8×8×64
        
        # Global average pooling
        x = self.avgpool(x)  # 8×8 → 1×1×64
        
        # Flatten
        x = x.view(x.size(0), -1)  # 1×1×64 → 64
        
        # Fully connected classification layers
        x = self.fc1(x)  # 64 → 256
        x = self.fc2(x)  # 256 → 128
        x = self.fc3(x)  # 128 → num_classes
        
        return x


def ai85alexnet(pretrained=False, **kwargs):
    """
    Constructs an AlexNet model adapted for MAX78000/MAX78002 with 64×64 inputs.
    
    Faithful adaptation of the original AlexNet architecture (Krizhevsky et al., 2012)
    with adjustments for the MAX78000 hardware constraints and smaller input size.
    
    Architecture:
    - 5 Convolutional layers: 64 → 192 → 384 → 256 → 256 channels
    - 3 MaxPool operations (after Conv1, Conv2, Conv5)
    - 3 Fully Connected layers: 256 → 512 → 256 → num_classes
    - Global Average Pooling instead of spatial flatten
    
    NOTE: Layers with >64 channels require streaming mode in synthesis YAML
    
    Use with: --model ai85alexnet --dataset CIFAR10_64
    """
    assert not pretrained
    return AI85AlexNet(**kwargs)


def ai85alexnet_small(pretrained=False, **kwargs):
    """
    Constructs a smaller AlexNet variant for MAX78000/MAX78002.
    
    RECOMMENDED for resource-constrained deployment and faster training.
    All channels ≤ 64, eliminating the need for streaming mode.
    
    Architecture:
    - 5 Convolutional layers: 32 → 64 → 64 → 64 → 64 channels
    - 3 MaxPool operations (after Conv1, Conv2, Conv5)
    - 3 Fully Connected layers: 64 → 256 → 128 → num_classes
    - Global Average Pooling
    
    Benefits:
    - No streaming mode required (all channels ≤ 64)
    - Faster synthesis and deployment
    - Lower power consumption
    - Maintains original AlexNet structure
    
    Use with: --model ai85alexnet_small --dataset CIFAR10_64
    """
    assert not pretrained
    return AI85AlexNetSmall(**kwargs)


models = [
    {
        'name': 'ai85alexnet',
        'min_input': 64,
        'dim': 2,
    },
    {
        'name': 'ai85alexnet_small',
        'min_input': 64,
        'dim': 2,
    },
]