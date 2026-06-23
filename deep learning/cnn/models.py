import torch.nn as nn
from torchvision import models as tv_models


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, use_bn: bool = True):
        super().__init__()
        padding = k // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGGLikeSmall(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, use_bn: bool = True, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(in_channels, 32, use_bn=use_bn),
            ConvBNReLU(32, 32, use_bn=use_bn),
            nn.MaxPool2d(2),
            ConvBNReLU(32, 64, use_bn=use_bn),
            ConvBNReLU(64, 64, use_bn=use_bn),
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128, use_bn=use_bn),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def make_resnet18(
    num_classes: int = 10,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    """ResNet18 for FashionMNIST.
    Uses ImageNet-pretrained weights (optional).
    FashionMNIST -> RGB conversion γίνεται στα transforms.
    """
    weights = None
    if pretrained:
        try:
            weights = tv_models.ResNet18_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = tv_models.resnet18(weights=weights)
    except Exception as e:
        print(f"[WARN] Could not load pretrained weights ({e}). Falling back to random init.")
        model = tv_models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = False

    return model
