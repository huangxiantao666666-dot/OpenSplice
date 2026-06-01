"""
Self-contained simOPA scorer — Object Placement Assessment.

4-channel ResNet18 that evaluates how "natural" a composited image looks.
Score range: 0 (unreasonable) to 1 (reasonable).

Based on: OPA (Object Placement Assessment), BCMI Lab, arXiv 2107.01889
Weight file placed at: OpenSplice/checkpoints/simopa.pth
"""

import logging
import pathlib
import numpy as np
import torch
import torch.nn as nn
import cv2
import PIL.Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _make_resnet18_backbone() -> nn.Sequential:
    """Build 4ch ResNet18 and return as Sequential matching the OPA checkpoint layout.

    The OPA checkpoint stores backbone as a flat Sequential:
      [0]=conv1(4→64), [1]=bn1, [2]=relu, [3]=maxpool,
      [4]=layer1, [5]=layer2, [6]=layer3, [7]=layer4
    Only [0]=conv1 is 4-channel; the rest is standard ResNet18.
    """
    # Build a standard ResNet18 first, then modify conv1 to 4ch
    block = BasicBlock
    layers = [2, 2, 2, 2]
    inplanes = 64

    # Conv1: 4 channels (RGB + mask)
    conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    bn1 = nn.BatchNorm2d(64)
    relu = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # layer1
    layer1 = nn.Sequential()
    downsample = None
    block1 = block(inplanes, 64, 1, downsample)
    layer1.add_module('0', block1)
    inplanes = 64 * block.expansion
    block1_1 = block(inplanes, 64, 1, None)
    layer1.add_module('1', block1_1)

    # layer2
    layer2 = nn.Sequential()
    downsample2 = nn.Sequential(
        nn.Conv2d(inplanes, 128 * block.expansion, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(128 * block.expansion),
    )
    block2_0 = block(inplanes, 128, 2, downsample2)
    layer2.add_module('0', block2_0)
    inplanes = 128 * block.expansion
    block2_1 = block(inplanes, 128, 1, None)
    layer2.add_module('1', block2_1)

    # layer3
    layer3 = nn.Sequential()
    downsample3 = nn.Sequential(
        nn.Conv2d(inplanes, 256 * block.expansion, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(256 * block.expansion),
    )
    block3_0 = block(inplanes, 256, 2, downsample3)
    layer3.add_module('0', block3_0)
    inplanes = 256 * block.expansion
    block3_1 = block(inplanes, 256, 1, None)
    layer3.add_module('1', block3_1)

    # layer4
    layer4 = nn.Sequential()
    downsample4 = nn.Sequential(
        nn.Conv2d(inplanes, 512 * block.expansion, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(512 * block.expansion),
    )
    block4_0 = block(inplanes, 512, 2, downsample4)
    layer4.add_module('0', block4_0)
    inplanes = 512 * block.expansion
    block4_1 = block(inplanes, 512, 1, None)
    layer4.add_module('1', block4_1)

    backbone = nn.Sequential()
    backbone.add_module('0', conv1)
    backbone.add_module('1', bn1)
    backbone.add_module('2', relu)
    backbone.add_module('3', maxpool)
    backbone.add_module('4', layer1)
    backbone.add_module('5', layer2)
    backbone.add_module('6', layer3)
    backbone.add_module('7', layer4)

    return backbone


class ObjectPlaceNet(nn.Module):
    """simOPA model — matches the OPA checkpoint structure exactly."""
    def __init__(self):
        super().__init__()
        self.backbone = _make_resnet18_backbone()
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)
        self.prediction_head = nn.Linear(512, 2, bias=False)

    def forward(self, img_cat):
        # img_cat: [B, 4, 256, 256]
        feat = self.backbone(img_cat)          # [B, 512, 8, 8]
        feat = self.avgpool1x1(feat)           # [B, 512, 1, 1]
        feat = feat.flatten(1)                 # [B, 512]
        return self.prediction_head(feat)      # [B, 2]


# ─── Scorer singleton ─────────────────────────────────────────────────────────

_scorer = None


def _get_scorer(device: str = 'cpu') -> ObjectPlaceNet | None:
    global _scorer
    if _scorer is not None:
        _scorer.to(device)
        return _scorer

    weight_paths = [
        pathlib.Path(__file__).parent.parent / 'checkpoints' / 'simopa.pth',
        pathlib.Path('checkpoints/simopa.pth'),
    ]
    weight = None
    for p in weight_paths:
        if p.exists():
            weight = str(p)
            break

    if weight is None:
        logger.warning('simOPA weight not found at checkpoints/simopa.pth')
        return None

    try:
        logger.info('Loading simOPA from %s ...', weight)
        model = ObjectPlaceNet()
        state = torch.load(weight, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model = model.eval().to(device)
        _scorer = model
        logger.info('simOPA ready (device=%s).', device)
        return _scorer
    except Exception as e:
        logger.warning('Failed to load simOPA: %s', e)
        return None


# ─── Public API ───────────────────────────────────────────────────────────────

def score_composition(
    composite: np.ndarray,
    mask: np.ndarray,
    device: str = 'cpu',
) -> float | None:
    """Score how natural a composited image looks.

    Args:
        composite: RGB image (H, W, 3) uint8.
        mask: Binary mask (H, W) uint8 (0 or 1).
        device: 'cpu' or 'cuda'.

    Returns:
        Score in [0, 1] (higher = more natural), or None if model unavailable.
    """
    model = _get_scorer(device)
    if model is None:
        return None

    try:
        if mask.max() <= 1:
            mask_8u = (mask * 255).astype(np.uint8)
        else:
            mask_8u = mask.astype(np.uint8)

        composite_pil = PIL.Image.fromarray(composite)
        mask_pil = PIL.Image.fromarray(mask_8u, mode='L')

        # NOTE: simOPA was trained WITHOUT ImageNet normalization —
        # only Resize + ToTensor (values in [0, 1]). Using normalization
        # shifts the input distribution and produces constant 0.0 scores.
        t_compose = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        img_t = t_compose(composite_pil)   # [3, 256, 256] — values [0, 1]
        mask_t = t_compose(mask_pil)       # [1, 256, 256] — values [0, 1]
        cat = torch.cat([img_t, mask_t], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(cat)
            probs = torch.softmax(logits, dim=-1)
            score = probs[0, 1].item()

        return float(score)  # full precision, display formats elsewhere
    except Exception as e:
        logger.warning('simOPA score failed: %s', e)
        return None
