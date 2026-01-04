import torch
from phognet.models.phognet import PHOGNet, PHOGProcessingBlock


def test_forward_pass():
    model = PHOGNet(PHOGProcessingBlock, [2, 2, 2, 2], num_classes=9, bins=20, levels=1, nInputPlane=3)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 9)
