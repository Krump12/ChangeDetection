import torch

from src.models.change_model import SiameseUNet


def test_model_forward_shape():
    model = SiameseUNet(base_channels=4)
    image_a = torch.rand(2, 3, 128, 128)
    image_b = torch.rand(2, 3, 128, 128)
    logits = model(image_a, image_b)
    assert logits.shape == (2, 1, 128, 128)


def test_model_forward_shape_with_configurable_non_square_size():
    model = SiameseUNet(base_channels=4)
    image_a = torch.rand(1, 3, 64, 96)
    image_b = torch.rand(1, 3, 64, 96)
    logits = model(image_a, image_b)
    assert logits.shape == (1, 1, 64, 96)
