from torchvision import transforms
import torch


def random_noise(noise: float = 0.1):
    """
    Returns a transform that applies random noise to the input image.
    """
    return lambda x: x + torch.randn_like(x) * noise


def random_noise_affine(
    noise: float = 0.1,
    degrees: float = 10,
    translate: float = 0.1,
    scale: float = 0.1,
):
    """
    Returns a transform that applies random noise and affine transformations to the input image.
    """
    return transforms.Compose(
        [
            random_noise(noise),
            transforms.RandomAffine(
                degrees=degrees,
                translate=(translate, translate),
                scale=(1 - scale, 1 + scale),
            ),
        ]
    )
