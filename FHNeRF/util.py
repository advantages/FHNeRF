import numpy as np
import torch
from torch._C import dtype
from typing import Dict
import sys

DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()


    coordinates = coordinates / (img.shape[2] - 1) - 0.5

    coordinates *= 2

    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))


def bpp(image, model):
    """Computes size in bits per pixel of model.

    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape)  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels


def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(65536.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()



def get_clamped_psnr(img, img_recon):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    # return psnr(img, clamp_image(img_recon))
    return psnr(img,img_recon)


def mean(list_):
    return np.mean(list_)
