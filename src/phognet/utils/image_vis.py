from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def is_grayscale(image: Image.Image) -> bool:
    return image.mode in ("L", "1")


def convert_grayscale_to_color(image: Image.Image, colormap: str = "jet") -> Image.Image:
    gray_array = np.array(image)
    norm_gray_array = gray_array / 255.0
    cmap = plt.get_cmap(colormap)
    color_image = cmap(norm_gray_array)
    color_image = (color_image[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(color_image)
