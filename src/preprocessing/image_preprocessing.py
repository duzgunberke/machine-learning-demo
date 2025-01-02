import cv2
import numpy as np
from typing import Tuple

def preprocess_image(image: np.ndarray, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Resmi gri tonlamaya çevirir ve yeniden boyutlandırır.
    :param image: Orijinal resim
    :param size: Boyutlandırılacak hedef boyut (genişlik, yükseklik)
    :return: İşlenmiş resim
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, size)
    return resized_image
