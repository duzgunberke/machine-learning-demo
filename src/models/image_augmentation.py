import cv2
import numpy as np

"""
Bir resme çeşitli veri artırma teknikleri uygular.
:param image: Orijinal resim
:return: Artırılmış resimlerin listesi
"""
def augment_image(image: np.ndarray) -> list:
    augmented_images = []

    augmented_images.append(image)

    for angle in [-15, 15]:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        augmented_images.append(rotated)

    augmented_images.append(cv2.flip(image, 1))  # Yatay
    augmented_images.append(cv2.flip(image, 0))  # Dikey

    for alpha in [0.8, 1.2]:  
        bright = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented_images.append(bright)

    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    augmented_images.append(noisy_image)

    return augmented_images
