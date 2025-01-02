import os
import cv2
from typing import List, Tuple
import numpy as np

"""
Klasör yapısındaki resimleri ve etiketleri yükler.
:param folder: Ana klasörün yolu
:return: (Resimler, Etiketler)
"""
def load_images_and_labels(folder: str) -> Tuple[List[np.ndarray], List[str]]:
    images, labels = [], []
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                images.append(image)
                labels.append(class_name)
    return images, labels

if __name__ == "__main__":
    images, labels = load_images_and_labels("../../data/raw/")
    print(f"{len(images)} resim yüklendi.")
    print(f"Örnek etiketler: {set(labels)}")
