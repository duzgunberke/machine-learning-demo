from skimage.feature import graycomatrix, graycoprops
import numpy as np

"""
Resimden GLCM özelliklerini çıkarır.
:param image: Gri tonlamalı resim
:param distances: Mesafeler
:param angles: Açı değerleri
:return: Özellik sözlüğü
"""
def extract_glcm_features(image: np.ndarray, distances=[1], angles=[0]) -> dict:
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean()
    }
    return features
