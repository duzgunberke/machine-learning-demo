import cv2
import joblib
from src.preprocessing.image_preprocessing import preprocess_image
from src.preprocessing.feature_extraction import extract_glcm_features

def predict_disease(image_path: str, model_path: str):
    """
    Bir resimden hastalık tahmini yapar.
    :param image_path: Resim dosyasının yolu
    :param model_path: Eğitilmiş modelin yolu
    :return: Tahmin sonucu
    """
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)
    features = extract_glcm_features(processed_image)
    model = joblib.load(model_path)
    prediction = model.predict([list(features.values())])
    return prediction[0]
