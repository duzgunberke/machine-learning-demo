import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import cv2
import joblib
from src.preprocessing.image_preprocessing import preprocess_image
from src.preprocessing.feature_extraction import extract_glcm_features
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLASS_NAMES = {
    0: "Pepper bell Bacterial spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___healthy",
    4: "Potato___Late_blight",
    5: "Tomato__Target_Spot",
    6: "Tomato__Tomato_mosaic_virus",
    7: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    8: "Tomato_Bacterial_spot",
    9: "Tomato_Early_blight",
    10: "Tomato_healthy",
    11: "Tomato_Late_blight",
    12: "Tomato_Leaf_Mold",
    13: "Tomato_Septoria_leaf_spot",
    14: "Tomato_Spider_mites_Two_spotted_spider_mite"
}

def load_model(model_path: str):
    logging.info(f"Model yükleniyor: {model_path}")
    try:
        model = joblib.load(model_path)
        logging.info("Model başarıyla yüklendi.")
        return model
    except Exception as e:
        logging.error(f"Model yüklenirken hata oluştu: {e}")
        return None

def preprocess_image(image):
    image_resized = cv2.resize(image, (128, 128))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    return image_gray

def predict_disease_with_alternatives(image_path: str, model, top_n: int = 3):
    logging.info(f"Resim yükleniyor: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logging.error("Resim yüklenemedi. Lütfen geçerli bir yol sağlayın.")
        return None, None

    logging.info("Resim işleniyor...")
    processed_image = preprocess_image(image)
    features = extract_glcm_features(processed_image)

    logging.info("Tahmin yapılıyor...")
    probabilities = model.predict_proba([list(features.values())])[0]
    classes = model.classes_

    top_indices = np.argsort(probabilities)[::-1][:top_n]
    top_classes = [(CLASS_NAMES[classes[i]], probabilities[i]) for i in top_indices]

    primary_prediction = top_classes[0]
    logging.info(f"Birincil Tahmin: {primary_prediction}")
    return primary_prediction, top_classes

def select_image():
    filepath = filedialog.askopenfilename(
        title="Resim Seç",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if filepath:
        logging.info(f"Seçilen dosya: {filepath}")
        primary_prediction, top_classes = predict_disease_with_alternatives(filepath, model, top_n=3)
        if primary_prediction:
            result_text = f"Tahmin edilen hastalık: {primary_prediction[0]}\n\n"
            result_text += "Alternatifler:\n"
            for alt_class, alt_prob in top_classes[1:]:
                result_text += f"  - {alt_class}: {alt_prob:.2f}\n"
            messagebox.showinfo("Tahmin Sonucu", result_text)
        else:
            messagebox.showerror("Hata", "Tahmin sırasında bir hata oluştu.")
    else:
        logging.warning("Kullanıcı dosya seçmedi.")

def main_ui():
    root = tk.Tk()
    root.title("Bitki Hastalığı Tespit Sistemi")
    root.geometry("400x300")

    label_title = tk.Label(root, text="Bitki Hastalığı Tespit Sistemi", font=("Arial", 16))
    label_title.pack(pady=10)

    label_desc = tk.Label(root, text="Bir resim seçin ve tahmin sonucunu öğrenin.", font=("Arial", 12))
    label_desc.pack(pady=5)

    btn_select_image = tk.Button(root, text="Resim Seç", command=select_image, font=("Arial", 12))
    btn_select_image.pack(pady=20)

    root.mainloop()

def main_console():
    image_path = input("Resim dosyasının tam yolunu girin: ")
    if image_path:
        primary_prediction, top_classes = predict_disease_with_alternatives(image_path, model, top_n=3)
        if primary_prediction:
            print(f"Tahmin edilen hastalık: {primary_prediction[0]}\n")
            print("Alternatifler:")
            for alt_class, alt_prob in top_classes[1:]:
                print(f"  - {alt_class}: {alt_prob:.2f}")
        else:
            print("Tahmin sırasında bir hata oluştu.")
    else:
        print("Geçerli bir resim yolu girilmedi.")

def main():
    global model
    model_path = "results/model_optimized.pkl"
    model = load_model(model_path)
    if not model:
        print("Model yüklenemedi. Lütfen modeli eğitip doğru yolu sağlayın.")
        return

    print("Bitki Hastalığı Tespit Sistemi")
    print("1. Grafik Arayüz (UI)")
    print("2. Konsol Modu (Tunnel)")
    choice = input("Seçim yapın (1 veya 2): ")

    if choice == "1":
        main_ui()
    elif choice == "2":
        main_console()
    else:
        print("Geçerli bir seçim yapılmadı. Program sonlandırılıyor.")

if __name__ == "__main__":
    main()
