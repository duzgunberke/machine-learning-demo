import os
import pandas as pd
import logging
from src.preprocessing.data_loader import load_images_and_labels
from src.preprocessing.image_preprocessing import preprocess_image
from src.preprocessing.feature_extraction import extract_glcm_features
from src.models.image_augmentation import augment_image
from src.models.train_model import train_and_save_model

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    csv_path = "data/processed/features_augmented.csv"

    if os.path.exists(csv_path):
        logging.info("Önceden işlenmiş CSV dosyası bulunuyor, yükleniyor...")
        df_features = pd.read_csv(csv_path)
        logging.info(f"{len(df_features)} özellik kaydı yüklendi.")
    else:
        logging.info("CSV dosyası bulunamadı, resimler yükleniyor...")
        images, labels = load_images_and_labels("data/raw/")
        logging.info(f"{len(images)} resim yüklendi.")

        logging.info("Resimler artırılmaya ve işlenmeye başlanıyor...")
        augmented_images = []
        augmented_labels = []

        for idx, (img, label) in enumerate(zip(images, labels), start=1):
            logging.info(f"Resim {idx}/{len(images)} işleniyor: Etiket={label}")

            augmented_set = augment_image(img)
            augmented_images.extend(augmented_set)
            augmented_labels.extend([label] * len(augmented_set))

            logging.info(f"Resim {idx} için {len(augmented_set)} artırma tamamlandı.")

        logging.info(f"Artırılmış toplam resim sayısı: {len(augmented_images)}")

        logging.info("Özellik çıkarımı başlıyor...")
        features = []

        for idx, img in enumerate(augmented_images, start=1):
            logging.info(f"Resim {idx}/{len(augmented_images)} işleniyor...")

            processed_img = preprocess_image(img)
            feature = extract_glcm_features(processed_img)
            features.append(feature)

            logging.info(f"Resim {idx} işlendi ve özellikler çıkarıldı.")

        logging.info("Özellikler DataFrame'e dönüştürülüyor...")
        df_features = pd.DataFrame(features)
        df_features["label"] = augmented_labels

        if not os.path.exists("data/processed"):
            os.makedirs("data/processed")
        df_features.to_csv(csv_path, index=False)
        logging.info("Özellikler çıkarıldı ve kaydedildi.")

    logging.info("Model eğitimi başlıyor...")
    train_and_save_model(df_features.drop("label", axis=1), df_features["label"], output_path="results/model_optimized.pkl")
    logging.info("Model eğitimi tamamlandı ve kaydedildi.")

if __name__ == "__main__":
    main()
