import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm  
import numpy as np

def train_with_optimization(features: pd.DataFrame, labels: pd.Series, output_path: str = "results/model_optimized.pkl"):
    """
    Randomized Search ve K-Fold Cross-Validation ile kapsamlı model eğitimi.
    :param features: Özellikler
    :param labels: Etiketler
    :param output_path: Modelin kaydedileceği dosya yolu
    """
    logging.info("Model eğitimi başlıyor...")

    logging.info("Veri eğitim ve test setlerine ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    logging.info(f"Veri başarıyla ayrıldı: {len(X_train)} eğitim, {len(X_test)} test.")

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    logging.info(f"Hiperparametre seti: {param_dist}")

    logging.info("Randomized Search ile hiperparametre optimizasyonu başlıyor...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, 
                                       n_iter=100, cv=skf, scoring='accuracy', n_jobs=-1, random_state=42)

    try:
        logging.info("Randomized Search başlatılıyor...")
        random_search.fit(X_train, y_train)
        logging.info("Randomized Search tamamlandı.")
    except Exception as e:
        logging.error(f"Randomized Search sırasında hata oluştu: {e}")
        return

    best_model = random_search.best_estimator_
    logging.info(f"En iyi hiperparametreler: {random_search.best_params_}")
    logging.info(f"Randomized Search ortalama doğruluk: {random_search.best_score_:.4f}")

    logging.info("Model test seti üzerinde değerlendiriliyor...")
    y_pred = best_model.predict(X_test)
    logging.info("Test seti sonuçları:")
    logging.info(classification_report(y_test, y_pred))

    logging.info("Confusion Matrix çiziliyor...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.title("Confusion Matrix")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.savefig("results/confusion_matrix.png")
    logging.info("Confusion Matrix kaydedildi.")

    logging.info(f"Model kaydediliyor: {output_path}")
    joblib.dump(best_model, output_path)
    logging.info(f"Model başarıyla kaydedildi: {output_path}")

    logging.info("Eğitim sürecinin görselleştirilmesi başlatılıyor...")
    visualize_training_process(random_search.cv_results_)

def visualize_training_process(cv_results):
    """
    Eğitim sürecini görselleştirmek için doğruluk grafikleri.
    :param cv_results: Randomized Search çapraz doğrulama sonuçları
    """
    logging.info("Eğitim süreci grafiği oluşturuluyor...")
    plt.figure(figsize=(10, 6))
    mean_test_scores = cv_results['mean_test_score']
    plt.plot(mean_test_scores, label="Ortalama Doğruluk")
    plt.title("Hiperparametre Setleri için Doğruluk")
    plt.xlabel("Hiperparametre Kombinasyonu")
    plt.ylabel("Doğruluk")
    plt.legend()
    plt.savefig("results/training_accuracy_plot.png")
    logging.info("Eğitim süreci grafiği kaydedildi.")
