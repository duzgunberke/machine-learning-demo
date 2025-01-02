import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
import joblib
import numpy as np

def train_and_save_model(features: pd.DataFrame, labels: pd.Series, output_path: str, n_splits: int = 5):
    logging.info("Model eğitimi başlıyor...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_predictions = []
    all_labels = []

    for fold, (train_index, test_index) in enumerate(kf.split(features)):
        logging.info(f"Katlama {fold + 1} için eğitim başlıyor...")
        
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        all_predictions.extend(predictions)
        all_labels.extend(y_test)
        
        logging.info(f"Katlama {fold + 1} sonuçları:")
        logging.info(classification_report(y_test, predictions))
    
    logging.info("Tüm katlamalar için genel değerlendirme:")
    logging.info(classification_report(all_labels, all_predictions))

    joblib.dump(model, output_path)
    logging.info(f"Model kaydedildi: {output_path}")

