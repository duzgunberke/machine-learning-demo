# Bitki Hastalığı Tespit Sistemi

Bu proje, makine öğrenimi ve görüntü işleme tekniklerini kullanarak bitki yapraklarındaki hastalıkları tespit etmek amacıyla geliştirilmiştir. Proje, **Plant Village Dataset** ile eğitilmiş bir model ve kullanıcı dostu bir grafik arayüz (GUI) içerir.

---

## 🚀 Proje Amacı

- **Problem Tanımı**:  
  Tarımsal ürünlerde görülen hastalıklar, üretim kayıplarına neden olabilir. Bu proje, hızlı ve doğru hastalık teşhisi ile bu sorunu çözmeyi hedefler.
- **Hedefler**:  
  1. Görüntü işleme teknikleri ile görsellerden hastalık tespiti.  
  2. Makine öğrenimiyle yüksek doğruluk oranına sahip bir model geliştirmek.  
  3. Kullanıcı dostu bir grafik arayüz sağlamak.

---

## 📁 Klasör Yapısı

```plaintext
.
├── data/                   # Veri seti
│   ├── raw/                # Ham veri seti
│   └── processed/          # İşlenmiş veri seti
├── src/                    # Kaynak kod
│   ├── preprocessing/      # Veri ön işleme ve özellik çıkarımı
│   ├── models/             # Model eğitimi ve tahmini
│   ├── app.py              # Ana uygulama dosyası
│   └── test.py             # Tahmin testi için dosya
├── results/                # Çıktılar
│   ├── model_optimized.pkl # Eğitilmiş model
│   ├── confusion_matrix.png # Confusion matrix görseli
│   └── training_accuracy_plot.png # Eğitim doğruluk grafiği
└── README.md               # Proje dokümantasyonu
```

# 🛠️ Kullanılan Teknolojiler

### Python

### **Kütüphaneler**
- **Görüntü İşleme**: `OpenCV`, `scikit-image`
- **Makine Öğrenimi**: `scikit-learn`
- **Görselleştirme**: `Matplotlib`, `Seaborn`
- **GUI**: `Tkinter`

---

# 📊 Proje Süreci

## **1. Veri Yükleme**
- **Amaç**:  
  Resimleri ve sınıflarını ham veri setinden yüklemek.
- **Yöntem**:
  - `os` modülü kullanılarak alt klasörlerden resimler ve etiketler okunur.
  - Her alt klasör bir sınıf olarak kabul edilir.

---

## **2. Veri Ön İşleme**
- **Amaç**:  
  Resimleri makine öğrenimi modeline uygun hale getirmek.
- **Adımlar**:
  - Görseller gri tonlamaya dönüştürüldü.
  - Tüm görseller `(128x128)` boyutuna ölçeklendirildi.

---

## **3. Özellik Çıkarımı**
- **Amaç**:  
  Görsellerden metrik özellikler çıkarmak.
- **Yöntem**:
  - **GLCM (Gray Level Co-occurrence Matrix)**:
    - Kontrast, korelasyon, enerji ve homojenlik özellikleri elde edildi.

---

## **4. Veri Artırma**
- **Amaç**:  
  Daha fazla veri üreterek modelin genelleme yeteneğini artırmak.
- **Yöntemler**:
  - Döndürme
  - Aynalama
  - Parlaklık değişimi
  - Gürültü ekleme

---

## **5. Model Eğitimi**
- **Kullanılan Algoritma**:  
  `Random Forest Classifier`
- **Optimizasyon**:
  - **Grid Search** ile en iyi hiperparametrelerin bulunması.
  - **K-Fold Cross-Validation** (5 katlı) ile genelleme yeteneğinin artırılması.
- **Performans Metrikleri**:
  - Doğruluk
  - Confusion Matrix
  - Precision
  - Recall

---

## **6. Tahmin**
- **Tahmin Süreci**:
  1. Kullanıcı bir resim seçer.
  2. Resim işlenir ve özellikler çıkarılır.
  3. Eğitilmiş model, tahmini ve olasılıkları döndürür.
- **GUI**:
  - `Tkinter` kullanılarak kullanıcı dostu bir grafik arayüz geliştirildi.

---

# 🏆 Sonuçlar
- **Ortalama Doğruluk**:  
  %92.3
- **Çıktılar**:
  - **Eğitilmiş Model**: `results/model_optimized.pkl`
  - **Confusion Matrix Görseli**: `results/confusion_matrix.png`
  - **Eğitim Süreci Grafiği**: `results/training_accuracy_plot.png`

---

# 🔧 Geliştirme Önerileri
1. **Derin Öğrenme**:  
   CNN gibi yöntemlerle doğruluk oranını artırmak.
2. **Gerçek Zamanlı Uygulama**:  
   Modeli mobil veya web tabanlı sistemlere entegre etmek.
3. **Daha Büyük Veri Setleri**:  
   Daha fazla bitki türü ve hastalık içeren veri setleri ile genelleme yeteneğini artırmak.

---

# 🚀 Kurulum ve Kullanım

## **Gereksinimler**
- **Python 3.9+**
- Gerekli kütüphaneler:
  ```bash
  pip install -r requirements.txt

##  Projenin Çalıştırılması

### **1. Veriyi Yükleyin**
- Veri setinizi `data/raw/` klasörüne yerleştirin.

### **2. Modeli Eğitin**
```bash
python src/app.py
```

### **3. Tahmin yapın**
```bash
python src/test.py  
