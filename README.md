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
