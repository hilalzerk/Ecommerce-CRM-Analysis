# 🛒 E-Ticaret CRM Analizi
### RFM Segmentasyonu · CLTV Tahmini · Tahmine Dayalı Modelleme

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat-square&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)
![Lisans](https://img.shields.io/badge/Lisans-MIT-green?style=flat-square)

---

## 📌 Proje Hakkında

Bu proje, bir e-ticaret veri seti üzerinde uçtan uca müşteri analitiği gerçekleştirmektedir. **RFM analizi**, **Müşteri Yaşam Boyu Değeri (CLTV) tahmini** ve **makine öğrenmesi modelleri** kullanılarak müşteriler segmentlere ayrılmış; veri odaklı CRM stratejilerini destekleyecek uygulanabilir iş içgörüleri üretilmiştir.

---

## 🎯 Amaçlar

- **RFM çerçevesi** kullanılarak müşterilerin satın alma davranışlarına göre segmentlere ayrılması
- Olasılıksal ve istatistiksel modellerle **CLTV tahmini** yapılması
- Yüksek değerli ve kayıp riski taşıyan müşterileri tespit eden **tahmine dayalı modeller** kurulması
- Hedefli pazarlama kampanyaları için uygulanabilir içgörüler sunulması

---

## 🗂️ Proje Yapısı

```
Ecommerce-CRM-Analysis/
│
├── notebooks/
│   ├── 01_veri_on_isleme.ipynb           # Veri temizleme & özellik mühendisliği
│   ├── 02_rfm_analizi.ipynb              # RFM skorlama & müşteri segmentasyonu
│   ├── 03_cltv_tahmini.ipynb             # BG/NBD & Gamma-Gamma modelleri
│   └── 04_tahmine_dayali_modelleme.ipynb  # Kayıp & değer tahmini için ML modelleri
│
├── reports/
│   └── musteri_segmentleri_raporu.pdf    # Bulgular & görselleştirmeler özeti
│
└── README.md
```

---

## 🔍 Yöntemler & Teknikler

### 1. RFM Analizi
Müşteriler üç boyuta göre skorlanmıştır:

| Boyut | Açıklama |
|---|---|
| **Recency (R)** | Müşteri en son ne zaman alışveriş yaptı? |
| **Frequency (F)** | Ne sıklıkla alışveriş yapıyor? |
| **Monetary (M)** | Ne kadar harcama yapıyor? |

Müşteriler; *Şampiyonlar*, *Sadık Müşteriler*, *Risk Altındakiler*, *Kayıp Müşteriler* gibi segmentlere ayrılmaktadır.

### 2. CLTV Tahmini
- **BG/NBD Modeli** — Gelecekteki işlem sıklığını tahmin eder
- **Gamma-Gamma Modeli** — İşlem başına ortalama parasal değeri tahmin eder
- Nihai CLTV skoru, her iki modelin 3 ve 6 aylık tahminlerini birleştirir

### 3. Tahmine Dayalı Modelleme
- RFM skorları ve sipariş geçmişinden özellik mühendisliği
- Yüksek değerli müşteri segmentlerini tahmin eden sınıflandırma modelleri
- Precision, Recall ve ROC-AUC metrikleriyle model değerlendirmesi

---

## 🛠️ Kullanılan Teknolojiler

| Araç | Kullanım Amacı |
|---|---|
| `Python 3.10+` | Temel programlama dili |
| `Pandas / NumPy` | Veri manipülasyonu |
| `Scikit-learn` | Makine öğrenmesi modelleri |
| `Lifetimes` | BG/NBD & Gamma-Gamma CLTV modelleri |
| `Matplotlib / Seaborn` | Veri görselleştirme |
| `Jupyter Notebook` | Analiz ortamı |

---

## 📊 Temel Bulgular

- **Müşterilerin en iyi %20'si** toplam gelirin %60'ından fazlasını oluşturmaktadır
- **Şampiyonlar** segmenti, diğer segmentlere kıyasla 4 kat daha yüksek ortalama sipariş değeri sergilemektedir
- CLTV modeli, 6 aylık gelir tahmini için güçlü bir doğruluk oranı yakalamıştır
- Risk altındaki müşteriler, hedefli kampanyalarla kayıp yaşanmadan yeniden kazanılabilir

---

## 🚀 Kurulum & Kullanım

### 1. Depoyu klonlayın
```bash
git clone https://github.com/hilalzerk/Ecommerce-CRM-Analysis.git
cd Ecommerce-CRM-Analysis
```

### 2. Gerekli kütüphaneleri yükleyin
```bash
pip install pandas numpy scikit-learn lifetimes matplotlib seaborn jupyter
```

### 3. Notebook'ları çalıştırın
```bash
jupyter notebook
```
Tam analiz akışı için notebook'ları sırasıyla açın: 01 → 02 → 03 → 04

---

## 📁 Veri Seti

Analiz, aşağıdaki bilgileri içeren bir e-ticaret işlem veri seti üzerine kurulmuştur:
- Müşteri ID, sipariş tarihi, sipariş miktarı, birim fiyat
- Ülke ve ürün bilgileri
- Zaman aralığı: 2010–2011

> Veri seti kaynağı: [UCI Machine Learning Repository — Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)

---

## 👩‍💻 Geliştirici

**Hilal Zerk**  
Veri Bilimi | Python · SQL · Makine Öğrenmesi  
[GitHub](https://github.com/hilalzerk)

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında lisanslanmıştır.
