###############################################################
# CRM ANALİZİ - ADIM 2: RFM SEGMENTASYONU
###############################################################

# =====================================================================
# BU ADIMDA NE YAPACAĞIZ?
# ---------------------------------------------------------------------
# RFM, müşterileri 3 temel boyutta değerlendiren bir segmentasyon yöntemidir:
#
# RECENCY   (Yenilik)    : Müşteri en son ne zaman alışveriş yaptı?
#                              → Ne kadar yakın zamanda ise o kadar iyi
# FREQUENCY (Sıklık)     : Müşteri kaç kez alışveriş yaptı?
#                              → Ne kadar çok ise o kadar iyi
# MONETARY  (Parasal)    : Müşteri toplamda ne kadar harcadı?
#                              → Ne kadar çok ise o kadar iyi
#
# ADIMLAR:
# 1. Veriyi hazırla ve temizle
# 2. RFM metriklerini hesapla
# 3. RFM skorlarını ata (1-5 arası)
# 4. Segmentleri oluştur
# 5. Segmentleri analiz et ve görselleştir
# 6. Sonuçları kaydet
# =====================================================================

import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
matplotlib.use('Agg')

plt.rcParams['figure.figsize'] = (14, 7)
sns.set_style("whitegrid")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("crm_project/data/raw/data.csv", encoding="latin-1")

dataframe = df_.copy()
dataframe.head()

# Dosya yolları
BASE_DIR = "crm_project"
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# =====================================================================
# YARDIMCI FONKSİYONLAR
# =====================================================================


def create_rfm(dataframe, csv=False):
    """
    Uçtan uca RFM analizi yaparak müşteri segmentlerini oluşturur.

    Bu fonksiyon sırasıyla şu adımları gerçekleştirir:
    1. Veriyi hazırlar ve temizler
    2. RFM metriklerini hesaplar (Recency, Frequency, Monetary)
    3. RFM skorlarını atar (1-5 arası)
    4. Müşterileri segmentlere ayırır

    Parameters
    ----------
    dataframe : pd.DataFrame
        Ham e-ticaret verisi. Şu kolonları içermeli:
        - InvoiceNo   : Fatura numarası
        - InvoiceDate : Fatura tarihi
        - CustomerID  : Müşteri ID
        - Quantity    : Ürün adedi
        - UnitPrice   : Birim fiyat

    csv : bool, optional
        True ise sonuçları 'rfm.csv' olarak kaydeder.
        Varsayılan: False

    Returns
    -------
    pd.DataFrame
        Müşteri bazında aşağıdaki kolonları içeren tablo:
        - recency   : Son alışverişten bu yana geçen gün sayısı (az = iyi)
        - frequency : Toplam alışveriş sayısı (çok = iyi)
        - monetary  : Toplam harcama tutarı (çok = iyi)
        - segment   : Müşteri segmenti (champions, at_risk vb.)

    Segments
    --------
        champions           : Son zamanda alışveriş yaptı, sık ve çok harcıyor
        loyal_customers     : Sık alışveriş yapıyor, son zamanlarda da aktif
        potential_loyalists : Yeni veya az aktif ama potansiyeli var
        new_customers       : Çok yeni geldi, henüz az alışveriş
        promising           : Yakın geçmişte geldi ama çok sık değil
        need_attention      : Orta seviyede aktif ama dikkat gerekiyor
        about_to_sleep      : Seyrek alışveriş, yakında kaybolabilir
        at_risk             : Eskiden sık alışveriş yapıyordu, artık yok
        cant_loose          : Eskiden değerli müşteriydi, kaybetmemeli
        hibernating         : Hem nadir hem eski alışveriş, uykuda

    Example
    -------
        df = pd.read_csv("data.csv", encoding="latin-1")
        rfm = create_rfm(df, csv=True)
        print(rfm.head())
    """

    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["UnitPrice"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["InvoiceNo"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 9, 11)
    dataframe['InvoiceDate'] = pd.to_datetime(dataframe['InvoiceDate'])

    rfm = dataframe.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (today_date - date.max()).days,
        'InvoiceNo': lambda num: num.nunique(),
        'TotalPrice': lambda price: price.sum()
    })
    rfm.columns = ['recency', 'frequency', 'monetary']
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_SCORE"] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm


output_path = os.path.join(OUTPUT_DIR, "rfm.csv")
rfm.to_csv(output_path, index=True)
print(f"✅ RFM sonuçları kaydedildi: {output_path}")


# =====================================================================
# ANALİZ VE GÖRSELLEŞTİRME
# =====================================================================

# Segment bazında ortalama değerler ve müşteri sayısı
segment_analysis = rfm.groupby('segment').agg(
    musteri_sayisi=('recency', 'count'),
    ort_recency=('recency', 'mean'),
    ort_frequency=('frequency', 'mean'),
    ort_monetary=('monetary', 'mean')
).round(1).sort_values('musteri_sayisi', ascending=False)

print(f"\n{segment_analysis}")


# --- Görsel 1: Segment Dağılımı (Pasta + Bar) ---

plt.close('all')

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('RFM Müşteri Segmentasyonu', fontsize=16, fontweight='bold')

colors = ['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#f39c12',
          '#e74c3c', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']

segment_counts = segment_analysis['musteri_sayisi']

# 1a. Pasta grafiği
axes[0].pie(
    segment_counts.values,
    labels=segment_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    pctdistance=0.85 # Metinlerin iç içe girmemesi için
)
axes[0].set_title('Segment Dağılımı (%)')

# 1b. Bar grafiği (müşteri sayısı)
bars = axes[1].barh(segment_counts.index, segment_counts.values, color=colors)
axes[1].set_xlabel('Müşteri Sayısı')
axes[1].set_title('Segment Bazında Müşteri Sayısı')
axes[1].bar_label(bars, fmt='%d', padding=5)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "02_rfm_segment_distribution.png"), dpi=72, bbox_inches='tight')


# --- Görsel 2: Segment Metrikleri Karşılaştırması ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Segment Bazında RFM Metrikleri', fontsize=16, fontweight='bold')

metrics = ['ort_recency', 'ort_frequency', 'ort_monetary']
titles = ['Ort. Recency (Gün) ↓ İyi', 'Ort. Frequency (Adet) ↑ İyi', 'Ort. Monetary (GBP) ↑ İyi']

for ax, metric, title in zip(axes, metrics, titles):
    data = segment_analysis[metric].sort_values(ascending=(metric == 'ort_recency'))
    bars = ax.bar(data.index, data.values, color='steelblue', alpha=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Segment')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, data.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.0f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "02_rfm_segment_metrics.png"), dpi=72, bbox_inches='tight')


# =====================================================================
# ÖNERİLER
# =====================================================================

actions = {
    'champions': '🏆 Ödüllendirme programı & Erken erişim teklifleri sunun',
    'loyal_customers': '💛 Üyelik avantajları & sadakat programına davet edin',
    'potential_loyalists': '🌟 Kişiselleştirilmiş öneri & onboarding kampanyası başlatın',
    'new_customers': '🆕 Hoş geldin e-postası & ilk alışveriş indirimi sunun',
    'promising': '📈 Marka hikayesini paylaşın & indirim kodu gönderin',
    'need_attention': '⚡ Limitli süreli tekliflerle aktifleştirin',
    'about_to_sleep': '⏰ Hatırlatıcı e-posta & "bizi özlediniz mi?" kampanyası',
    'at_risk': '⚠️  Özel indirim & kişisel temas kurun (email/SMS)',
    'cant_loose': '🚨 Kişisel iletişim & VIP hizmet teklifleri yapın',
    'hibernating': '😴 Reaktivasyon kampanyası veya süreci kapat'
}

for segment in segment_analysis.index:
    count = segment_analysis.loc[segment, 'musteri_sayisi']
    action = actions.get(segment, 'Strateji belirlenmedi')
    print(f"\n{segment.upper()} ({count:,} müşteri)")
    print(f"   → {action}")





