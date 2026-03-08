###############################################################
# CRM ANALİZİ - ADIM 3: CLTV HESAPLAMA
###############################################################

# =====================================================================
# BU ADIMDA NE YAPACAĞIZ?
# ---------------------------------------------------------------------
# CLTV (Customer Lifetime Value = Müşteri Yaşam Boyu Değeri),
# bir müşterinin şirketle ilişkisi boyunca sağlayacağı tahmini toplam değerdir.
#
# FORMÜL TABANLI CLTV:
# ─────────────────────────────────────────────────────────
#
#  1. Average Order Value (Ortalama Sipariş Değeri)
#     = Toplam Ciro / Toplam İşlem Sayısı
#
#  2. Purchase Frequency (Satın Alma Sıklığı)
#     = Toplam İşlem Sayısı / Toplam Benzersiz Müşteri Sayısı
#
#  3. Repeat Rate (Tekrar Oranı)
#     = Birden fazla alışveriş yapan müşteri / Toplam müşteri
#
#  4. Churn Rate (Kayıp Oranı)
#     = 1 - Repeat Rate
#
#  5. Profit Margin (Kâr Marjı)
#     = Toplam Ciro × kâr_marji_orani (varsayılan: %10)
#
#  6. Customer Value (Müşteri Değeri)
#     = Average Order Value × Purchase Frequency
#
#  7. CLTV = (Customer Value / Churn Rate) × Profit Margin
#
# NOT: Bu formül tabanlı yaklaşım geçmişe dönük bir hesaplamadır.
#      Tahmin yapmak için 04_cltv_prediction.py'yi kullanacağız.
# =====================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['figure.figsize'] = (15, 8)
sns.set_style("whitegrid")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Dosya yolları
BASE_DIR = "crm_project"
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


df_ = pd.read_csv("crm_project/data/raw/data.csv", encoding="latin-1")

dataframe = df_.copy()
dataframe.head()
dataframe.describe()

# =====================================================================
# YARDIMCI FONKSİYONLAR
# =====================================================================
def outlier_thresholds(dataframe, variable):
    """
    Aykırı değer eşiklerini hesaplar (IQR yöntemi, %1-%99 yüzdelik).

    Not: Standart IQR %25-%75 yerine %1-%99 kullanıyoruz çünkü
    e-ticaret verisinde uç değerler gerçek müşteri davranışını yansıtıyor.
    """
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * interquantile_range
    up_limit = quartile3 + 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



def create_cltv_c(dataframe, profit=0.10):

    # Veriyi hazırlama
    dataframe = dataframe[~dataframe["InvoiceNo"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe.dropna(inplace=True)
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "UnitPrice")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["UnitPrice"]
    cltv_c = dataframe.groupby('CustomerID').agg({'InvoiceNo': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'TotalPrice': lambda x: x.sum()})
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']
    # avg_order_value
    cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']
    # purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]
    # repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    # profit_margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit
    # Customer Value
    cltv_c['customer_value'] = cltv_c['avg_order_value'] * cltv_c["purchase_frequency"]
    # Customer Lifetime Value
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * profit
    # Segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c

# =====================================================================
# SEGMENT BAZINDA ÖZET
# =====================================================================

segment_summary = cltv_c.groupby('segment').agg(
    musteri_sayisi=('cltv', 'count'),
    ort_cltv=('cltv', 'mean'),
    toplam_cltv=('cltv', 'sum'),
    min_cltv=('cltv', 'min'),
    max_cltv=('cltv', 'max')
)
print(segment_summary)

# =====================================================================
# GÖRSELLEŞTİRME
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CLTV Analizi', fontsize=16, fontweight='bold')

# 1 CLTV Dağılımı (histogram)
cltv_vals = cltv_c[cltv_c['cltv'] < cltv_c['cltv'].quantile(0.95)]['cltv']
axes[0, 0].hist(cltv_vals, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
axes[0, 0].set_title('CLTV Dağılımı (alt 95. yüzdelik)')
axes[0, 0].set_xlabel('CLTV')
axes[0, 0].set_ylabel('Müşteri Sayısı')

# 2 Segment Bazında Müşteri Sayısı
seg_counts = cltv_c['segment'].value_counts().sort_index()
colors_seg = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
bars = axes[0, 1].bar(seg_counts.index, seg_counts.values, color=colors_seg, alpha=0.9)
axes[0, 1].set_title('Segment Bazında Müşteri Sayısı')
axes[0, 1].set_xlabel('Segment')
axes[0, 1].set_ylabel('Müşteri Sayısı')
for bar, val in zip(bars, seg_counts.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{val:,}', ha='center', fontsize=11, fontweight='bold')

# 3 Segment Bazında Ortalama CLTV
seg_mean_cltv = cltv_c.groupby('segment')['cltv'].mean().sort_index()
bars2 = axes[1, 0].bar(seg_mean_cltv.index, seg_mean_cltv.values, color=colors_seg, alpha=0.9)
axes[1, 0].set_title('Segment Bazında Ortalama CLTV')
axes[1, 0].set_xlabel('Segment')
axes[1, 0].set_ylabel('Ortalama CLTV')
for bar, val in zip(bars2, seg_mean_cltv.values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', fontsize=10)

# 4 Toplam CLTV'nin Segmentlere Dağılımı (pasta)
seg_total_cltv = cltv_c.groupby('segment')['cltv'].sum()
axes[1, 1].pie(seg_total_cltv.values, labels=[f'Segment {s}' for s in seg_total_cltv.index],
               autopct='%1.1f%%', colors=colors_seg, startangle=90)
axes[1, 1].set_title('Toplam CLTV\'nin Segmentlere Dağılımı')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "03_cltv_analysis.png"), dpi=72, bbox_inches='tight')


# =====================================================================
# ANALİTİK ÖNGÖRÜLER
# =====================================================================

total_cltv = cltv_c['cltv'].sum()
seg_a_cltv = cltv_c[cltv_c['segment'] == 'A']['cltv'].sum()
seg_a_count = cltv_c[cltv_c['segment'] == 'A'].shape[0]
total_customers = cltv_c.shape[0]

print(f"""
Toplam Beklenen CLTV : {total_cltv:,.0f}
Toplam Müşteri       : {total_customers:,}
Kişi Başına CLTV     : {total_cltv/total_customers:,.0f}

Segment A (En Değerli):
Müşteri Sayısı  : {seg_a_count:,} ({seg_a_count/total_customers*100:.1f}% müşteri)
Toplam CLTV     : {seg_a_cltv:,.0f} ({seg_a_cltv/total_cltv*100:.1f}% gelir)

80/20 Kuralı Analizi:
Üst %20 müşteri toplam gelirin {seg_a_cltv/total_cltv*100:.0f}%'ini oluşturuyor.
""")

# =====================================================================
# KAYDET
# =====================================================================

output_path = os.path.join(OUTPUT_DIR, "cltv_calculated.csv")
cltv_c.to_csv(output_path)

