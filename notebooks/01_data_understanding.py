###############################################################
# CRM ANALİZİ - ADIM 1: VERİYİ ANLAMA (Data Understanding)
###############################################################

# =====================================================================
# BU ADIMDA NE YAPACAĞIZ?
# ---------------------------------------------------------------------
# 1. Veri setini yükleyeceğiz
# 2. Genel yapısını inceleyeceğiz (boyut, sütunlar, veri tipleri)
# 3. Eksik değerleri tespit edeceğiz
# 4. Temel istatistikleri çıkaracağız
# 5. Görsel analizler yapacağız
# 6. Sonraki adımlar için veriyi hazırlayacağız
#
# VERİ SETİ: UCI Online Retail (Kaggle - carrie1/ecommerce-data)
# https://www.kaggle.com/datasets/carrie1/ecommerce-data
#
# DEĞİŞKENLER:
# InvoiceNo    : Fatura numarası. C ile başlıyorsa iptal.
# StockCode    : Ürün kodu
# Description  : Ürün açıklaması
# Quantity     : Satılan adet
# InvoiceDate  : Fatura tarihi
# UnitPrice    : Birim fiyat (GBP)
# CustomerID   : Müşteri ID
# Country      : Ülke
# =====================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

# Görselleştirme ayarları
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# =====================================================================
# 1. VERİYİ YÜKLEME
# =====================================================================

df_ = pd.read_csv("crm_project/data/raw/data.csv", encoding="latin-1")


df = df_.copy()  # Orijinal veriyi korumak için kopyasını kullanıyoruz

# =====================================================================
# 2. VERİNİN GENEL YAPISI
# =====================================================================

def check_df(dataframe, head=5):
    print("########## Sahepe ########")
    print(dataframe.shape)

    print("########## Types ########")
    print(dataframe.dtypes)

    print("########## Head ########")
    print(dataframe.head(head))

    print("########## Tail ########")
    print(dataframe.tail(head))

    print("########## NA ########")
    print(dataframe.isnull().sum())

    print("########## Quantiles ########")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# =====================================================================
# 3. EKSİK DEĞER ANALİZİ
# =====================================================================

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Eksik Değer Sayısı': missing,
    'Eksik Yüzdesi (%)': missing_pct})

print(f"\n{missing_df[missing_df['Eksik Değer Sayısı'] > 0]}")

# NOT: CustomerID'si olmayan satırlar müşteri bazlı analizde kullanılamaz
# Bu satırları ileride sileceğiz (dropna)

print(f"\n CustomerID eksik olan {df['CustomerID'].isnull().sum():,} satır "
      f"müşteri analizi dışında bırakılacaktır.")

# =====================================================================
# 4. VERİ KALİTESİ KONTROLLERI
# =====================================================================

# 4.1 Negatif/Sıfır Miktar ve Fiyat
print(f"\n Negatif Quantity (iade/iptal): {(df['Quantity'] < 0).sum():,} adet")
print(f" Sıfır veya Negatif UnitPrice: {(df['UnitPrice'] <= 0).sum():,} adet")

# 4.2 İptal edilen faturalar (C ile başlayanlar)
cancelled = df[df['InvoiceNo'].astype(str).str.contains('C', na=False)]
print(f" İptal Edilen Fatura Sayısı: {cancelled['InvoiceNo'].nunique():,}")
print(f" İptal Edilen Satır Sayısı : {len(cancelled):,}")

# 4.3 Ülke dağılımı
print(f"\n Ülke Sayısı: {df['Country'].nunique()}")
print(f"\nEn Çok Sipariş Gelen İlk 10 Ülke:")
print(df['Country'].value_counts().head(10))

# 4.4 Tarih aralığı
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print(f"\n Veri Tarih Aralığı:")
print(f"   Başlangıç : {df['InvoiceDate'].min()}")
print(f"   Bitiş     : {df['InvoiceDate'].max()}")
print(f"   Süre      : {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} gün")

# =====================================================================
# 5. GÖRSEL ANALİZ
# =====================================================================

# Çıktı klasörünü oluştur
BASE_DIR = "crm_project"
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "data.csv")
output_dir = os.path.join(BASE_DIR, "reports")
os.makedirs(output_dir, exist_ok=True)


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('E-Ticaret Veri Seti - Keşifsel Veri Analizi', fontsize=16, fontweight='bold')

# 5.1 Aylık Sipariş Sayısı
df_clean_temp = df.dropna(subset=['CustomerID'])
df_clean_temp = df_clean_temp[~df_clean_temp['InvoiceNo'].astype(str).str.contains('C')]
df_clean_temp = df_clean_temp[df_clean_temp['Quantity'] > 0]
df_clean_temp['YearMonth'] = df_clean_temp['InvoiceDate'].dt.to_period('M')

monthly_orders = df_clean_temp.groupby('YearMonth')['InvoiceNo'].nunique()
axes[0, 0].bar(range(len(monthly_orders)), monthly_orders.values, color='steelblue', alpha=0.8)
axes[0, 0].set_xticks(range(len(monthly_orders)))
axes[0, 0].set_xticklabels([str(p) for p in monthly_orders.index], rotation=45, ha='right', fontsize=9)
axes[0, 0].set_title('Aylık Sipariş Sayısı')
axes[0, 0].set_ylabel('Sipariş Sayısı')

# 5.2 Ülke Bazında Sipariş Dağılımı (İlk 10)
top_countries = df_clean_temp['Country'].value_counts().head(10)
axes[0, 1].barh(top_countries.index[::-1], top_countries.values[::-1], color='coral', alpha=0.8)
axes[0, 1].set_title('Ülke Bazında Sipariş Sayısı (İlk 10)')
axes[0, 1].set_xlabel('Sipariş Sayısı')

# 5.3 Günlük Toplam Ciro
df_clean_temp['TotalPrice'] = df_clean_temp['Quantity'] * df_clean_temp['UnitPrice']
daily_revenue = df_clean_temp.groupby(df_clean_temp['InvoiceDate'].dt.date)['TotalPrice'].sum()
axes[1, 0].plot(daily_revenue.index, daily_revenue.values, color='green', alpha=0.7, linewidth=1)
axes[1, 0].set_title('Günlük Toplam Ciro (GBP)')
axes[1, 0].set_ylabel('Ciro (GBP)')
axes[1, 0].set_xlabel('Tarih')
axes[1, 0].tick_params(axis='x', rotation=45)

# 5.4 Müşteri Başına Sipariş Dağılımı
customer_orders = df_clean_temp.groupby('CustomerID')['InvoiceNo'].nunique()
axes[1, 1].hist(customer_orders[customer_orders <= 20], bins=20, color='purple', alpha=0.7)
axes[1, 1].set_title('Müşteri Başına Sipariş Dağılımı (≤20 sipariş)')
axes[1, 1].set_xlabel('Sipariş Sayısı')
axes[1, 1].set_ylabel('Müşteri Sayısı')

from IPython.display import display, Image
plt.tight_layout()
plot_path = os.path.join(output_dir, "01_eda_analysis.png")
plt.savefig(plot_path, dpi=72, bbox_inches='tight')
plt.close()
print(f"\nGörsel kaydedildi: {plot_path}")
display(Image(plot_path))


# =====================================================================
# 6. ÖZET
# =====================================================================

print(f"""
VERİ SETİ ÖZETI
------------------
Toplam Satır          : {df.shape[0]:,}
Toplam Fatura         : {df['InvoiceNo'].nunique():,}
Toplam Müşteri        : {df['CustomerID'].nunique():,}
Toplam Ürün           : {df['StockCode'].nunique():,}
Toplam Ülke           : {df['Country'].nunique()}
Tarih Aralığı         : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}

TEMİZLEME GEREKSİNİMLERİ
-----------------------------
CustomerID Eksik      : {df['CustomerID'].isnull().sum():,} satır silinecek
İptal Faturalar (C)   : {cancelled['InvoiceNo'].nunique():,} fatura silinecek
Negatif Miktar        : {(df['Quantity'] < 0).sum():,} satır silinecek
Geçersiz Fiyat (≤0)   : {(df['UnitPrice'] <= 0).sum():,} satır silinecek
""")


