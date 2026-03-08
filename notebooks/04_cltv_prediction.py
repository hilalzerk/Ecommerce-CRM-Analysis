###############################################################
# CRM ANALİZİ - ADIM 4: CLTV TAHMİNİ (BG-NBD & Gamma-Gamma)
###############################################################

# =====================================================================
# BU ADIMDA NE YAPACAĞIZ?
# ---------------------------------------------------------------------
# İSTATİSTİKSEL MODELLER KULLANARAK GELECEKTEKI MÜŞTERİ DEĞERİNİ TAHMİN EDECEĞİZ
#
# 1. BG/NBD MODELİ (Beta-Geometric / Negative Binomial Distribution)
#    ─────────────────────────────────────────────────────────────────
#    → Gelecekte kaç satın alma yapılacağını tahmin eder
#    → İki şeyi aynı anda modeller:
#       - Müşteri hâlâ aktif mi? (Beta-Geometric kısmı)
#       - Aktifse ne sıklıkta alışveriş yapar? (NBD kısmı)
#
# 2. GAMMA-GAMMA MODELİ
#    ─────────────────────────────────────────────────────────────────
#    → Her satın almada ortalama ne kadar harcama yapılacağını tahmin eder
#    → Gamma-Gamma varsayımı: monetary değerleri ile frequency bağımsızdır
#
# 3. CLTV TAHMİNİ = BG-NBD × Gamma-Gamma
#    ─────────────────────────────────────────────────────────────────
#    → "Bu müşteri önümüzdeki 3 ayda ne kadar gelir getirir?"
#
# GEREKLİ KURULUM:
#    pip install lifetimes
#
# INPUT METRİKLERİ (lifetimes kütüphanesi için):
#    recency  : Son ve ilk alışveriş arasındaki süre (hafta)
#    T        : İlk alışverişten analiz tarihine kadar geçen süre (hafta)
#    frequency: Tekrar eden alışveriş sayısı (>1 olanlar)
#    monetary : Alışveriş başına ortalama kazanç
# =====================================================================

import os
import datetime as dt
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
matplotlib.use('Agg')
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')


plt.rcParams['figure.figsize'] = (14, 7)
sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Dosya yolları
BASE_DIR = "crm_project"
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =====================================================================
# VERİNİN OKUNMASI
# =====================================================================
df_ = pd.read_csv("crm_project/data/raw/data.csv", encoding="latin-1")
dataframe = df_.copy()

dataframe.describe().T
dataframe.head()
dataframe.isnull().sum()

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


# =====================================================================
# LIFETIME VERİ YAPISININ HAZIRLANMASI
# =====================================================================

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

def create_cltv_p(dataframe, month=3):
    # 1. Veri Ön İşleme
    dataframe['InvoiceDate'] = pd.to_datetime(dataframe['InvoiceDate'])
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["InvoiceNo"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["UnitPrice"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "UnitPrice")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["UnitPrice"]
    today_date = dt.datetime(2011, 12, 11)


    cltv_df = dataframe.groupby('CustomerID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'InvoiceNo': lambda InvoiceNo: InvoiceNo.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    cltv_df


    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_6_month"] = bgf.predict(24,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])
    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final,bgf,ggf,cltv_df

cltv_final, bgf, ggf, cltv_df = create_cltv_p(dataframe, month=3)

cltv_final, bgf, ggf, cltv_df = create_cltv_p(dataframe, month=6)


# --- Model Doğrulama ---

fig, ax = plt.subplots(figsize=(12, 6))
plot_period_transactions(bgf, ax=ax)
ax.set_title('BG-NBD Model Doğrulama: Gerçek vs Tahmin Edilen İşlem Sayıları')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "04_bgnbd_validation.png"), dpi=72, bbox_inches='tight')



# Gamma-Gamma varsayımı kontrolü:
# frequency ile monetary arasında korelasyon düşük olmalı (<0.3 iyi)

corr = cltv_df[['frequency', 'monetary']].corr().iloc[0, 1]
print(f"\n Frequency-Monetary Korelasyonu: {corr:.4f}")
if abs(corr) < 0.3:
    print("   Korelasyon düşük - Gamma-Gamma varsayımı sağlanıyor")
else:
    print("   Korelasyon yüksek - Model varsayımı zayıf, sonuçları dikkatli yorumlayın")




# =====================================================================
# GÖRSELLEŞTİRME
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CLTV Tahmini - BG-NBD & Gamma-Gamma Modeli', fontsize=16, fontweight='bold')

colors_seg = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

# 3 Aylık CLTV Dağılımı
cltv_vals = cltv_final[cltv_final['expected_purc_3_month'] < cltv_final['expected_purc_3_month'].quantile(0.95)]['expected_purc_3_month']
axes[0, 0].hist(cltv_vals, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
axes[0, 0].set_title('3 Aylık CLTV Dağılımı (alt 95. yüzdelik)')
axes[0, 0].set_xlabel('CLTV')
axes[0, 0].set_ylabel('Müşteri Sayısı')

# Segment Müşteri Sayısı
seg_counts = cltv_final['segment'].value_counts().sort_index()
bars = axes[0, 1].bar(seg_counts.index, seg_counts.values, color=colors_seg, alpha=0.9)
axes[0, 1].set_title('Segment Bazında Müşteri Sayısı (3 Aylık)')
axes[0, 1].set_xlabel('Segment')
axes[0, 1].set_ylabel('Müşteri Sayısı')
for bar, val in zip(bars, seg_counts.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:,}', ha='center', fontsize=11, fontweight='bold')

#  1 ay vs 3 ay CLTV Karşılaştırması (Scatter)
sample = cltv_final.sample(min(500, len(cltv_final)), random_state=42)
scatter_colors = sample['segment'].map({'D': '#e74c3c', 'C': '#f39c12', 'B': '#3498db', 'A': '#2ecc71'})
axes[1, 0].scatter(sample['expected_purc_3_month'], sample['expected_purc_6_month'],
                   c=scatter_colors, alpha=0.6, s=30)
axes[1, 0].set_title('3 Aylık vs 6 Aylık CLTV (Örneklem)')
axes[1, 0].set_xlabel('3 Aylık CLTV')
axes[1, 0].set_ylabel('6 Aylık CLTV')
# Renk lejandı
for seg, color in zip(['D', 'C', 'B', 'A'], colors_seg):
    axes[1, 0].scatter([], [], c=color, label=f'Segment {seg}', s=50)
axes[1, 0].legend()

# Segment Bazında Toplam CLTV
seg_total = cltv_final.groupby('segment')['expected_purc_3_month'].sum()
axes[1, 1].pie(seg_total.values,
               labels=[f'Segment {s}\n{v:,.0f}' for s, v in zip(seg_total.index, seg_total.values)],
               autopct='%1.1f%%', colors=colors_seg, startangle=90)
axes[1, 1].set_title('Segment Bazında Toplam 3 Aylık CLTV')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "04_cltv_prediction.png"), dpi=120, bbox_inches='tight')


# =====================================================================
# ZAMAN BAZLI TEST
# Train-Test Split / Hold-out
# =====================================================================

last_date = dataframe['InvoiceDate'].max()
cut_off_date = last_date - pd.DateOffset(months=3)

# Eğitim ve Test setleri
train_df = dataframe[dataframe['InvoiceDate'] <= cut_off_date] # ilk 9 ay
test_df = dataframe[(dataframe['InvoiceDate'] > cut_off_date)] # son 3 ay

# Model ilk 9 aylık veri ile eğitiliyor.
# son 3 ay test verisi olarak kullanıldı.

cltv_train, bgf, ggf, cltv_df = create_cltv_p(train_df, month=3)

# 1. Test setindeki gerçek işlem sayılarını al (Müşteri bazlı)
actual_counts = test_df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
actual_counts.columns = ['CustomerID', 'actual_transaction_count']

# 2. Tahminler ile Gerçekleri Birleştir
validation_df = cltv_train[['CustomerID', 'expected_purc_3_month']].merge(actual_counts, on='CustomerID', how='left')
validation_df.fillna(0, inplace=True) # Test döneminde hiç alışveriş yapmayanlar 0

# 3. Skorlama
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(validation_df['actual_transaction_count'], validation_df['expected_purc_3_month']))
mae = mean_absolute_error(validation_df['actual_transaction_count'], validation_df['expected_purc_3_month'])

print(f"Model 3 Aylık Tahmin Başarısı:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}")

#Model 3 Aylık Tahmin Başarısı:
# RMSE: 4.32
# MAE: 1.99

# MAE  < ortalama işlem sayısının %30
# RMSE < ortalama işlem sayısının %50

# İdeal MAE             : < 0.69
# İdeal RMSE            : < 1.15

print(actual_counts['actual_transaction_count'].describe())



# Aykırı müşterileri çıkarıp tekrar hesapla
filtreli = validation_df[validation_df['actual_transaction_count'] <= 10]

rmse_filtreli = np.sqrt(mean_squared_error(
    filtreli['actual_transaction_count'],
    filtreli['expected_purc_3_month']
))
mae_filtreli = mean_absolute_error(
    filtreli['actual_transaction_count'],
    filtreli['expected_purc_3_month']
)

print(f"Aykırı değerler hariç:")
print(f"RMSE : {rmse_filtreli:.2f}")
print(f"MAE  : {mae_filtreli:.2f}")
print(f"Etkilenen müşteri sayısı: {len(validation_df) - len(filtreli)}")

# Aykırı değerler hariç:
# RMSE : 2.52
# MAE  : 1.66
# Etkilenen müşteri sayısı: 34


actual_counts.sort_values('actual_transaction_count', ascending=False).head(20)

actual_counts['actual_transaction_count'].describe().T

actual_counts['actual_transaction_count'].value_counts().sort_index().head(10)

#1 işlem yapan : 1503 müşteri → toplam 2893'ün %52'si
#2 işlem yapan :  633 müşteri → %22
#3+             :  757 müşteri → %26

# Validation'ı sadece 2+ işlem yapan müşterilerle yap
filtreli = validation_df[validation_df['actual_transaction_count'] >= 2]

rmse_filtreli = np.sqrt(mean_squared_error(
    filtreli['actual_transaction_count'],
    filtreli['expected_purc_3_month']
))
mae_filtreli = mean_absolute_error(
    filtreli['actual_transaction_count'],
    filtreli['expected_purc_3_month']
)

print(f"Tüm müşteriler    → RMSE: 4.32  MAE: 1.99  ({len(validation_df):,} müşteri)")
print(f"2+ işlem yapanlar → RMSE: {rmse_filtreli:.2f}  MAE: {mae_filtreli:.2f}  ({len(filtreli):,} müşteri)")


# Tüm müşteriler    → RMSE: 4.32  MAE: 1.99  (2,003 müşteri)
# 2+ işlem yapanlar → RMSE: 6.48  MAE: 3.93  (882 müşteri)


# 2-10 arası işlem yapanlar
filtreli2 = validation_df[validation_df['actual_transaction_count'].between(2, 10)]

rmse_f2 = np.sqrt(mean_squared_error(
    filtreli2['actual_transaction_count'],
    filtreli2['expected_purc_3_month']
))
mae_f2 = mean_absolute_error(
    filtreli2['actual_transaction_count'],
    filtreli2['expected_purc_3_month']
)

print(f"Tüm müşteriler     → RMSE: 4.32  MAE: 1.99  ({len(validation_df):,} müşteri)")
print(f"2+ işlem yapanlar  → RMSE: 6.48  MAE: 3.93  ({len(filtreli):,} müşteri)")
print(f"2-10 işlem arasında→ RMSE: {rmse_f2:.2f}  MAE: {mae_f2:.2f}  ({len(filtreli2):,} müşteri)")

# Tüm müşteriler     → RMSE: 4.32  MAE: 1.99  (2,003 müşteri)
# 2+ işlem yapanlar  → RMSE: 6.48  MAE: 3.93  (882 müşteri)
# 2-10 işlem arasında→ RMSE: 3.79  MAE: 3.26  (848 müşteri)




# 11+ işlem yapan kaç müşteri var?
print(validation_df[validation_df['actual_transaction_count'] > 10]['actual_transaction_count'].describe())

# Bu müşterilerin tahmin hatası
aykiri = validation_df[validation_df['actual_transaction_count'] > 10]
rmse_aykiri = np.sqrt(mean_squared_error(
    aykiri['actual_transaction_count'],
    aykiri['expected_purc_3_month']
))
print(f"11+ işlem yapanlar → RMSE: {rmse_aykiri:.2f}  ({len(aykiri)} müşteri)")

# 34 müşteri    → RMSE: 27.07   ← tüm skoru mahveden grup
# 1969 müşteri  → RMSE: ~2-3    ← model burada iyi çalışıyor

# Ne Anlama Geliyor:
# 34 müşteri toplamın sadece %1.7'si ama RMSE'yi 4.32'ye çekiyor. Bu müşteriler büyük ihtimalle:

# Toptancılar
# Kurumsal müşteriler
# Düzensiz ama çok sık alışveriş yapanlar

# BG-NBD bu tür aşırı aktif müşterileri modelleyemiyor çünkü davranışları normal dağılıma uymuyor.


# Validation'ı 3 gruba böl
normal    = validation_df[validation_df['actual_transaction_count'] <= 10]
aykiri    = validation_df[validation_df['actual_transaction_count'] > 10]

rmse_normal = np.sqrt(mean_squared_error(
    normal['actual_transaction_count'],
    normal['expected_purc_3_month']
))
mae_normal = mean_absolute_error(
    normal['actual_transaction_count'],
    normal['expected_purc_3_month']
)

print(f"Normal müşteriler (≤10 işlem) → RMSE: {rmse_normal:.2f}  MAE: {mae_normal:.2f}  ({len(normal):,} müşteri)")
print(f"Aykırı müşteriler (>10 işlem) → RMSE: 27.07              ({len(aykiri):,} müşteri)")
print(f"\nModel normal müşterilerin %{(len(normal)/len(validation_df))*100:.0f}'ini iyi tahmin ediyor")
print(f"Aykırı %{(len(aykiri)/len(validation_df))*100:.1f} için ayrı model gerekiyor")


# Normal müşteriler (≤10 işlem) → RMSE: 2.52  MAE: 1.66  (1,969 müşteri)
# Aykırı müşteriler (>10 işlem) → RMSE: 27.07              (34 müşteri)
# Model normal müşterilerin %98'ini iyi tahmin ediyor
# Aykırı %1.7 için ayrı model gerekiyor




# Model 2.003 müşterinin %98.3'ünü makul hatayla tahmin ediyor. Kalan %1.7'lik grup (toptancılar/kurumsal)
# BG-NBD modelinin varsayımlarına uymuyor. Bu segment için kural tabanlı veya ayrı bir model önerilir.


#############################################
# XGBoost ile CLTV Tahmini
# BG-NBD Modeli ile Karşılaştırma
############################################

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# XGBoost için cltv_df'deki metrikleri kullanacağız.
# Hedef değişken: actual_transaction_count (gerçek işlem sayısı)

# cltv_df ve actual_counts zaten tanımlı
# train_df ve test_df de tanımlı


# Feature'ları hazırla
features = cltv_df[['recency', 'T', 'frequency', 'monetary']].copy()
features['recency_ratio'] = features['recency'] / features['T']
features['monetary_frequency'] = features['monetary'] * features['frequency']
features = features.reset_index()

features.head()
features.shape


# Hedef Değişken
model_df = features.merge(actual_counts, on='CustomerID', how='inner')
model_df = model_df.fillna(0)

model_df.shape
model_df.describe().T

#Train-Test Split

X = model_df[['recency', 'T', 'frequency', 'monetary',
              'recency_ratio', 'monetary_frequency']]
y = model_df['actual_transaction_count']

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2,      # %20 test, %80 train
    random_state=42)     # tekrarlanabilirlik için sabit seed

print(f"Train seti: {X_train.shape[0]:,} müşteri")
print(f"Test seti : {X_test.shape[0]:,} müşteri")


xgb_model = XGBRegressor(
    n_estimators=100,
    # kaç tane ağaç kurulsun? Fazla olursa overfitting riski artar
    learning_rate=0.1,
    # her adımda ne kadar öğrensin? Düşükse daha yavaş ama daha stabil
    max_depth=4,
    # her ağaç en fazla kaç dal derinliğinde olsun?
    subsample=0.8,
    # her ağaç için verinin %80'ini kullan → overfitting önler
    colsample_bytree=0.8,
    # her ağaç için feature'ların %80'ini kullan
    random_state=42,
    verbosity=0
)

xgb_model.fit(X_train, y_train)
print("Model eğitildi ✅")

# Tahmin ve Skorlama

y_pred = xgb_model.predict(X_test)
y_pred = np.clip(y_pred, 0, None)

# clip(0, None) → negatif tahminleri 0'a çek
# işlem sayısı negatif olamaz

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
mae_xgb  = mean_absolute_error(y_test, y_pred)

print(f"RMSE : {rmse_xgb:.2f}")
print(f"MAE  : {mae_xgb:.2f}")


# Model       RMSE    MAE
# BG-NBD      4.32   1.99
# XGBoost     3.34   1.33

## 7. Cross Validation
# Tek bir train-test split şansa bağlı olabilir
# Cross validation daha güvenilir sonuç verir.

cv_scores = cross_val_score(
    xgb_model, X, y,
    cv=5,
    # 5-fold → veriyi 5'e böl, her seferinde 1'ini test yap
    scoring='neg_mean_absolute_error'
    # negatif MAE → sklearn convention, büyük = iyi
)

cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"5-Fold CV MAE : {cv_mae:.2f} ± {cv_std:.2f}")

# Test MAE      : 1.33
# CV MAE        : 1.40 ± 0.08

# ± 0.08 oldukça düşük, model her veri bölümünde tutarlı çalışıyor, şansa bağlı değil.
# Test MAE ile CV MAE arasındaki fark sadece 0.07 bu da overfitting olmadığını gösteriyor.

# BG-NBD istatistiksel yorumlanabilirlik sağlarken XGBoost tahmin doğruluğunda %33 daha iyi performans gösterdi. 5-Fold CV MAE 1.40 ± 0.08 ile modelin stabil çalıştığı doğrulandı.
# Production için XGBoost, iş yorumu için BG-NBD önerilir.


#################################################################



# Feature Importance
# Hangi değişken tahminde daha etkili?

importance_df = pd.DataFrame({
    'feature'   : X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Önem Sırası:")
print(importance_df)

# Görselleştir
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(importance_df['feature'], importance_df['importance'])
ax.set_title('XGBoost — Feature Importance', fontsize=13, fontweight='bold')
ax.set_xlabel('Önem Skoru')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "05_xgboost_feature_importance.png"), dpi=72, bbox_inches='tight')



# Tahmin vs Gerçek Grafik


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('XGBoost vs BG-NBD Model Karşılaştırması', fontsize=14, fontweight='bold')

# Scatter: Tahmin vs Gerçek
axes[0].scatter(y_test, y_pred, alpha=0.4, color='#4A90D9', s=20)
max_val = max(y_test.max(), y_pred.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5)
# kırmızı çizgi → mükemmel tahmin çizgisi. Nokta bu çizgiye yakınsa iyi.
axes[0].set_title('XGBoost: Tahmin vs Gerçek')
axes[0].set_xlabel('Gerçek İşlem Sayısı')
axes[0].set_ylabel('Tahmin Edilen')

# Bar: Model Karşılaştırması
modeller = ['BG-NBD', 'XGBoost']
rmse_values = [4.32, rmse_xgb]
mae_values  = [1.99, mae_xgb]

x = np.arange(len(modeller))
width = 0.35
axes[1].bar(x - width/2, rmse_values, width, label='RMSE', color='#e74c3c', alpha=0.8)
axes[1].bar(x + width/2, mae_values,  width, label='MAE',  color='#3498db', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(modeller)
axes[1].set_title('Model Karşılaştırması (düşük = iyi)')
axes[1].set_ylabel('Hata Skoru')
axes[1].legend()

for i, (r, m) in enumerate(zip(rmse_values, mae_values)):
    axes[1].text(i - width/2, r + 0.05, f'{r:.2f}', ha='center', fontsize=10)
    axes[1].text(i + width/2, m + 0.05, f'{m:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "05_model_comparison.png"), dpi=72, bbox_inches='tight')

