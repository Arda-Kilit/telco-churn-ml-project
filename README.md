# Telekom Müşteri Kaybı (Telco Customer Churn) Tahmin Sistemi

BTK Uçtan Uca Makine Öğrenmesi Atölyesi kapsamında geliştirilmiş, veri kalitesi denetiminden canlı web arayüzüne kadar tüm aşamaları kapsayan sınıflandırma (Classification) ve karar destek projesidir.

## 1. Problem Tanımı ve İş Değeri
Telekomünikasyon sektöründe mevcut bir müşteriyi elde tutmak, yeni müşteri kazanmaktan belirgin biçimde daha düşük maliyetlidir. Bu proje, abonelik özellikleri ve kullanım geçmişine bakarak önümüzdeki dönemde hizmeti iptal etme (`Churn = Yes`) riski taşıyan müşterileri erken tespit etmeyi ve tutundurma (retention) kampanyalarını yüksek riskli segmente odaklamayı amaçlar.

## 2. Veri Seti ve Veri Kalitesi Özeti
- **Kaynak:** [Telco Customer Churn (Kaggle / IBM Sample Data Sets)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Boyut:** 7.043 müşteri × 21 sütun
- **Hedef Değişken (`Churn`):** %73,46 Kaldı (`No`, n=5.174) / %26,54 Ayrıldı (`Yes`, n=1.869) — orta düzey sınıf dengesizliği (~2,77:1).
- **Veri Kalitesi Bulguları:**
  - `TotalCharges` sütunu 11 adet boşluk (`" "`) karakteri nedeniyle metin (`object`) tipindedir; bu 11 kaydın tamamı `tenure == 0` olan yeni abonelerdir ve Pipeline içinde `0.0` olarak temizlenir.
  - Tam satır veya `customerID` tekrarı yoktur; `customerID` hariç özellik düzeyinde 22 tekrar eden kayıt mevcuttur.
  - Sayısal sütunlarda IQR tabanlı aşırı uç değer (outlier) tespit edilmemiştir.

## 3. Metodoloji (Aşama 1–15)
- **Sızıntısız Pipeline (`src/features.py`):** Tüm temizleme, alan tabanlı özellik mühendisliği (`ActiveServices`, `AvgMonthlySpend`, `ChargeRatio`, `SecuritySupportCount`, `StreamingCount`, `IsAutoPay`, `IsMonthToMonth`, `TenureGroup`), ölçekleme (`StandardScaler`) ve kodlama (`OneHotEncoder`) işlemleri `scikit-learn` `Pipeline` + `ColumnTransformer` içinde uygulanır.
- **Deney Tasarımı:** Stratified %70 Eğitim / %15 Doğrulama (Validation) / %15 Kilitli Test bölmesi (`random_state=42`) ve 5-Katlı Stratified Çapraz Doğrulama (5-Fold CV).
- **Baseline + 6+ Güçlü Rakip Karşılaştırması:**
  - Baseline: `DummyClassifier (most_frequent & stratified)`
  - Güçlü Rakipler (7 Algoritma): **Logistic Regression**, **Random Forest**, **Extra Trees**, **XGBoost**, **LightGBM**, **CatBoost**, **MLP Neural Network**
- **Adil Hiperparametre ve Eşik Optimizasyonu:** CV PR-AUC'a göre en iyi 3 aday model eşit deneme bütçesiyle (`RandomizedSearchCV`) optimize edilir; karar eşiği yalnızca doğrulama setinde F1-skoru maksimize edecek şekilde belirlenir. Test seti yalnızca final değerlendirmede bir kez kullanılır.

## 4. Kurulum ve Çalıştırma

### Seçenek A — Lokal Bilgisayar
```bash
git clone <repo-url>
cd Telco_Churn_BTK_Proje
pip install -r requirements.txt

# 1. Model eğitimi, 7 algoritma karşılaştırması, tuning ve grafik üretimi
python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 2. Streamlit arayüzünü başlatma
streamlit run app.py
```

### Seçenek B — Google Colab (Kurulum Gerektirmez)
1. `Telco_Churn_BTK_Proje.zip` içindeki `Telco_Churn_Colab.ipynb` dosyasını [Google Colab](https://colab.research.google.com/) üzerinde açın.
2. Tüm hücreleri sırayla çalıştırın (`Runtime -> Run all`).
3. Son hücrede otomatik indirilen `btk_deney_sonuclari.zip` dosyasındaki `models/`, `results/` ve `figures/` klasörlerini projenize ekleyin.

## 5. Streamlit Community Cloud ile Canlıya Alma (Deployment)
1. Proje klasörünü (eğitilmiş `models/best_model.joblib`, `models/model_meta.json`, `results/metrics.json` ve `figures/` ile birlikte) bir GitHub reposuna yükleyin.
2. [Streamlit Community Cloud](https://share.streamlit.io/) üzerinden **New app** seçeneğine tıklayın.
3. Repo adını, branch'i (`main`) ve ana dosya olarak `app.py` dosyasını seçip **Deploy** butonuna basın.

## 6. Proje Dizin Yapısı
```text
Telco_Churn_BTK_Proje/
├── app.py                     # Streamlit kullanıcı arayüzü
├── train.py                   # Uçtan uca eğitim, tuning ve değerlendirme
├── Telco_Churn_Colab.ipynb    # Google Colab tek tıkla çalıştırma defteri
├── requirements.txt           # Python bağımlılıkları
├── README.md                  # Proje dokümantasyonu
├── src/
│   └── features.py            # Sızıntısız Feature Engineering Transformer
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── figures/                   # EDA ve değerlendirme grafikleri (PNG)
├── models/                    # Eğitilmiş final model (best_model.joblib)
└── results/                   # Gerçek deney metrikleri (metrics.json)
```

## 7. Akademik Dürüstlük ve Şeffaflık Notu
Tüm veri analizi, model metrikleri ve grafikler doğrudan gerçek deney çıktılarından üretilir. Kod mimarisi ve dokümantasyon sürecinde yapay zekâ asistanından destek alınmış, tüm kod ve deney çıktıları proje sahibi tarafından doğrulanmıştır.



## Gerekli PDF ve PowerPoint Sunumu
-----------------------------------
## PowerPoint Sunumu > 
https://docs.google.com/presentation/d/19lERB4_sPYBGRhFZI21RNkTnhLv5m5OK/edit?usp=drive_link&ouid=114680489079357240033&rtpof=true&sd=true

## PDF > 
https://drive.google.com/file/d/1MxTAoGgymvCmZcKLTGDbjXO1fn4zeD6t/view?usp=drive_link


