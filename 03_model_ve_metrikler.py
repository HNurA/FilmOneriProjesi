import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 1. Temiz Veriyi Yükle
df = pd.read_csv("temiz_film_verisi.csv")

# 2. Model İçin Veriyi Hazırla
# Girdi (X): Bütçe, Süre, Oy Sayısı
X = df[['budget', 'runtime', 'vote_count']] 
# Hedef (y): İyi film mi (1) Kötü mü (0)?
y = df['is_high_rated']

# Eğitim ve Test Setlerine Ayırma (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modeli Kur ve Eğit (Random Forest Algoritması)
print("Model eğitiliyor...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Tahmin Yap
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # ROC eğrisi için olasılıklar

# --- RAPOR İÇİN METRİKLER VE GRAFİKLER ---

# A. Metrikleri Yazdır
print("\n--- SONUÇLAR ---")
print(classification_report(y_test, y_pred))

# B. Confusion Matrix (Karmaşıklık Matrisi) Grafiği
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Durum')
plt.title('Confusion Matrix (Karmaşıklık Matrisi)')
plt.savefig("grafik4_confusion_matrix.png")
print("Grafik 4 kaydedildi: grafik4_confusion_matrix.png")
plt.show()

# C. ROC Eğrisi Grafiği
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (Alan = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Rastgele tahmin çizgisi
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Alarm)')
plt.ylabel('True Positive Rate (Doğru Tespit)')
plt.title('ROC Curve (Alıcı İşletim Karakteristiği)')
plt.legend(loc="lower right")
plt.savefig("grafik5_roc_curve.png")
print("Grafik 5 kaydedildi: grafik5_roc_curve.png")
plt.show()

# Modelin başarısını yüzdesel olarak görelim
basari = model.score(X_test, y_test)
print(f"\nModel Doğruluk Oranı (Accuracy): %{basari*100:.2f}")