import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 1. Temiz Veriyi Yükle
df = pd.read_csv("temiz_film_verisi.csv")

# Görselleştirme Ayarları
sns.set(style="whitegrid")

# --- GRAFİK 1: HEDEF DEĞİŞKEN DAĞILIMI (Pasta Grafiği) ---
# Raporun "Veri Analizi" kısmına koymalık.
plt.figure(figsize=(8, 6))
labels = ['Düşük/Orta Puanlı (0)', 'Yüksek Puanlı (1)']
sizes = df['is_high_rated'].value_counts().values
colors = ['#ff9999','#66b3ff']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Veri Setindeki İyi ve Kötü Film Dağılımı')
plt.axis('equal') 
plt.savefig("grafik1_sinif_dagilimi.png") # Resmi kaydeder
print("Grafik 1 kaydedildi: grafik1_sinif_dagilimi.png")
plt.show()

# --- GRAFİK 2: PUANLAMA DAĞILIMI (Histogram) ---
# Verinin normal dağılıp dağılmadığını gösterir.
plt.figure(figsize=(10, 6))
sns.histplot(df['vote_average'], kde=True, color='purple', bins=30)
plt.title('Film Puanlarının Genel Dağılımı')
plt.xlabel('Puan (0-10)')
plt.ylabel('Film Sayısı')
plt.savefig("grafik2_puan_dagilimi.png")
print("Grafik 2 kaydedildi: grafik2_puan_dagilimi.png")
plt.show()

# --- GRAFİK 3: KELİME BULUTU (Word Cloud) ---
# Filmlerin özetlerinde en çok geçen kelimeler
text = " ".join(review for review in df.overview.astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Film Özetlerinde En Çok Geçen Kelimeler')
plt.savefig("grafik3_kelime_bulutu.png")
print("Grafik 3 kaydedildi: grafik3_kelime_bulutu.png")
plt.show()