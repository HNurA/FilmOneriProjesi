import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Veriyi Yükle
df = pd.read_csv("temiz_film_verisi.csv")

# 2. Verileri Birleştir (Etiket Oluşturma)
# Filmin türünü, özetini, anahtar kelimelerini ve oyuncularını tek bir metin haline getiriyoruz.
# Böylece makine hepsini tek bir "çorba" gibi okuyup benzerlik kurabilecek.
print("Film özellikleri birleştiriliyor...")

# CSV'den okurken bu sütunlar metin olarak gelir (örn: "['Action', 'Adventure']").
# Bunları temizleyip birleştiriyoruz.
df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast']

# Gereksiz karakterleri (köşeli parantez, tırnak vs.) temizleyelim ki makine kelimeleri net görsün
df['tags'] = df['tags'].str.replace(r'[^\w\s]', '', regex=True)

# 3. Vektörleştirme (Metni Sayıya Çevirme)
# Makine kelimeden anlamaz, sayıdan anlar. Her kelimeyi bir sayıya dönüştürüyoruz.
print("Benzerlik matrisi hesaplanıyor (Bu işlem 10-15 saniye sürebilir)...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags'].astype(str)).toarray()

# 4. Benzerlik Hesaplama (Cosine Similarity)
# Her filmin diğer 4800 filmle olan benzerlik puanını hesaplıyoruz.
similarity = cosine_similarity(vectors)

# --- ÖNERİ FONKSİYONU ---
def film_oner(film_adi):
    try:
        # 1. Filmin indexini bul
        film_index = df[df['original_title'] == film_adi].index[0]
        
        # 2. Bu filme en çok benzeyenlerin listesini al (kendisi hariç)
        distances = similarity[film_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        print(f"\n'{film_adi}' filmini sevenler bunları da sevebilir:")
        print("-" * 50)
        
        for i in movies_list:
            onerilen_film = df.iloc[i[0]].original_title
            puan = df.iloc[i[0]].vote_average
            print(f"- {onerilen_film} (IMDB: {puan})")
            
    except IndexError:
        print(f"\nHata: '{film_adi}' veri setinde bulunamadı. Lütfen tam adını doğru yazdığından emin ol.")

# --- TEST ETME ---
# Buraya istediğin filmi yazıp test edebilirsin
film_oner('The Matrix')
film_oner('Avatar')
film_oner('Interstellar')