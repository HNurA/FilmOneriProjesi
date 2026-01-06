import pandas as pd
import ast 
import zipfile
import os

# --- OTOMATİK ZIP ÇIKARMA ---
# Eğer csv dosyaları klasörde yoksa, archive.zip içinden çıkar
if not os.path.exists("tmdb_5000_movies.csv") or not os.path.exists("tmdb_5000_credits.csv"):
    print("archive.zip dosyası dışarı aktarılıyor...")
    with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Dosyalar başarıyla çıkarıldı!")

# 1. Veriyi Yükle
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
credits.columns = ['id', 'tittle', 'cast', 'crew']
df = movies.merge(credits, on='id')

# --- DÜZELTME BURADA ---
# Listeye 'budget' ve 'runtime' eklendi
df = df[['id', 'original_title', 'overview', 'genres', 'keywords', 'cast', 'vote_average', 'vote_count', 'budget', 'runtime']]

# --- TEMİZLİK FONKSİYONLARI ---
def donustur(text):
    L = []
    try:
        for i in ast.literal_eval(text):
            L.append(i['name'])
    except:
        pass
    return L

def oyunculari_al(text):
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
                counter+=1
            else:
                break
    except:
        pass
    return L

print("Veriler temizleniyor... Lütfen bekleyin.")

# Fonksiyonları uygula
df['genres'] = df['genres'].apply(donustur)
df['keywords'] = df['keywords'].apply(donustur)
df['cast'] = df['cast'].apply(oyunculari_al)

# Hedef sütunu oluştur (Classification için)
df['is_high_rated'] = df['vote_average'].apply(lambda x: 1 if x >= 6.5 else 0)

# Eksik verileri at (Süre veya bütçesi boş olanlar gitsin)
df.dropna(inplace=True)

# Yeni dosyayı kaydet
df.to_csv("temiz_film_verisi.csv", index=False)

print("\nİşlem Tamam! 'temiz_film_verisi.csv' dosyası güncellendi.")
print("Artık budget ve runtime sütunları dosyada mevcut.")
print(df.info())
