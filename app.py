import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Film Öneri Sistemi", page_icon="🎬", layout="centered")

# --- BAŞLIK VE AÇIKLAMA ---
st.title("🎬 Yapay Zeka Film Önericisi")
st.markdown("Favori filmini seç, yapay zeka sana **içerik, tür ve oyuncu** benzerliğine göre en uygun filmleri önersin.")

# --- 1. VERİYİ YÜKLE VE ÖNBELLEĞE AL (Hız İçin) ---
@st.cache_data
def veriyi_hazirla():
    # Temizlenmiş veriyi yüklüyoruz
    df = pd.read_csv("temiz_film_verisi.csv")
    
    # Benzerlik hesaplamak için metinleri birleştiriyoruz (Tags)
    # Eğer önceki adımda 'tags' sütunu oluşturup kaydetmediysek burada anlık oluşturuyoruz
    df['tags'] = df['overview'] + " " + df['genres'] + " " + df['keywords'] + " " + df['cast']
    df['tags'] = df['tags'].fillna('') # Boş veri varsa doldur
    return df

try:
    df = veriyi_hazirla()
except FileNotFoundError:
    st.error("HATA: 'temiz_film_verisi.csv' dosyası bulunamadı. Lütfen önce veri temizleme kodunu çalıştır.")
    st.stop()

# --- 2. MODELİ KUR (Benzerlik Matrisi) ---
@st.cache_resource
def modeli_calistir(df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags'].astype(str)).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

with st.spinner('Yapay Zeka modelleri yükleniyor... Lütfen bekleyin...'):
    similarity = modeli_calistir(df)

# --- 3. KULLANICI ARAYÜZÜ (Seçim Kutusu) ---
st.divider() # Araya çizgi çek

# Kullanıcıya film listesini sun (Alfabetik sırala)
film_listesi = sorted(df['original_title'].unique())

# AÇILIR KUTU (Selectbox)
secilen_film = st.selectbox(
    "Hangi filmi sevdin? (Listeden seç veya yaz)",
    film_listesi
)

# --- 4. ÖNERİ BUTONU VE SONUÇLAR ---
if st.button('Benzer Filmleri Göster 🚀'):
    
    try:
        # Seçilen filmin indexini bul
        film_index = df[df['original_title'] == secilen_film].index[0]
        
        # Benzerlik puanlarını al
        distances = similarity[film_index]
        
        # En çok benzeyen 5 filmi sırala (Kendisi hariç [1:6])
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        st.success(f"'{secilen_film}' filmini sevdiysen bunları kesin izlemelisin:")
        
        # Sonuçları listele
        for i in movies_list:
            film_adi = df.iloc[i[0]].original_title
            puan = df.iloc[i[0]].vote_average
            ozet = df.iloc[i[0]].overview
            
            # Kart şeklinde gösterim
            with st.container():
                st.subheader(f"🍿 {film_adi}")
                st.caption(f"IMDB Puanı: ⭐ {puan}/10")
                st.write(f"_{ozet[:150]}..._") # Özetin ilk 150 karakteri
                st.divider()
                
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")

# --- ALT BİLGİ ---
st.markdown("---")
st.caption("Veri Bilimine Giriş Dersi Final Projesi | 2026")