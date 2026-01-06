# 🎬 Movie Recommendation & Analysis System (Film Öneri ve Analiz Sistemi)

Bu proje, **İstanbul Medeniyet Üniversitesi - Veri Bilimine Giriş** dersi final projesi kapsamında geliştirilmiştir. Proje, TMDB veri setini kullanarak filmlerin başarı durumunu tahmin eden bir yapay zeka modeli ve kullanıcılara benzer filmleri sunan bir öneri sistemi içerir.

## 🚀 Özellikler

* **Veri Analizi (EDA):** Film dünyasına dair istatistiksel grafikler, kelime bulutları ve dağılım analizleri.
* **Başarı Tahmini (Classification):** Random Forest algoritması ile bir filmin "Yüksek Puanlı" olup olmayacağının tahmini (Confusion Matrix ve ROC Analizi ile).
* **Film Öneri Sistemi (Recommendation):** NLP ve Cosine Similarity kullanılarak seçilen filme en benzer içeriklerin önerilmesi.
* **Web Arayüzü:** Streamlit ile geliştirilmiş, kullanıcı dostu interaktif web uygulaması.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python 3.x
* **Arayüz:** Streamlit
* **Veri İşleme:** Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-learn (Random Forest, CountVectorizer)
* **Görselleştirme:** Matplotlib, Seaborn, WordCloud

## 📂 Dosya Yapısı

* `app.py`: Streamlit arayüzünü çalıştıran ana dosya.
* `01_veri_temizleme.py`: Ham veriyi temizleyen ve model için hazırlayan script.
* `03_model_ve_metrikler.py`: Sınıflandırma modelini eğiten ve başarı metriklerini raporlayan script.
* `archive.zip`: Ham veri setlerini (Movies & Credits) içeren sıkıştırılmış dosya.
* `requirements.txt`: Gerekli kütüphane listesi.

## ⚙️ Kurulum ve Çalıştırma

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Repoyu klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/REPO_ADIN.git](https://github.com/KULLANICI_ADIN/REPO_ADIN.git)
    cd REPO_ADIN
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Veriyi Hazırlayın:**
    *(Bu komut `archive.zip` dosyasını otomatik olarak açar ve veriyi temizler)*
    ```bash
    python 01_veri_temizleme.py
    ```

4.  **Uygulamayı Başlatın:**
    ```bash
    python -m streamlit run app.py
    ```
