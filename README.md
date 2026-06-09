# İki Aşamalı CRF ile Türkçe Morfolojik Analiz ve Çözümleme

Bu proje, Türkçe metinler üzerinde morfolojik analiz ve kelime türü (Part-of-Speech / POS) etiketlemesi yapmak amacıyla geliştirilmiştir. Yüksek başarım oranları elde etmek ve eğitim süresini optimize etmek için **İki Aşamalı Koşullu Rastgele Alanlar (Two-Stage Conditional Random Fields - CRF)** mimarisi kullanılmıştır.

Proje, IMST-UD (Universal Dependencies) Türkçe veri seti (CoNLL-U formatında) üzerinde eğitilmiş ve test edilmiştir.

## 🚀 Özellikler ve Model Mimarisi

Sistem iki temel CRF modelinin ardışık olarak çalıştırılmasına dayanır:

1. **Aşama 1 (UPOS Modeli):** Metinlerdeki kelimelerin Evrensel Kelime Türü (Universal Part-of-Speech - UPOS) etiketlerini tahmin eder.
2. **Aşama 2 (FEATS Modeli):** Birinci aşamadan elde edilen UPOS tahminlerini bir "öznitelik (feature)" olarak (Teacher Forcing yöntemi ile) alır ve kelimelerin detaylı morfolojik özelliklerini (FEATS) tahmin eder.

Bu iki aşamalı yaklaşım, tek ve karmaşık bir model (Joint Model) eğitmeye kıyasla **eğitim süresini saatlerden saniyelere (< 1 dakika)** indirirken, tahmin başarısını da optimize etmektedir. 

Öznitelik çıkarımı (Feature Extraction) sürecinde hem kelime köklerini ve eklerini bulmak hem de bağlamsal özellikleri oluşturmak için **Zemberek (JPype via Java)** kütüphanesinden yararlanılmaktadır.

## 📊 Başarı Metrikleri

Geliştirilen İki Aşamalı CRF mimarisinin, test veri seti (`tr_imst-ud-test.conllu`) üzerindeki doğruluk (Accuracy) metrikleri şöyledir:

- **Sadece UPOS Doğruluğu:** ~%98.88
- **Joint Doğruluk (UPOS + FEATS Tam Eşleşme):** ~%73.13

*Sınıf bazında detaylı metrikleri ve yapılan hataların analizini `confusion_matrix_heatmap.png` görselinde ve çalışma raporunda bulabilirsiniz.*

## ⚙️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

### 1. Gereksinimleri Yükleyin
Proje, Python 3.8+ ortamında çalışmaktadır. Gerekli kütüphaneleri yüklemek için:
```bash
pip install -r requirements.txt
```

*(Kullanılan temel kütüphaneler: `sklearn-crfsuite`, `conllu`, `JPype1`, `zeyrek`, `scikit-learn`, `matplotlib`, `seaborn`)*

### 2. Zemberek ve Veri Seti Dosyalarını Ekleyin
Çalışma dizininde aşağıdaki dosyaların bulunduğundan emin olun:
- **Zemberek JAR:** `zemberek-full.jar` (Eğer ortamda Java kurulu değilse sistem otomatik olarak saf Python kütüphanesi olan `zeyrek`'e geçiş yapar ancak Zemberek kullanılması tavsiye edilir.)
- **IMST-UD Veri Setleri:** `tr_imst-ud-train.conllu`, `tr_imst-ud-dev.conllu` ve `tr_imst-ud-test.conllu` veri setleri.

## 🖥️ Kullanım

Modelin eğitilmesi ve test edilmesi için aşağıdaki komutu çalıştırmanız yeterlidir:

```bash
python morphological_disambiguation.py
```

Bu komut sırasıyla şu işlemleri gerçekleştirir:
1. Veri setlerini (Train, Dev, Test) okur ve ayrıştırır.
2. Kelimeler üzerinden (Zemberek kullanarak) öznitelik (feature) çıkarımı yapar.
3. Aşama 1 (UPOS) CRF modelini eğitir.
4. Aşama 2 (FEATS) CRF modelini (Aşama 1'in UPOS çıktılarını kullanarak) eğitir.
5. Test seti üzerinde tahminlerde bulunur ve Başarı Metrikleri (Accuracy, Classification Report) ile Karmaşıklık Matrisi (Confusion Matrix) hesaplar.

## 📂 Çıktılar

Uygulama çalışmasını tamamladıktan sonra projenin ana dizininde aşağıdaki çıktılar oluşur:

- `predictions_output.conllu`: Test veri seti üzerindeki modelin morfolojik etiketleme tahminlerinin bulunduğu dosya.
- `confusion_matrix_heatmap.png`: UPOS sınıfları (Türkçe karşılıklarıyla birlikte) için oluşturulmuş yüksek çözünürlüklü karmaşıklık matrisi görseli.

## 📚 Dokümantasyon ve Raporlama

Projenin geliştirilme süreci, alınan tasarım kararları ve detaylı hata analizi hakkında daha fazla bilgi edinmek için projedeki analiz raporlarını (`Morfolojik_Cozumleme_Raporu.docx`) ve walkthrough belgelerini inceleyebilirsiniz.
