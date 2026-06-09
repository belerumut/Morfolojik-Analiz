import os
import sys
import warnings
import logging
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from conllu import parse_incr
from tqdm import tqdm

import scipy.stats
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ── Dosya Yolları ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "tr_imst-ud-train.conllu")
TEST_FILE  = os.path.join(BASE_DIR, "tr_imst-ud-test.conllu")
ZEMBEREK_JAR = os.path.join(BASE_DIR, "zemberek-full.jar")
OUTPUT_FILE  = os.path.join(BASE_DIR, "predictions_output.conllu")
HEATMAP_FILE = os.path.join(BASE_DIR, "confusion_matrix_heatmap.png")

# ── Eğitim/Test Boyutu Sınırı ──

MAX_TRAIN_SENTS = None
MAX_TEST_SENTS  = None

# =====================================================================
# BÖLÜM 1: ZEMBEREK / ZEYREK ENTEGRASYONU
# =====================================================================

class MorphAnalyzerBase:
    """Morfolojik analizör arayüzü."""
    def analyze(self, word: str) -> list[str]:
        raise NotImplementedError

    def get_stems(self, word: str) -> list[str]:
        """Kelime köklerini döndürür."""
        raise NotImplementedError


class ZemberekAnalyzer(MorphAnalyzerBase):
    """
    Java Zemberek kütüphanesi ile morfolojik analiz.
    """
    def __init__(self, jar_path: str):
        import jpype
        if not jpype.isJVMStarted():
            jpype.startJVM(
                jpype.getDefaultJVMPath(),
                '-ea',
                f'-Djava.class.path={jar_path}',
                convertStrings=True
            )
        TurkishMorphology = jpype.JClass('zemberek.morphology.TurkishMorphology')
        self._morphology = TurkishMorphology.createWithDefaults()
        logger.info("Zemberek (Java/JPype) morfoloji motoru başlatıldı.")

    # Kelimeyi analiz eder ve morfolojik analiz sonuçlarını döndürür.
    def analyze(self, word: str) -> list[str]:
        try:
            results = self._morphology.analyze(word)
            candidates = []
            for r in results:
                fmt = str(r.formatLong())
                candidates.append(fmt)
            return candidates if candidates else [f"UNK_{word}"]
        except Exception:
            return [f"UNK_{word}"]

    # Kelimenin köklerini döndürür.
    def get_stems(self, word: str) -> list[str]:
        try:
            results = self._morphology.analyze(word)
            stems = []
            for r in results:
                lemmas = list(r.getLemmas())
                if lemmas:
                    stems.append(str(lemmas[0]))
            return list(set(stems)) if stems else [word.lower()]
        except Exception:
            return [word.lower()]


class ZeyrekAnalyzer(MorphAnalyzerBase):
    """
    Saf Python 'zeyrek' kütüphanesi ile morfolojik analiz.
    """
    def __init__(self):
        import zeyrek
        self._analyzer = zeyrek.MorphAnalyzer()
        logger.info("Zeyrek (saf Python) morfoloji motoru başlatıldı.")

    # Kelimeyi analiz eder ve morfolojik analiz sonuçlarını döndürür.
    def analyze(self, word: str) -> list[str]:
        try:
            parses = self._analyzer.analyze(word)
            candidates = []
            for word_parses in parses:
                for p in word_parses:
                    candidates.append(str(p))
            return candidates if candidates else [f"UNK_{word}"]
        except Exception:
            return [f"UNK_{word}"]


    # Kelimenin köklerini döndürür.
    def get_stems(self, word: str) -> list[str]:
        """Zeyrek'ten kelime köklerini çıkarır."""
        try:
            parses = self._analyzer.analyze(word)
            stems = []
            for word_parses in parses:
                for p in word_parses:
                    s = str(p)
                    if ':' in s:
                        parts = s.split()
                        for part in parts:
                            if ':' in part:
                                root = part.split(':')[0]
                                if root:
                                    stems.append(root.lower())
                                break
            return list(set(stems)) if stems else [word.lower()]
        except Exception:
            return [word.lower()]

# Zembereği başlatır, zemberek mevcut değilse zeyreği kullanır.
def create_morph_analyzer() -> MorphAnalyzerBase:
    if os.path.isfile(ZEMBEREK_JAR):
        try:
            return ZemberekAnalyzer(ZEMBEREK_JAR)
        except Exception as e:
            logger.warning(f"Zemberek başlatılamadı: {e}. Zeyrek'e geçiliyor...")

    try:
        return ZeyrekAnalyzer()
    except Exception as e:
        logger.error(f"Zeyrek de başlatılamadı: {e}")
        raise RuntimeError(
        )


# =====================================================================
# BÖLÜM 2: VERİ OKUMA
# =====================================================================

def read_conllu(filepath: str, max_sents: int = None) -> list:
    """
    CoNLL-U dosyasını okur. Her bir kelimeyi alır ve bir sözlük (dictionary) haline getirir.
    """
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            if max_sents and len(sentences) >= max_sents:
                break

            tokens = []
            for token in sentence:
                if isinstance(token['id'], tuple) or not isinstance(token['id'], int):
                    continue

                form = token['form']
                lemma = token['lemma'] or form
                upos = token['upos'] or '_'
                xpos = token['xpos'] or '_'
                feats = token['feats']

                if feats:
                    feats_str = '|'.join(f"{k}={v}" for k, v in sorted(feats.items()))
                else:
                    feats_str = '_'

                if feats_str != '_':
                    gold_tag = f"{upos}|{feats_str}"
                else:
                    gold_tag = upos

                tokens.append({
                    'form': form,
                    'lemma': lemma,               
                    'upos': upos,
                    'xpos': xpos,                  
                    'feats_str': feats_str,
                    'gold_tag': gold_tag,
                    'token_obj': token,
                })
            if tokens:
                sentences.append(tokens)
    logger.info(f"  -> {filepath}: {len(sentences)} cümle okundu.")
    return sentences


def add_zemberek_candidates(sentences: list, analyzer: MorphAnalyzerBase) -> list:
    """
    Okunan kelimeleri zembereğe gönderir, zemberek kelimenin kökünü alabileceği ekleri tahmin eder.
    """
    logger.info("Morfolojik aday çözümler üretiliyor...")
    for sent in tqdm(sentences, desc="Morfolojik analiz"):
        for tok in sent:
            tok['zemberek_candidates'] = analyzer.analyze(tok['form'])
            tok['zemberek_stems'] = analyzer.get_stems(tok['form'])
    return sentences


# =====================================================================
# BÖLÜM 3: ÖZELLİK ÇIKARIMI (Feature Extraction)
# =====================================================================

def word2features(sent: list, i: int, upos_list: list = None) -> dict:
    """
    Genişletilmiş CRF özellik vektörü.
    [-2, +2] bağlam penceresi, kelimenin son 5 karakteri inceler.
    """
    word = sent[i]['form']
    word_lower = word.lower()
    candidates = sent[i].get('zemberek_candidates', [])
    stems = sent[i].get('zemberek_stems', [])
    lemma = sent[i].get('lemma', word_lower)         
    xpos = sent[i].get('xpos', '_')               


    features = {
        'bias': 1.0,
        'word.lower()': word_lower,
        'word.length': len(word_lower),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.has_hyphen': '-' in word,
        'word.has_apostrophe': "'" in word or '\u2019' in word,
    }

    features['lemma'] = lemma.lower()
    features['xpos'] = xpos

    # ── UPOS ÖZELLİKLERİ──
    if upos_list is not None:
        features['upos'] = upos_list[i]

    # ── TÜRKÇE EKLER ──
    for length in range(1, 6):
        if len(word_lower) >= length:
            features[f'word[-{length}:]'] = word_lower[-length:]
            features[f'word[:{length}]'] = word_lower[:length]

    # ── ZEMBEREK / ZEYREK ADAYLARI ──
    for idx, cand in enumerate(candidates[:5]):
        features[f'zemberek_cand_{idx}'] = cand
    features['zemberek_n_candidates'] = len(candidates)

    # ── KELİME KÖKLERİ ──
    for idx, stem in enumerate(stems[:3]):
        features[f'stem_{idx}'] = stem
    features['n_stems'] = len(stems)

    # ── ÖNCEKİ KELİMELER (i-1 ve i-2) ──
    if i > 0:
        w1 = sent[i-1]['form']
        w1_lower = w1.lower()
        features.update({
            '-1:word.lower()': w1_lower,
            '-1:word.istitle()': w1.istitle(),
            '-1:word.isupper()': w1.isupper(),
            '-1:lemma': sent[i-1].get('lemma', w1_lower).lower(),
            '-1:xpos': sent[i-1].get('xpos', '_'),
        })
        if upos_list is not None:
            features['-1:upos'] = upos_list[i-1]
        if len(w1_lower) >= 2:
            features['-1:word[-2:]'] = w1_lower[-2:]
        if len(w1_lower) >= 3:
            features['-1:word[-3:]'] = w1_lower[-3:]
        for idx, cand in enumerate(sent[i-1].get('zemberek_candidates', [])[:3]):
            features[f'-1:zemberek_cand_{idx}'] = cand

        if i > 1:
            w2 = sent[i-2]['form']
            features.update({
                '-2:word.lower()': w2.lower(),
                '-2:word.istitle()': w2.istitle(),
                '-2:xpos': sent[i-2].get('xpos', '_'),             
            })
            if upos_list is not None:
                features['-2:upos'] = upos_list[i-2]
        else:
            features['BOS2'] = True
    else:
        features['BOS'] = True

    # ── SONRAKİ KELİMELER ──
    if i < len(sent) - 1:
        wn1 = sent[i+1]['form']
        wn1_lower = wn1.lower()
        features.update({
            '+1:word.lower()': wn1_lower,
            '+1:word.istitle()': wn1.istitle(),
            '+1:word.isupper()': wn1.isupper(),
            '+1:lemma': sent[i+1].get('lemma', wn1_lower).lower(),  
            '+1:xpos': sent[i+1].get('xpos', '_'),                 
        })
        if upos_list is not None:
            features['+1:upos'] = upos_list[i+1]
        if len(wn1_lower) >= 2:
            features['+1:word[-2:]'] = wn1_lower[-2:]
        if len(wn1_lower) >= 3:
            features['+1:word[-3:]'] = wn1_lower[-3:]
        for idx, cand in enumerate(sent[i+1].get('zemberek_candidates', [])[:3]):
            features[f'+1:zemberek_cand_{idx}'] = cand

        if i < len(sent) - 2:
            wn2 = sent[i+2]['form']
            features.update({
                '+2:word.lower()': wn2.lower(),
                '+2:word.istitle()': wn2.istitle(),
                '+2:xpos': sent[i+2].get('xpos', '_'),               
            })
            if upos_list is not None:
                features['+2:upos'] = upos_list[i+2]
        else:
            features['EOS2'] = True
    else:
        features['EOS'] = True

    return features

# Cümledeki her kelime için word2features fonksiyonunu çalıştırır.
def sent2features(sent: list, upos_list: list = None) -> list[dict]:
    """Bir cümledeki tüm kelimeler için özellik vektörlerini döndürür."""
    return [word2features(sent, i, upos_list) for i in range(len(sent))]

# Eğitim için cümledeki her kelime için doğru etiketi döndürür.
def sent2labels(sent: list, target_type: str = 'joint') -> list[str]:
    """
    Bir cümledeki tüm kelimelerin doğru etiketlerini döndürür.
    target_type:
      'joint' -> upos|feats_str (veya feats yoksa sadece upos)
      'upos'  -> upos
      'feats' -> feats_str
    """
    if target_type == 'upos':
        return [tok['upos'] for tok in sent]
    elif target_type == 'feats':
        return [tok['feats_str'] for tok in sent]
    else:
        return [tok['gold_tag'] for tok in sent]


# =====================================================================
# BÖLÜM 4: MODEL EĞİTİMİ
# =====================================================================

USE_HYPEROPT = False

def train_crf_model(X_train, y_train, model_name="CRF", all_possible_transitions=False):
    """
    CRF modelini eğitir.
    """
    if USE_HYPEROPT:
        return _train_with_hyperopt(X_train, y_train, model_name=model_name, all_possible_transitions=all_possible_transitions)

    logger.info(f"{model_name} modeli eğitiliyor (L-BFGS, c1=0.1, c2=0.1, max_iterations=20, min_freq=3)...")
    logger.info("  Not: Eğitim süresi optimize edilmiştir.")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=20,
        min_freq=3,
        all_possible_transitions=all_possible_transitions,
        verbose=True,
    )
    crf.fit(X_train, y_train)
    logger.info(f"{model_name} eğitimi tamamlandı.")
    logger.info(f"  -> Toplam etiket sınıfı sayısı: {len(crf.classes_)}")
    return crf

# Model eğitimi için en iyi parametreleri bulmak için deneme yanılma yapar.
def _train_with_hyperopt(X_train, y_train, model_name="CRF", all_possible_transitions=False):
    """RandomizedSearchCV ile c1/c2 hiperparametre optimizasyonu."""
    logger.info(f"{model_name} hiperparametre optimizasyonu başlatılıyor...")

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=30,
        min_freq=3,
        all_possible_transitions=all_possible_transitions,
    )

    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    f1_scorer = make_scorer(
        crf_metrics.flat_f1_score,
        average='weighted',
    )

    rs = RandomizedSearchCV(
        crf,
        params_space,
        cv=3,
        verbose=1,
        n_jobs=-1,
        n_iter=10,
        scoring=f1_scorer,
        random_state=42,
    )

    rs.fit(X_train, y_train)
    logger.info(f"{model_name} optimizasyonu tamamlandı!")
    logger.info(f"  -> En iyi c1 ve c2: {rs.best_params_}")
    logger.info(f"  -> En iyi Weighted F1 Skoru: {rs.best_score_:.4f}")
    return rs.best_estimator_


# =====================================================================
# BÖLÜM 5: DEĞERLENDİRME VE GÖRSELLEŞTİRME
# =====================================================================

def evaluate_model(y_test, y_pred):
    """Tahmin sonuçlarını gerçek etiketlerle karşılaştırarak değerlendirir."""
    logger.info("Tahmin sonuçları değerlendiriliyor...")

    y_test_flat = [label for sent in y_test for label in sent]
    y_pred_flat = [label for sent in y_pred for label in sent]

    accuracy = accuracy_score(y_test_flat, y_pred_flat)
    logger.info(f"\n{'='*60}")
    logger.info(f"  GENEL DOĞRULUK (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"{'='*60}")

    label_counts = Counter(y_test_flat)
    top_labels = [label for label, _ in label_counts.most_common(20)]

    known_labels = set(y_test_flat).union(set(y_pred_flat))
    sorted_labels = sorted(
        [l for l in top_labels if l in known_labels],
        key=lambda x: label_counts[x],
        reverse=True
    )

    print("\n" + "="*80)
    print("  SINIFLANDIRMA RAPORU")
    print("="*80)
    report = crf_metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=4
    )
    print(report)

    return y_test_flat, y_pred_flat


def plot_confusion_matrix(y_true_flat, y_pred_flat, output_path):
    """En sık 10 sınıf için karışıklık matrisi ısı haritası oluşturur."""
    logger.info("Karışıklık matrisi (Confusion Matrix) oluşturuluyor...")

    label_counts = Counter(y_true_flat)
    top10_labels = [label for label, _ in label_counts.most_common(10)]

    mask = [(yt in top10_labels) for yt in y_true_flat]
    y_true_filtered = [yt for yt, m in zip(y_true_flat, mask) if m]
    y_pred_filtered = [yp for yp, m in zip(y_pred_flat, mask) if m]

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top10_labels)

    label_translations = {
        'PUNCT': 'Noktalama',
        'NOUN|Case=Nom|Number=Sing|Person=3': 'Yalın İsim',
        'ADJ': 'Sıfat',
        'ADV': 'Zarf',
        'CCONJ': 'Bağlaç',
        'ADP': 'Edat',
        'PROPN|Case=Nom|Number=Sing|Person=3': 'Yalın Özel İsim',
        'VERB|Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Past': 'Geçmiş Zaman Fiili',
        'DET|PronType=Ind': 'Belgisiz Sıfat/Zarf',
        'NOUN|Case=Nom|Number=Sing|Number[psor]=Sing|Person=3|Person[psor]=3': 'İyelik Ekli İsim',
    }

    display_labels = []
    for label in top10_labels:
        translation = label_translations.get(label, 'Diğer')
        display_labels.append(f"{label}\n({translation})")

    plt.figure(figsize=(18, 14))
    sns.set_theme(style="whitegrid", font_scale=0.8)

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=display_labels, yticklabels=display_labels,
        linewidths=0.5, linecolor='gray',
        cbar_kws={'label': 'Sayı (Count)', 'shrink': 0.8}
    )

    plt.title('Karışıklık Matrisi — En Sık 10 Morfolojik Etiket ve Türkçe Karşılıkları',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Tahmin Edilen Etiket (Predicted)', fontsize=12, labelpad=10)
    plt.ylabel('Gerçek Etiket (True)', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  -> Isı haritası kaydedildi: {output_path}")


# =====================================================================
# BÖLÜM 6: CONLL-U ÇIKTI DOSYASI
# =====================================================================

def write_predictions_conllu(test_file_path, y_pred, output_path):
    """Test verisinin orijinal CoNLL-U yapısını koruyarak tahminleri yazar."""
    logger.info("Tahmin sonuçları CoNLL-U formatında yazılıyor...")

    pred_flat = [tag for sent_preds in y_pred for tag in sent_preds]
    pred_idx = 0

    output_lines = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            if line.startswith('#') or line.strip() == '':
                output_lines.append(line)
                continue

            fields = line.split('\t')

            if '-' in str(fields[0]) or '.' in str(fields[0]):
                output_lines.append(line)
                continue

            if pred_idx < len(pred_flat):
                predicted_tag = pred_flat[pred_idx]
                pred_idx += 1

                parts = predicted_tag.split('|', 1)
                pred_upos = parts[0]
                pred_feats = parts[1] if len(parts) > 1 else '_'

                if len(fields) >= 10:
                    fields[3] = pred_upos
                    fields[5] = pred_feats

            output_lines.append('\t'.join(fields))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines) + '\n')

    logger.info(f"  -> Tahmin çıktısı kaydedildi: {output_path}")
    logger.info(f"  -> Toplam {pred_idx} token yazıldı.")


# =====================================================================
# ANA PIPELINE
# =====================================================================

def main():
    """Tam pipeline: Veri okuma → Analiz → Özellik çıkarımı (UPOS/FEATS) → İki Aşamalı Eğitim → Birleştirme → Değerlendirme"""

    print("=" * 70)
    print("  TÜRKÇE MORFOLOJİK ÇÖZÜMLEME (MORPHOLOGICAL DISAMBIGUATION)")
    print("  CRF Tabanlı İki Aşamalı Etiketleme Modeli (UPOS -> FEATS)")
    print("=" * 70)

    for fpath in [TRAIN_FILE, TEST_FILE]:
        if not os.path.isfile(fpath):
            logger.error(f"Dosya bulunamadı: {fpath}")
            sys.exit(1)

    # ── Morfolojik Analizör ──
    analyzer = create_morph_analyzer()

    # ── Veri Okuma ──
    logger.info("Eğitim verisi okunuyor...")
    train_sentences = read_conllu(TRAIN_FILE, max_sents=MAX_TRAIN_SENTS)
    logger.info("Test verisi okunuyor...")
    test_sentences = read_conllu(TEST_FILE, max_sents=MAX_TEST_SENTS)

    # ── Zemberek/Zeyrek Aday Üretimi ──
    train_sentences = add_zemberek_candidates(train_sentences, analyzer)
    test_sentences = add_zemberek_candidates(test_sentences, analyzer)

    # ── 1. AŞAMA (UPOS): Özellik Çıkarımı ve Eğitim ──
    logger.info("1. Aşama (UPOS) özellik vektörleri oluşturuluyor...")
    X_train_upos = [sent2features(s, upos_list=None) for s in tqdm(train_sentences, desc="Eğitim UPOS özellikleri")]
    y_train_upos = [sent2labels(s, target_type='upos') for s in train_sentences]
    
    X_test_upos  = [sent2features(s, upos_list=None) for s in tqdm(test_sentences, desc="Test UPOS özellikleri")]
    y_test_upos  = [sent2labels(s, target_type='upos') for s in test_sentences]

    logger.info("1. Aşama (UPOS) model eğitimi başlatılıyor...")
    crf_upos = train_crf_model(X_train_upos, y_train_upos, model_name="Stage-1 (UPOS)", all_possible_transitions=True)

    # ── UPOS Tahminleri ──
    logger.info("Test verisi için UPOS tahminleri yapılıyor...")
    y_pred_upos = crf_upos.predict(X_test_upos)

    # ── 2. AŞAMA (FEATS): Özellik Çıkarımı ve Eğitim ──
    logger.info("2. Aşama (FEATS) özellik vektörleri oluşturuluyor...")
    
    # Eğitim için doğru UPOS etiketlerini kullanıyoruz 
    train_upos_gold = [[tok['upos'] for tok in s] for s in train_sentences]
    X_train_feats = [sent2features(s, upos_list=gold_upos) for s, gold_upos in zip(tqdm(train_sentences, desc="Eğitim FEATS özellikleri"), train_upos_gold)]
    y_train_feats = [sent2labels(s, target_type='feats') for s in train_sentences]

    # Test için 1. aşamadan tahmin edilen UPOS etiketlerini kullanıyoruz
    X_test_feats = [sent2features(s, upos_list=pred_upos) for s, pred_upos in zip(tqdm(test_sentences, desc="Test FEATS özellikleri"), y_pred_upos)]

    logger.info("2. Aşama (FEATS) model eğitimi başlatılıyor...")
    crf_feats = train_crf_model(X_train_feats, y_train_feats, model_name="Stage-2 (FEATS)", all_possible_transitions=False)

    # ── FEATS Tahminleri ──
    logger.info("Test verisi için FEATS tahminleri yapılıyor...")
    y_pred_feats = crf_feats.predict(X_test_feats)

    # ── TAHMİNLERİN BİRLEŞTİRİLMESİ ──
    logger.info("UPOS ve FEATS tahminleri birleştiriliyor...")
    y_pred = []
    for i in range(len(test_sentences)):
        sent_pred = []
        for j in range(len(test_sentences[i])):
            u = y_pred_upos[i][j]
            f = y_pred_feats[i][j]
            if f != '_':
                sent_pred.append(f"{u}|{f}")
            else:
                sent_pred.append(u)
        y_pred.append(sent_pred)

    y_test = [sent2labels(s, target_type='joint') for s in test_sentences]

    logger.info(f"  Eğitim: {len(train_sentences)} cümle, {sum(len(s) for s in train_sentences)} token")
    logger.info(f"  Test  : {len(test_sentences)} cümle, {sum(len(s) for s in test_sentences)} token")

    all_train_labels = [l for s in train_sentences for l in sent2labels(s, target_type='joint')]
    n_unique = len(set(all_train_labels))
    logger.info(f"  Benzersiz birleşik etiket sayısı: {n_unique}")

    # ── Değerlendirme ──
    y_test_flat, y_pred_flat = evaluate_model(y_test, y_pred)

    # ── Karışıklık Matrisi ──
    plot_confusion_matrix(y_test_flat, y_pred_flat, HEATMAP_FILE)

    # ── CoNLL-U Çıktı ──
    write_predictions_conllu(TEST_FILE, y_pred, OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("  PIPELINE TAMAMLANDI")
    print(f"  -> Karışıklık Matrisi : {HEATMAP_FILE}")
    print(f"  -> Tahmin Çıktısı     : {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == '__main__':
    main()
