# 🐦 Jar-Qush Tili — Kanareyka Ovozi Tadqiqot Loyihasi

> **Kanareyka qushining (Serinus canaria) ovoz signallarini TweetyBERT self-supervised transformer yordamida tahlil qilish.**

---

## 📌 Maqsad

- Xeno-canto'dan kanareyka qo'shig'i yozuvlarini yuklab olish
- Mel-spektrogramma hisoblash
- TweetyBERT encoder orqali bo'g'in (syllable) embedding'larini chiqarish
- Bo'g'inlarni segmentatsiya qilish va klasterlash
- Apple M3 MPS orqali GPU tezlashtirish

---

## 🛠️ Texnologiyalar

| Vosita | Maqsad |
|--------|--------|
| Python 3.11 | Asosiy til |
| PyTorch (MPS) | Apple M3 GPU hisob |
| TweetyBERT | Self-supervised audio transformer |
| librosa | Audio qayta ishlash |
| Xeno-canto API | Dataset manbai |
| Kaggle T4 | Model o'qitish (cloud) |
| UMAP + matplotlib | Vizualizatsiya |

---

## 📁 Loyiha Tuzilmasi

```
Jar-qush tili/
├── tweety_bert/              ← TweetyBERT modeli (GitHub'dan)
├── data/
│   ├── raw/                  ← Xeno-canto'dan .wav fayllar
│   └── processed/            ← Qayta ishlangan ma'lumotlar
├── outputs/
│   ├── spectrograms/         ← Spektrogramma rasmlari (.png)
│   └── embeddings/           ← Bo'g'in embedding'lari (.json)
├── dataset_loader.py         ← Xeno-canto API → WAV yuklovchi
├── explore_jar.py            ← TweetyBERT inference skripti
├── kaggle_train.ipynb        ← Kaggle T4 da o'qitish notebooki
├── environment.yml           ← Conda muhit (Apple M3)
└── README.md
```

---

## 🚀 Ishga Tushirish

### 1. Muhitni sozlash

```bash
# Conda muhitni yaratish
conda env create -f environment.yml
conda activate jar-env

# MPS mavjudligini tekshirish
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# ffmpeg (MP3→WAV uchun)
brew install ffmpeg

# TweetyBERT'ni yuklab olish
git clone https://github.com/georgevenven/tweety_bert.git
```

### 2. Dataset yuklab olish

```bash
python dataset_loader.py
# Natija: data/raw/ papkasida 8 ta .wav fayl
```

### 3. Inference va embedding chiqarish

```bash
python explore_jar.py --audio data/raw/
# Natija: outputs/embeddings/*.json va outputs/spectrograms/*.png
```

### 4. Kaggle'da o'qitish

`kaggle_train.ipynb` ni Kaggle'ga yuklang → GPU T4 ni yoqing → Run All

---

## 📊 Natijalar Formati

```json
{
  "audio_file": "kanareyka_001.wav",
  "duration_sec": 12.5,
  "device": "mps",
  "syllables": [
    {
      "onset_time": 0.42,
      "duration_sec": 0.18,
      "embedding": [0.12, -0.34, "..."]
    }
  ]
}
```

---

## 📚 Manbalar

- [TweetyBERT GitHub](https://github.com/georgevenven/tweety_bert)
- [Xeno-canto API](https://xeno-canto.org/explore/api)
- [Kaggle Notebooks](https://www.kaggle.com/code)
- [librosa Documentation](https://librosa.org/doc/)

---

## 👤 Muallif

**Abdulvosit** — 2026
