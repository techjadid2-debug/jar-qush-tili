"""
dataset_loader.py
=================
Xeno-canto API v3 orqali Kanareyka (Serinus canaria) qo'shig'i yozuvlarini
yuklab olish va MP3 → WAV formatiga o'tkazish skripti.

API v3 uchun xeno-canto.org da ro'yxatdan o'ting va API kalitini oling:
    https://xeno-canto.org/account

API kalitini .env faylga yozing:
    XC_API_KEY=sizning_kalitingiz

Yoki to'g'ridan-to'g'ri argument sifatida bering:
    python dataset_loader.py --key sizning_kalitingiz

Ishlatish:
    python dataset_loader.py --key YOUR_KEY
    python dataset_loader.py --key YOUR_KEY --max 10 --quality A

Natija:
    data/raw/*.wav         — audio fayllar
    data/raw/metadata.json — yozuvlar haqida ma'lumot
"""

import os
import json
import time
import argparse
import requests
from pathlib import Path

# Pydub (MP3→WAV). Agar yo'q bo'lsa: pip install pydub
# ffmpeg ham kerak:  brew install ffmpeg
try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False
    print("⚠️  pydub topilmadi: pip install pydub")

try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False

# ─────────────────────────────────────────────
# SOZLAMALAR
# ─────────────────────────────────────────────
XENO_CANTO_API = "https://xeno-canto.org/api/3/recordings"
RAW_DIR        = Path("data/raw")
MAX_RETRIES    = 3          # Har bir so'rovda qayta urinish soni
RETRY_DELAY    = 2.0        # Qayta urinishlar orasidagi kutish (soniya)
REQUEST_DELAY  = 1.5        # So'rovlar orasidagi kutish (serverni himoya qilish)


def fetch_recordings(species: str, quality: str, rec_type: str, max_count: int, api_key: str) -> list[dict]:
    """
    Xeno-canto API v3 dan yozuvlar ro'yxatini olish.

    API v3 o'zgarishlari:
    - Endpoint: /api/3/recordings
    - 'key' parametri majburiy
    - Query format: 'sp:"Serinus canaria"' (tag asosida)

    Args:
        species:   Qush turi ilmiy nomi (masalan "Serinus canaria")
        quality:   Sifat filtri ("A", "B", "C", ...)
        rec_type:  Yozuv turi ("song", "call", ...)
        max_count: Eng ko'pi bilan nechta yozuv olish
        api_key:   Xeno-canto API kaliti

    Returns:
        Yozuvlar ro'yxati (dict'lar)
    """
    print(f"\n🔍 Xeno-canto v3 da qidirilmoqda: {species!r} | sifat={quality} | tur={rec_type}")

    # API v3 query format: sp:"Genus species" + q:A + type:song
    # Ikkita so'zli nomlar qo'shtirnoq ichida bo'lishi shart
    query = f'sp:"{species}" q:{quality} type:{rec_type}'
    params = {
        "query": query,
        "key":   api_key,
        "per_page": min(max_count, 100),  # API v3: 50-500 oralig'ida
    }

    # Qayta urinish bilan so'rov yuborish
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(XENO_CANTO_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as e:
            print(f"  ⚠️  So'rovda xato (urinish {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise RuntimeError(f"Xeno-canto API'ga ulana olmadi: {e}") from e

    recordings = data.get("recordings", [])
    total      = data.get("numRecordings", "?")
    print(f"   Jami topildi: {total} ta yozuv → birinchi {min(max_count, len(recordings))} tasi yuklanadi")

    return recordings[:max_count]


def download_audio(url: str, dest_path: Path) -> bool:
    """
    Audio faylni yuklab olish.

    Args:
        url:       Yuklab olish manzili
        dest_path: Saqlash joyi

    Returns:
        True — muvaffaqiyatli, False — xato
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()

            # Faylni yozish
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Bo'sh fayl tekshiruvi
            if dest_path.stat().st_size < 1024:
                print(f"  ⚠️  Fayl juda kichik (bo'sh?): {dest_path.name}")
                dest_path.unlink(missing_ok=True)
                return False

            return True

        except requests.RequestException as e:
            print(f"  ⚠️  Yuklab olishda xato (urinish {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                dest_path.unlink(missing_ok=True)
                return False


def mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """
    MP3 faylni WAV formatiga o'tkazish (pydub yordamida).

    Args:
        mp3_path: Manba MP3 fayl
        wav_path: Maqsad WAV fayl

    Returns:
        True — muvaffaqiyatli, False — xato
    """
    if not PYDUB_OK:
        print("  ❌  pydub o'rnatilmagan — MP3→WAV o'tkazilmadi")
        return False

    try:
        audio = AudioSegment.from_mp3(mp3_path)
        # 22050 Hz, mono, 16-bit — TweetyBERT standart format
        audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format="wav")
        return True
    except Exception as e:
        print(f"  ❌  MP3→WAV xato ({mp3_path.name}): {e}")
        print("      Eslatma: ffmpeg o'rnatilganmi? → brew install ffmpeg")
        return False


def load_dataset(
    species:   str = "Serinus canaria",
    quality:   str = "A",
    rec_type:  str = "song",
    max_count: int = 8,
    keep_mp3:  bool = False,
    api_key:   str = "",
) -> None:
    """
    Asosiy funksiya: Xeno-canto'dan audio yuklab, WAV ga o'tkazib saqlash.

    Args:
        species:   Qush turi ilmiy nomi
        quality:   Xeno-canto sifat filtri
        rec_type:  Yozuv turi
        max_count: Yuklab olish kerak bo'lgan yozuvlar soni
        keep_mp3:  MP3 faylni saqlash (True) yoki o'chirish (False)
        api_key:   Xeno-canto API v3 kaliti
    """
    # API key tekshiruvi
    if not api_key:
        # .env fayldan yoki muhit o'zgaruvchisidan urinib ko'rish
        import os
        api_key = os.environ.get("XC_API_KEY", "")
    if not api_key:
        print("❌  API key topilmadi!")
        print("   Xeno-canto v3 API key talab qiladi.")
        print("   1. https://xeno-canto.org/account sahifasiga kiring")
        print("   2. API kalitingizni nusxalang")
        print("   3. Ishlatish: python dataset_loader.py --key SIZNING_KALITINGIZ")
        print("   Yoki: export XC_API_KEY=SIZNING_KALITINGIZ")
        return
    # Papkalarni yaratish
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # API'dan yozuvlar ro'yxatini olish
    recordings = fetch_recordings(species, quality, rec_type, max_count, api_key)

    if not recordings:
        print("❌  Yozuvlar topilmadi!")
        return

    # Metadata ro'yxati
    metadata = []
    success_count = 0

    iterable = tqdm(recordings, desc="⬇️  Yuklanmoqda") if TQDM_OK else recordings

    for rec in iterable:
        rec_id    = rec.get("id", "unknown")
        rec_name  = rec.get("en", "unknown")       # Inglizcha nomi
        country   = rec.get("cnt", "unknown")      # Mamlakat
        duration  = rec.get("length", "?")         # Davomiyligi
        file_url  = rec.get("file", "")
        file_name = rec.get("file-name", f"{rec_id}.mp3")

        if not file_url:
            print(f"  ⚠️  {rec_id}: URL yo'q, o'tkazildi")
            continue

        # URL'ni tekshirish va to'liq qilish
        if not file_url.startswith("http"):
            file_url = "https:" + file_url

        print(f"\n  📥 [{rec_id}] {rec_name} ({country}) — {duration}")

        # Saqlash yo'llari
        mp3_path = RAW_DIR / f"xc{rec_id}.mp3"
        wav_path = RAW_DIR / f"xc{rec_id}.wav"

        # Allaqachon yuklanganmi?
        if wav_path.exists():
            print(f"  ✅  Allaqachon mavjud: {wav_path.name}")
            success_count += 1
        else:
            # MP3 yuklab olish
            ok = download_audio(file_url, mp3_path)
            if not ok:
                continue

            # MP3 → WAV
            ok = mp3_to_wav(mp3_path, wav_path)
            if not ok:
                continue

            # MP3 ni o'chirish (agar keep_mp3=False)
            if not keep_mp3:
                mp3_path.unlink(missing_ok=True)

            success_count += 1
            print(f"  ✅  Saqlandi: {wav_path.name} ({wav_path.stat().st_size // 1024} KB)")

        # Metadata yig'ish
        metadata.append({
            "id":        rec_id,
            "file":      wav_path.name,
            "species":   rec.get("gen", "") + " " + rec.get("sp", ""),
            "name_en":   rec_name,
            "country":   country,
            "location":  rec.get("loc", ""),
            "date":      rec.get("date", ""),
            "duration":  duration,
            "quality":   rec.get("q", ""),
            "recordist": rec.get("rec", ""),
            "url":       f"https://xeno-canto.org/{rec_id}",
        })

        # Serverni himoya qilish uchun kutish
        time.sleep(REQUEST_DELAY)

    # Metadata JSON sifatida saqlash
    meta_path = RAW_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Yakuniy hisobot
    print(f"\n{'═' * 50}")
    print(f"✅  Muvaffaqiyatli yuklandi: {success_count}/{len(recordings)} ta fayl")
    print(f"📂  Papka: {RAW_DIR.resolve()}")
    print(f"📋  Metadata: {meta_path.name}")
    print(f"{'═' * 50}\n")


# ─────────────────────────────────────────────
# COMMAND LINE INTERFACE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xeno-canto'dan qush ovozi yozuvlarini yuklab olish",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--species", type=str, default="Serinus canaria",
        help="Qush turi ilmiy nomi"
    )
    parser.add_argument(
        "--quality", type=str, default="A",
        choices=["A", "B", "C", "D", "E"],
        help="Xeno-canto sifat filtri (A = eng yaxshi)"
    )
    parser.add_argument(
        "--type", dest="rec_type", type=str, default="song",
        help="Yozuv turi: song, call, alarm, ..."
    )
    parser.add_argument(
        "--max", type=int, default=8,
        help="Yuklab olish kerak bo'lgan yozuvlar soni"
    )
    parser.add_argument(
        "--keep-mp3", action="store_true",
        help="MP3 fayllarni saqlash (odatda o'chiriladi)"
    )
    parser.add_argument(
        "--key", type=str, default="",
        help="Xeno-canto API v3 kaliti (https://xeno-canto.org/account)"
    )

    args = parser.parse_args()

    load_dataset(
        species   = args.species,
        quality   = args.quality,
        rec_type  = args.rec_type,
        max_count = args.max,
        keep_mp3  = args.keep_mp3,
        api_key   = args.key,
    )
