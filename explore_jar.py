"""
explore_jar.py
==============
Kanareyka audio fayllarini TweetyBERT yordamida tahlil qilish skripti.

Pipeline:
  WAV → STFT Spektrogramma → NPZ → TweetyBERT inference → Embedding + Syllable JSON

Ishlatish:
    # Bitta fayl
    python explore_jar.py --audio data/raw/xc12345.wav

    # Barcha fayllar
    python explore_jar.py --audio data/raw/

    # Checkpoint bilan (Kaggle'dan yuklab olgandan keyin)
    python explore_jar.py --audio data/raw/ --checkpoint checkpoints/model.pt

Natijalar:
    outputs/spectrograms/*.png  — spektrogramma rasmlari
    outputs/embeddings/*.json   — bo'g'in embedding'lari
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Kutubxona importlari
# ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("❌  PyTorch topilmadi. O'rnatish: pip install torch")

try:
    import librosa
    import librosa.display
    import soundfile as sf
except ImportError:
    sys.exit("❌  librosa topilmadi. O'rnatish: pip install librosa soundfile")

try:
    import matplotlib
    matplotlib.use("Agg")  # GUI yo'q serverlarda ishlashi uchun
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("❌  matplotlib topilmadi. O'rnatish: pip install matplotlib")

try:
    from scipy.signal import ellip, filtfilt
except ImportError:
    sys.exit("❌  scipy topilmadi. O'rnatish: pip install scipy")

# TweetyBERT'ni import qilish
TWEETY_SRC = Path(__file__).parent / "tweety_bert" / "src"
if not TWEETY_SRC.exists():
    sys.exit(
        "❌  tweety_bert/src topilmadi!\n"
        "   git clone https://github.com/georgevenven/tweety_bert.git"
    )
sys.path.insert(0, str(TWEETY_SRC.parent))
sys.path.insert(0, str(TWEETY_SRC))

from src.model import TweetyBERT  # noqa: E402

# ─────────────────────────────────────────────────────────────
# SOZLAMALAR — TweetyBERT asl maqolasidan olingan
# ─────────────────────────────────────────────────────────────
# Spektrogramma
NFFT      = 1024    # FFT hajmi
HOP_LEN   = 119     # Hop length (step_size)
FREQ_BINS = 196     # STFT chiqish chastota binlari soni (nfft//2 + 1 = 513, keyin 196 olinadi)
SPEC_ROWS = 196     # Model kutadigan chastota binlari (data_class.py: spectrogram[20:216])

# TweetyBERT model parametrlari (maqola defaults)
MODEL_CFG = dict(
    d_transformer     = 256,   # Transformer o'lchami
    nhead_transformer = 4,     # Attention boshlar soni
    num_freq_bins     = SPEC_ROWS,
    num_labels        = 40,    # Sinf soni (kanareyka bo'g'inlari)
    dropout           = 0.1,
    transformer_layers = 3,
    dim_feedforward   = 128,
    m                 = 33,
    p                 = 0.01,
    alpha             = 1,
    length            = 1000,  # Kontekst oyna uzunligi
    pos_enc_type      = "relative",
)

# Papkalar
OUT_SPEC   = Path("outputs/spectrograms")
OUT_EMB    = Path("outputs/embeddings")

# ─────────────────────────────────────────────────────────────
# QURILMA TANLASH: MPS (Apple M3) → CUDA → CPU
# ─────────────────────────────────────────────────────────────
def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("🍎  Apple MPS (Metal) ishlatilmoqda — M-chip GPU tezlashtirish")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"🟢  CUDA GPU ishlatilmoqda: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("🔵  CPU ishlatilmoqda (MPS yoki CUDA topilmadi)")
    return dev


# ─────────────────────────────────────────────────────────────
# MODEL YUKLAB OLISH
# ─────────────────────────────────────────────────────────────
def load_model(device: torch.device, checkpoint_path: str | None = None) -> TweetyBERT:
    """
    TweetyBERT modelini yaratish va (ixtiyoriy) checkpoint'ni yuklash.

    Args:
        device:          Torch qurilmasi
        checkpoint_path: .pt fayl yo'li (None = tasodifiy og'irliklar)

    Returns:
        eval() rejimidagi TweetyBERT modeli
    """
    print("\n🧠  TweetyBERT modeli yaratilmoqda...")

    # device str sifatida uzatiladi (model ichida torch.device() chaqiriladi)
    model = TweetyBERT(**MODEL_CFG, device=str(device))
    model = model.to(device)

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"  ⚠️  Checkpoint topilmadi: {ckpt_path}")
            print("  ℹ️   Tasodifiy og'irliklar bilan davom etilmoqda")
        else:
            print(f"  📂  Checkpoint yuklanmoqda: {ckpt_path.name}")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Ba'zi checkpoint'larda 'model_state_dict' kaliti bo'lishi mumkin
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
            print("  ✅  Checkpoint muvaffaqiyatli yuklandi")
    else:
        print("  ℹ️   Pretrained checkpoint ko'rsatilmadi → tasodifiy og'irliklar ishlatilmoqda")
        print("       (Haqiqiy tahlil uchun: --checkpoint checkpoints/model.pt)")

    model.eval()
    # MPS float32 ishlatadi (float16 ba'zi operatsiyalarda xato beradi)
    model = model.float()
    return model


# ─────────────────────────────────────────────────────────────
# AUDIO → STFT SPEKTROGRAMMA
# ─────────────────────────────────────────────────────────────
def audio_to_spectrogram(wav_path: Path) -> tuple[np.ndarray, int]:
    """
    WAV faylni STFT spektrogrammaga aylantirish.
    TweetyBERT maqolasining preprocessing pipeline'ini takrorlaydi.

    Args:
        wav_path: WAV fayl yo'li

    Returns:
        (spec_db, samplerate) — dB shkalaдagi STFT, namunalash chastotasi
    """
    print(f"  🎵  Audio yuklanmoqda: {wav_path.name}")

    # Audio o'qish
    try:
        data, sr = sf.read(str(wav_path), dtype="int16")
    except Exception as e:
        raise RuntimeError(f"Audio o'qishda xato: {e}") from e

    # Ko'p kanalli bo'lsa, birinchi kanal
    if data.ndim > 1:
        data = data[:, 0]

    print(f"     Davomiyligi: {len(data)/sr:.2f}s | SR: {sr} Hz | O'lcham: {data.shape}")

    # Yuqori o'tkazgich filtr (maqola: 5-order elliptic, 500 Hz)
    try:
        b, a = ellip(5, 0.2, 40, 500 / (sr / 2), "high")
        data = filtfilt(b, a, data.astype(float))
    except Exception as e:
        warnings.warn(f"Filtr xato: {e} — filtrsiz davom etilmoqda")
        data = data.astype(float)

    # STFT hisoblash
    Sxx = librosa.stft(data, n_fft=NFFT, hop_length=HOP_LEN, window="hann")
    spec_db = librosa.amplitude_to_db(np.abs(Sxx), ref=np.max)

    print(f"     Spektrogramma o'lchami: {spec_db.shape} (chastota_bin × vaqt)")
    return spec_db, sr


# ─────────────────────────────────────────────────────────────
# SPEKTROGRAMMA → MODEL INPUT TENSOR
# ─────────────────────────────────────────────────────────────
def spec_to_tensor(spec_db: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    dB spektrogrammani TweetyBERT kutadigan tensor formatiga aylantirish.

    data_class.py: spectrogram = spectrogram[20:216]  → 196 chastota bini
    Format: (batch=1, 1, freq=196, time) — CNN kirishi uchun

    Args:
        spec_db: (freq, time) — STFT dB
        device:  Torch qurilmasi

    Returns:
        (1, 1, 196, time) tensor
    """
    # Chastota binlarini qirqish: [20:216] = 196 bin (maqola talabi)
    freq_bins, time_bins = spec_db.shape

    # Agar freq_bins 216 dan kichik bo'lsa, 0 bilan to'ldirish
    if freq_bins < 216:
        pad = np.zeros((216 - freq_bins, time_bins))
        spec_db = np.vstack([spec_db, pad])

    spec_cropped = spec_db[20:216, :]  # (196, time)

    # Z-score normalizatsiya
    mean = spec_cropped.mean()
    std  = spec_cropped.std() + 1e-8
    spec_norm = (spec_cropped - mean) / std

    # (time, freq) → tensor → (1, 1, freq, time)
    tensor = torch.from_numpy(spec_norm).float()   # (196, time)
    tensor = tensor.unsqueeze(0).unsqueeze(0)      # (1, 1, 196, time)
    tensor = tensor.permute(0, 1, 2, 3)            # unchanged, already correct

    return tensor.to(device)


# ─────────────────────────────────────────────────────────────
# INFERENCE — EMBEDDING CHIQARISH
# ─────────────────────────────────────────────────────────────
def extract_embeddings(
    model: TweetyBERT,
    spec_tensor: torch.Tensor,
    device: torch.device,
    context_len: int = 1000,
) -> np.ndarray:
    """
    TweetyBERT encoder orqali embedding chiqarish.
    Uzun spektrogrammalar kontekst oynalarga bo'linadi.

    Args:
        model:       TweetyBERT modeli (eval rejimida)
        spec_tensor: (1, 1, 196, time) tensor
        device:      Torch qurilmasi
        context_len: Oyna uzunligi (vaqt binlarida)

    Returns:
        (total_time, d_model) — numpy embedding massivi
    """
    _, _, freq_bins, total_time = spec_tensor.shape
    all_embeddings = []

    # Kontekst oynalarga bo'lish
    for start in range(0, total_time, context_len):
        end   = min(start + context_len, total_time)
        chunk = spec_tensor[:, :, :, start:end]   # (1, 1, 196, chunk_len)

        # Model CNN kirishi: (batch, 1, freq, time) yoki (batch, freq, time)?
        # inference_forward: x = self.feature_extractor_forward(x)
        # feature_extractor_forward: conv1 kutadi (batch, 1, freq, time) → MaxPool2d(2,1) → freq/16
        # Keyin x.flatten(1,2): (batch, feat*freq_reduced, time)
        # Keyin permute(0,2,1): (batch, time, feat)

        with torch.no_grad():
            try:
                # inference_forward chiqaradi: (x_reconstructed, all_outputs)
                _, layer_outputs = model.inference_forward(chunk)
                # Oxirgi transformer qatlami chiqishi: 'feed_forward_output'
                emb = layer_outputs[-1]["feed_forward_output"]  # (1, time, d_model)
                emb = emb.squeeze(0).cpu().float().numpy()       # (time, d_model)
                all_embeddings.append(emb)
            except Exception as e:
                print(f"    ⚠️  Oynada xato (start={start}): {e}")
                continue

    if not all_embeddings:
        raise RuntimeError("Hech qanday embedding chiqarilmadi!")

    embeddings = np.concatenate(all_embeddings, axis=0)  # (total_time, d_model)
    return embeddings


# ─────────────────────────────────────────────────────────────
# BO'G'IN SEGMENTATSIYA
# ─────────────────────────────────────────────────────────────
def detect_syllables(wav_path: Path, sr: int) -> list[dict]:
    """
    librosa onset detection yordamida bo'g'in (syllable) boshlanish nuqtalarini aniqlash.

    Args:
        wav_path: WAV fayl
        sr:       Namunalash chastotasi

    Returns:
        [{"onset_time": float, "duration_sec": float}, ...]
    """
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

    # Onset detection
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=HOP_LEN,
        backtrack=True, units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LEN)

    # Davomiylik hisoblash (keyingi bo'g'in boshlanishigacha)
    total_dur = len(y) / sr
    syllables = []
    for i, t in enumerate(onset_times):
        if i + 1 < len(onset_times):
            dur = float(onset_times[i + 1] - t)
        else:
            dur = float(total_dur - t)
        syllables.append({"onset_time": float(t), "duration_sec": dur})

    return syllables


# ─────────────────────────────────────────────────────────────
# SPEKTROGRAMMA RASM SAQLASH
# ─────────────────────────────────────────────────────────────
def save_spectrogram_image(spec_db: np.ndarray, sr: int, wav_name: str) -> Path:
    """
    Spektrogrammani PNG rasmiga aylantirish va saqlash.

    Args:
        spec_db:  (freq, time) — dB spektrogramma
        sr:       Namunalash chastotasi
        wav_name: Fayl nomi (kengaytmasiz)

    Returns:
        Saqlangan fayl yo'li
    """
    OUT_SPEC.mkdir(parents=True, exist_ok=True)
    out_path = OUT_SPEC / f"{wav_name}.png"

    fig, ax = plt.subplots(figsize=(14, 4), dpi=120)
    img = librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=HOP_LEN,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"🐦 Kanareyka Spektrogrammasi — {wav_name}", fontsize=12, pad=10)
    ax.set_xlabel("Vaqt (soniya)")
    ax.set_ylabel("Chastota (Hz)")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ─────────────────────────────────────────────────────────────
# JSON EKSPORT
# ─────────────────────────────────────────────────────────────
def export_embeddings_json(
    wav_name:   str,
    duration:   float,
    device:     torch.device,
    syllables:  list[dict],
    embeddings: np.ndarray,
    sr:         int,
) -> Path:
    """
    Bo'g'in embedding'larini JSON formatida saqlash.

    Args:
        wav_name:   Fayl nomi
        duration:   Audio davomiyligi (s)
        device:     Ishlatilgan qurilma
        syllables:  Bo'g'inlar ro'yxati
        embeddings: (time, d_model) embedding massivi
        sr:         Namunalash chastotasi

    Returns:
        Saqlangan JSON fayl yo'li
    """
    OUT_EMB.mkdir(parents=True, exist_ok=True)

    # Har bir bo'g'in uchun mos embedding topish (vaqt indeksi orqali)
    sec_per_bin = HOP_LEN / sr
    enriched_syllables = []

    for syl in syllables:
        onset_bin = int(syl["onset_time"] / sec_per_bin)
        onset_bin = min(onset_bin, len(embeddings) - 1)

        # Embedding: onset vaqtidagi vektor
        emb_vec = embeddings[onset_bin].tolist()

        enriched_syllables.append({
            "onset_time":   round(syl["onset_time"],   4),
            "duration_sec": round(syl["duration_sec"], 4),
            "embedding_dim": len(emb_vec),
            "embedding":    [round(v, 6) for v in emb_vec],
        })

    result = {
        "audio_file":    f"{wav_name}.wav",
        "duration_sec":  round(duration, 3),
        "samplerate":    sr,
        "device":        str(device),
        "model":         "TweetyBERT",
        "embedding_dim": embeddings.shape[1] if embeddings.ndim > 1 else 0,
        "total_time_bins": int(embeddings.shape[0]),
        "num_syllables": len(enriched_syllables),
        "syllables":     enriched_syllables,
    }

    out_path = OUT_EMB / f"{wav_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return out_path


# ─────────────────────────────────────────────────────────────
# ASOSIY FUNKSIYA
# ─────────────────────────────────────────────────────────────
def process_audio_file(
    wav_path:        Path,
    model:           TweetyBERT,
    device:          torch.device,
    save_spec_image: bool = True,
) -> dict:
    """
    Bitta WAV faylni tahlil qilish.

    Args:
        wav_path:        WAV fayl yo'li
        model:           TweetyBERT modeli
        device:          Torch qurilmasi
        save_spec_image: Spektrogramma rasmini saqlash

    Returns:
        Natija haqida ma'lumot dict
    """
    wav_name = wav_path.stem
    print(f"\n{'─'*55}")
    print(f"🔬  Tahlil: {wav_path.name}")
    print(f"{'─'*55}")

    # 1. Audio → Spektrogramma
    spec_db, sr = audio_to_spectrogram(wav_path)
    duration     = spec_db.shape[1] * HOP_LEN / sr

    # 2. Spektrogramma rasmini saqlash
    if save_spec_image:
        spec_img = save_spectrogram_image(spec_db, sr, wav_name)
        print(f"  🖼️   Spektrogramma saqlandi: {spec_img}")

    # 3. Tensor yasash
    spec_tensor = spec_to_tensor(spec_db, device)
    print(f"  📐  Tensor o'lchami: {tuple(spec_tensor.shape)}")

    # 4. Embedding chiqarish
    print("  🔄  Embedding chiqarilmoqda...")
    embeddings = extract_embeddings(model, spec_tensor, device)
    print(f"  ✅  Embedding o'lchami: {embeddings.shape}  (vaqt_bin × d_model)")

    # 5. Bo'g'in segmentatsiya
    print("  🎼  Bo'g'inlar aniqlanmoqda...")
    syllables = detect_syllables(wav_path, sr)
    print(f"  🔢  Topilgan bo'g'inlar: {len(syllables)} ta")

    # 6. JSON eksport
    json_path = export_embeddings_json(wav_name, duration, device, syllables, embeddings, sr)
    print(f"  💾  JSON saqlandi: {json_path}")

    return {
        "wav":         str(wav_path),
        "json":        str(json_path),
        "spec_img":    str(spec_img) if save_spec_image else None,
        "syllables":   len(syllables),
        "emb_shape":   embeddings.shape,
        "duration":    duration,
    }


# ─────────────────────────────────────────────────────────────
# COMMAND LINE INTERFACE
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Kanareyka audio fayllarini TweetyBERT bilan tahlil qilish",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio", type=str, required=True,
        help="WAV fayl yo'li yoki papka yo'li (ichidagi barcha .wav)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Pretrained TweetyBERT checkpoint (.pt fayl)"
    )
    parser.add_argument(
        "--no-spec", action="store_true",
        help="Spektrogramma rasmlarini saqlama"
    )
    parser.add_argument(
        "--context", type=int, default=1000,
        help="Inference oyna uzunligi (vaqt binlarida)"
    )
    args = parser.parse_args()

    # ── Papkalarni tayyorlash ──────────────────────────────
    OUT_SPEC.mkdir(parents=True, exist_ok=True)
    OUT_EMB.mkdir(parents=True, exist_ok=True)

    # ── WAV fayllar ro'yxati ───────────────────────────────
    audio_path = Path(args.audio)
    if audio_path.is_dir():
        wav_files = sorted(audio_path.glob("*.wav"))
        if not wav_files:
            sys.exit(f"❌  {audio_path} papkasida .wav fayl topilmadi!")
        print(f"📁  Papkadan {len(wav_files)} ta WAV fayl topildi: {audio_path}")
    elif audio_path.is_file() and audio_path.suffix.lower() == ".wav":
        wav_files = [audio_path]
    else:
        sys.exit(f"❌  Fayl yoki papka topilmadi: {audio_path}")

    # ── Qurilma ───────────────────────────────────────────
    device = select_device()

    # ── Model ─────────────────────────────────────────────
    model = load_model(device, args.checkpoint)

    # ── Har bir faylni tahlil qilish ──────────────────────
    results = []
    errors  = []

    for i, wav_path in enumerate(wav_files, 1):
        print(f"\n[{i}/{len(wav_files)}]", end="")
        try:
            res = process_audio_file(
                wav_path        = wav_path,
                model           = model,
                device          = device,
                save_spec_image = not args.no_spec,
            )
            results.append(res)
        except Exception as e:
            print(f"\n  ❌  Xato: {e}")
            errors.append({"file": str(wav_path), "error": str(e)})

    # ── Yakuniy hisobot ────────────────────────────────────
    print(f"\n{'═'*55}")
    print("📊  YAKUNIY HISOBOT")
    print(f"{'═'*55}")
    print(f"  ✅  Muvaffaqiyatli: {len(results)}/{len(wav_files)} ta fayl")
    if errors:
        print(f"  ❌  Xatolar:        {len(errors)} ta fayl")
        for err in errors:
            print(f"       • {Path(err['file']).name}: {err['error']}")
    print(f"\n  📂  Spektrogrammalar: {OUT_SPEC.resolve()}")
    print(f"  📂  Embedding'lar:    {OUT_EMB.resolve()}")

    for r in results:
        print(f"\n  🐦  {Path(r['wav']).name}")
        print(f"       Davomiylik:  {r['duration']:.2f}s")
        print(f"       Bo'g'inlar:  {r['syllables']} ta")
        print(f"       Embedding:   {r['emb_shape']}")

    print(f"\n{'═'*55}\n")


if __name__ == "__main__":
    main()
