"""
Noctury TTS — RunPod Jobs Handler (Autonome)
=============================================
Handler asynchrone pour les générations longues (épisodes > 90s).
Version autonome : charge les modèles directement sans importer server.py.

Utilisation côté client :
  POST https://api.runpod.ai/v2/{endpoint_id}/run
  Body: {
    "input": {
      "action": "generate_episode_design",
      "text": "...",
      "instruct": "...",
      "language": "French",
      "chunk_max_chars": 1600
    }
  }

  GET https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}
  → {"status": "COMPLETED", "output": {"audio_b64": "...", "duration": 360.5}}
"""

import os
import sys
import time
import base64
import io
import hashlib
import logging
import uuid
import numpy as np
import boto3
from botocore.client import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "French")

# ─── Cloudflare R2 ────────────────────────────────────────────────────────────
R2_ENDPOINT = os.getenv("R2_ENDPOINT", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET = os.getenv("R2_BUCKET", "")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "").rstrip("/")

def upload_audio_to_r2(audio_data: np.ndarray, sample_rate: int, prefix: str = "jobs") -> str:
    """Convertit l'audio en MP3 et l'uploade sur R2. Retourne l'URL publique."""
    import soundfile as sf
    from pydub import AudioSegment

    # WAV en mémoire
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_data, sample_rate, format="WAV")
    wav_buf.seek(0)

    # Conversion WAV → MP3 192kbps
    segment = AudioSegment.from_wav(wav_buf)
    mp3_buf = io.BytesIO()
    segment.export(mp3_buf, format="mp3", bitrate="192k")
    mp3_buf.seek(0)
    mp3_bytes = mp3_buf.read()

    # Upload sur R2
    filename = f"{prefix}/{uuid.uuid4().hex}.mp3"
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=filename,
        Body=mp3_bytes,
        ContentType="audio/mpeg",
    )
    url = f"{R2_PUBLIC_URL}/{filename}"
    logger.info(f"[R2] Upload terminé : {url} ({len(mp3_bytes)/1024:.0f} KB)")
    return url


MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1600"))
CROSSFADE_MS = int(os.getenv("CROSSFADE_MS", "200"))
VD_TEMPERATURE = float(os.getenv("VD_TEMPERATURE", "0.7"))
VD_TOP_K = int(os.getenv("VD_TOP_K", "30"))
VD_TOP_P = float(os.getenv("VD_TOP_P", "0.85"))
VD_REPETITION_PENALTY = float(os.getenv("VD_REPETITION_PENALTY", "1.1"))

VOLUME_MODELS = "/runpod-volume/models"
HF_HOME = os.getenv("HF_HOME", "/root/.cache/huggingface")

# ─── Chargement des modèles ───────────────────────────────────────────────────
logger.info("[Handler] Initialisation des modèles TTS...")

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[Handler] Device: {device}")

voice_design_model = None
MODELS_LOADED = False
MODELS_ERROR = None

try:
    from qwen_tts import Qwen3TTSModel

    # Chemin du modèle VoiceDesign
    candidate_vd = os.path.join(VOLUME_MODELS, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    if os.path.isdir(candidate_vd):
        vd_path = candidate_vd
        logger.info(f"[Handler] VoiceDesign depuis volume: {vd_path}")
    else:
        vd_path = os.getenv("NOCTURY_MODEL_VOICEDESIGN", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        logger.warning(f"[Handler] Volume non trouvé à {candidate_vd} — utilisation de: {vd_path}")

    voice_design_model = Qwen3TTSModel.from_pretrained(
        vd_path,
        device_map=device,
        dtype=torch.bfloat16,
    )
    MODELS_LOADED = True
    logger.info("[Handler] Modèle VoiceDesign chargé avec succès")

except Exception as e:
    MODELS_ERROR = f"{type(e).__name__}: {e}"
    logger.error(f"[Handler] Erreur chargement modèle: {MODELS_ERROR}", exc_info=True)
    MODELS_LOADED = False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def voice_seed(instruct: str) -> int:
    """Calcule un seed déterministe depuis le prompt instruct."""
    return int(hashlib.md5(instruct.encode()).hexdigest(), 16) % (2**31)


def set_voice_seed(instruct: str) -> None:
    """Fixe le seed aléatoire pour la cohérence inter-chunks."""
    seed = voice_seed(instruct)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enrich_french_instruct(instruct: str) -> str:
    """Enrichit le prompt instruct pour forcer l'accent français."""
    lower = instruct.lower()
    has_french = any(w in lower for w in ["français", "française", "parisien", "parisienne", "french"])

    if not has_french:
        prefix = "Locutrice native française, accent parisien pur, prononciation française authentique. "
        suffix = " Aucun accent anglais, voyelles et consonnes françaises natives."
        return prefix + instruct + suffix
    else:
        suffix = " Prononciation française native, voyelles françaises pures, aucun accent anglais."
        return instruct + suffix


def split_text_into_chunks(text: str, max_chars: int = 1600) -> list:
    """Découpe le texte en chunks respectant les limites de phrases."""
    import re
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            if len(sentence) > max_chars:
                # Découpe forcée sur les virgules
                parts = sentence.split(", ")
                sub = ""
                for part in parts:
                    if len(sub) + len(part) + 2 <= max_chars:
                        sub = (sub + ", " + part).strip(", ") if sub else part
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = part
                if sub:
                    current = sub
                else:
                    current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def assemble_chunks_audio(audio_chunks: list, sample_rate: int, crossfade_ms: int = 200) -> np.ndarray:
    """Assemble les chunks audio avec crossfade."""
    if not audio_chunks:
        return np.array([], dtype=np.float32)
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = audio_chunks[0]

    for chunk in audio_chunks[1:]:
        if crossfade_samples > 0 and len(result) >= crossfade_samples and len(chunk) >= crossfade_samples:
            fade_out = np.linspace(1.0, 0.0, crossfade_samples)
            fade_in = np.linspace(0.0, 1.0, crossfade_samples)
            result[-crossfade_samples:] = result[-crossfade_samples:] * fade_out + chunk[:crossfade_samples] * fade_in
            result = np.concatenate([result, chunk[crossfade_samples:]])
        else:
            result = np.concatenate([result, chunk])

    return result


def wav_to_base64(audio_data: np.ndarray, sample_rate: int) -> str:
    """Convertit un array numpy en WAV base64."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─── Actions ──────────────────────────────────────────────────────────────────

def handle_generate_episode_design(job_input: dict) -> dict:
    """Génère un épisode complet avec VoiceDesign sans limite de durée."""
    text = job_input.get("text", "")
    instruct = job_input.get("instruct", "")
    language = job_input.get("language", DEFAULT_LANGUAGE)
    chunk_max = int(job_input.get("chunk_max_chars", CHUNK_MAX_CHARS))

    if not text:
        return {"error": "Le champ 'text' est requis"}
    if not instruct:
        return {"error": "Le champ 'instruct' est requis"}

    enriched_instruct = enrich_french_instruct(instruct)
    logger.info(f"[Jobs] generate_episode_design | {len(text)} chars | instruct: {enriched_instruct[:80]}")

    chunks = split_text_into_chunks(text, max_chars=chunk_max)
    logger.info(f"[Jobs] {len(chunks)} chunks de max {chunk_max} chars")

    audio_chunks = []
    sample_rate = None
    total_start = time.time()

    for i, chunk_text in enumerate(chunks):
        chunk_start = time.time()
        logger.info(f"[Jobs] Chunk {i+1}/{len(chunks)}: '{chunk_text[:60]}...'")

        set_voice_seed(enriched_instruct)
        wavs, chunk_sr = voice_design_model.generate_voice_design(
            text=chunk_text,
            language=language,
            instruct=enriched_instruct,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=VD_TEMPERATURE,
            top_k=VD_TOP_K,
            top_p=VD_TOP_P,
            repetition_penalty=VD_REPETITION_PENALTY,
        )

        if sample_rate is None:
            sample_rate = chunk_sr

        audio_chunks.append(wavs[0])
        chunk_duration = len(wavs[0]) / chunk_sr
        logger.info(f"[Jobs] Chunk {i+1} terminé en {time.time()-chunk_start:.1f}s ({chunk_duration:.1f}s audio)")

    final_audio = assemble_chunks_audio(audio_chunks, sample_rate, CROSSFADE_MS)
    total_duration = len(final_audio) / sample_rate
    total_elapsed = time.time() - total_start

    logger.info(f"[Jobs] Assemblage terminé : {total_duration:.1f}s audio en {total_elapsed:.1f}s")

    # Upload sur R2 si configuré, sinon fallback base64 (petits audios seulement)
    audio_size_mb = len(final_audio) * 4 / (1024 * 1024)  # float32 = 4 bytes
    if R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_BUCKET:
        try:
            audio_url = upload_audio_to_r2(final_audio, sample_rate, prefix="jobs")
            return {
                "audio_url": audio_url,
                "duration": round(total_duration, 2),
                "sample_rate": sample_rate,
                "chunks": len(chunks),
                "elapsed": round(total_elapsed, 2),
                "instruct_enriched": enriched_instruct,
            }
        except Exception as e:
            logger.error(f"[R2] Erreur upload : {e} — fallback base64")

    # Fallback base64 (pour petits audios < 5 MB)
    if audio_size_mb < 5:
        audio_b64 = wav_to_base64(final_audio, sample_rate)
        return {
            "audio_b64": audio_b64,
            "duration": round(total_duration, 2),
            "sample_rate": sample_rate,
            "chunks": len(chunks),
            "elapsed": round(total_elapsed, 2),
            "instruct_enriched": enriched_instruct,
        }
    else:
        return {
            "error": f"Audio trop volumineux ({audio_size_mb:.1f} MB) et R2 non configuré — configurez R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_BUCKET, R2_PUBLIC_URL",
            "duration": round(total_duration, 2),
            "chunks": len(chunks),
            "elapsed": round(total_elapsed, 2),
        }


def handle_health() -> dict:
    """Retourne l'état de santé du worker."""
    return {
        "status": "ok" if MODELS_LOADED else "error",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": MODELS_LOADED,
        "models_error": MODELS_ERROR,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }


def handle_debug() -> dict:
    """Retourne des informations de débogage sur l'environnement."""
    import subprocess
    volume_ls = subprocess.run(
        ["ls", "-la", "/runpod-volume/models/"],
        capture_output=True, text=True
    )
    pip_qwen = subprocess.run(
        ["pip3", "show", "qwen-tts"],
        capture_output=True, text=True
    )
    return {
        "models_loaded": MODELS_LOADED,
        "models_error": MODELS_ERROR,
        "volume_ls": volume_ls.stdout or volume_ls.stderr,
        "qwen_tts_installed": pip_qwen.stdout or pip_qwen.stderr,
        "env_model_vd": os.getenv("NOCTURY_MODEL_VOICEDESIGN", "non défini"),
        "volume_path_exists": os.path.isdir("/runpod-volume/models"),
        "candidate_vd_exists": os.path.isdir("/runpod-volume/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        "python_path": sys.path,
    }


# ─── Point d'entrée RunPod Jobs ───────────────────────────────────────────────

def handler(job: dict) -> dict:
    """
    Point d'entrée principal du RunPod Jobs handler.
    RunPod appelle cette fonction avec {"id": "...", "input": {...}}
    """
    job_input = job.get("input", {})
    action = job_input.get("action", "generate_episode_design")

    logger.info(f"[Jobs] Received job | action={action} | id={job.get('id', 'unknown')}")

    if not MODELS_LOADED:
        return {"error": f"Modèles non chargés — {MODELS_ERROR or 'erreur inconnue au démarrage du worker'}"}

    try:
        if action == "generate_episode_design":
            return handle_generate_episode_design(job_input)
        elif action == "health":
            return handle_health()
        elif action == "debug":
            return handle_debug()
        else:
            return {"error": f"Action inconnue : '{action}'. Disponibles : generate_episode_design, health, debug"}

    except Exception as e:
        logger.error(f"[Jobs] Erreur : {e}", exc_info=True)
        return {"error": str(e)}


# ─── Démarrage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import runpod
    logger.info("[Jobs] Démarrage du worker RunPod Jobs (mode autonome)...")
    runpod.serverless.start({"handler": handler})
