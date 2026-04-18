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
      "chunk_max_chars": 800
    }
  }

  GET https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}
  → {"status": "COMPLETED", "output": {"audio_url": "https://...", "duration": 360.5}}

Stratégie cohérence de voix :
  - Chunk 1 : generate_voice_design (instruct texte) → extrait 3s de référence
  - Chunks 2+ : generate_voice_clone (voice_clone_prompt) → voix identique garantie
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


# ─── Paramètres de génération ─────────────────────────────────────────────────
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
# 800 chars comme server.py — meilleure qualité et plus rapide par chunk
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "800"))
CROSSFADE_MS = int(os.getenv("CROSSFADE_MS", "200"))
VD_TEMPERATURE = float(os.getenv("VD_TEMPERATURE", "0.7"))
VD_TOP_K = int(os.getenv("VD_TOP_K", "30"))
VD_TOP_P = float(os.getenv("VD_TOP_P", "0.85"))
VD_REPETITION_PENALTY = float(os.getenv("VD_REPETITION_PENALTY", "1.1"))
# Durée de l'extrait de référence pour le voice cloning inter-chunks (en secondes)
VOICE_REF_DURATION_S = float(os.getenv("VOICE_REF_DURATION_S", "5.0"))

VOLUME_MODELS = "/runpod-volume/models"
HF_HOME = os.getenv("HF_HOME", "/root/.cache/huggingface")

# ─── Chargement des modèles ───────────────────────────────────────────────────
logger.info("[Handler] Initialisation des modèles TTS...")

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[Handler] Device: {device}")

voice_design_model = None
base_model = None  # Modèle Base pour create_voice_clone_prompt
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

    # Charger le modèle Base pour create_voice_clone_prompt (méthode Base-only)
    candidate_base = os.path.join(VOLUME_MODELS, "Qwen3-TTS-12Hz-1.7B-Base")
    if os.path.isdir(candidate_base):
        base_path = candidate_base
        logger.info(f"[Handler] Base depuis volume: {base_path}")
    else:
        base_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        logger.warning(f"[Handler] Base non trouvé à {candidate_base} — utilisation de: {base_path}")

    base_model = Qwen3TTSModel.from_pretrained(
        base_path,
        device_map=device,
        dtype=torch.bfloat16,
    )
    logger.info("[Handler] Modèle Base chargé avec succès (pour voice cloning)")

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


def split_text_into_chunks(text: str, max_chars: int = 800) -> list:
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


def extract_voice_reference(audio_data: np.ndarray, sample_rate: int, duration_s: float = 5.0) -> np.ndarray:
    """Extrait les N premières secondes d'audio comme référence de voix."""
    n_samples = int(sample_rate * duration_s)
    return audio_data[:n_samples]


def transcribe_audio(audio_data: np.ndarray, sample_rate: int) -> str:
    """Transcrit l'audio en texte pour le voice cloning (utilise whisper si disponible)."""
    try:
        import whisper
        import soundfile as sf
        import tempfile
        # Sauvegarder en WAV temporaire
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, sample_rate)
            tmp_path = f.name
        whisper_model = whisper.load_model("base", device=device)
        result = whisper_model.transcribe(tmp_path, language="fr")
        os.unlink(tmp_path)
        return result.get("text", "").strip()
    except Exception as e:
        logger.warning(f"[Handler] Transcription échouée: {e} — utilisation d'un texte vide")
        return ""


# ─── Actions ──────────────────────────────────────────────────────────────────

def handle_generate_episode_design(job_input: dict) -> dict:
    """
    Génère un épisode complet avec VoiceDesign sans limite de durée.

    Stratégie cohérence de voix :
    - Chunk 1 : generate_voice_design (instruct texte)
    - Extrait 5s de référence depuis le chunk 1
    - Chunks 2+ : generate_voice_clone avec voice_clone_prompt → voix identique
    """
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
    voice_clone_prompt = None  # Sera créé après le chunk 1

    for i, chunk_text in enumerate(chunks):
        chunk_start = time.time()
        logger.info(f"[Jobs] Chunk {i+1}/{len(chunks)}: '{chunk_text[:60]}...'")

        if i == 0:
            # Chunk 1 : VoiceDesign depuis le prompt instruct
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
            chunk_audio = wavs[0]
            sample_rate = chunk_sr

            # Créer le voice_clone_prompt depuis les 5 premières secondes du chunk 1
            try:
                ref_audio = extract_voice_reference(chunk_audio, chunk_sr, VOICE_REF_DURATION_S)
                ref_text = transcribe_audio(ref_audio, chunk_sr)
                logger.info(f"[Jobs] Référence vocale extraite ({VOICE_REF_DURATION_S}s) | transcription: '{ref_text[:60]}'")
                # create_voice_clone_prompt est Base-only — utiliser base_model
                clone_model = base_model if base_model is not None else voice_design_model
                voice_clone_prompt = clone_model.create_voice_clone_prompt(
                    ref_audio=(ref_audio, chunk_sr),
                    ref_text=ref_text if ref_text else None,
                    x_vector_only_mode=(ref_text == ""),
                )
                logger.info("[Jobs] voice_clone_prompt créé — chunks suivants en mode clone")
            except Exception as e:
                logger.warning(f"[Jobs] Impossible de créer voice_clone_prompt: {e} — fallback VoiceDesign pour tous les chunks")
                voice_clone_prompt = None

        else:
            # Chunks 2+ : VoiceClone pour cohérence de ton
            # generate_voice_clone est Base-only — utiliser base_model
            if voice_clone_prompt is not None and base_model is not None:
                torch.manual_seed(42)
                wavs, chunk_sr = base_model.generate_voice_clone(
                    text=chunk_text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                chunk_audio = wavs[0]
            else:
                # Fallback : VoiceDesign avec seed fixe
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
                chunk_audio = wavs[0]

        audio_chunks.append(chunk_audio)
        chunk_duration = len(chunk_audio) / chunk_sr
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
                "voice_clone_used": voice_clone_prompt is not None,
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
            "voice_clone_used": voice_clone_prompt is not None,
        }
    else:
        return {
            "error": f"Audio trop volumineux ({audio_size_mb:.1f} MB) et R2 non configuré",
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
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "voice_ref_duration_s": VOICE_REF_DURATION_S,
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
        logger.error(f"[Jobs] Erreur inattendue: {e}", exc_info=True)
        return {"error": f"{type(e).__name__}: {str(e)}"}


import runpod
runpod.serverless.start({"handler": handler})
