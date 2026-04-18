"""
Noctury TTS — RunPod Jobs Handler
==================================
Handler asynchrone pour les générations longues (épisodes > 90s).

Architecture :
  - Le serveur FastAPI (server.py) reste actif pour les routes courtes
    (health, synthesize_speech, synthesize_speech_design, etc.)
  - Ce handler est appelé par RunPod via POST /run (asynchrone) ou
    POST /runsync (synchrone, mais sans timeout de proxy)

Utilisation côté client (Railway) :
  POST https://{endpoint_id}-{job_id}.proxy.runpod.net/run
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

Actions supportées :
  - generate_episode_design : génération VoiceDesign texte long
  - generate_episode        : génération Voice Clone texte long
  - health                  : vérification santé du worker
"""

import os
import sys
import time
import base64
import io
import wave
import logging
import hashlib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── Import des modèles depuis server.py ─────────────────────────────────────
# On importe les modèles déjà chargés en mémoire par server.py pour éviter
# de les recharger deux fois (économie de VRAM et de temps de démarrage)
sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    from server import (
        voice_design_model,
        base_model,
        device,
        DEFAULT_LANGUAGE,
        CHUNK_MAX_CHARS,
        VD_TEMPERATURE,
        VD_TOP_K,
        VD_TOP_P,
        VD_REPETITION_PENALTY,
        MAX_NEW_TOKENS,
        split_text_into_chunks,
        assemble_chunks_audio,
        enrich_french_instruct,
        set_voice_seed,
        get_or_create_voice_cache,
        generate_speech_with_prompt,
        resources_dir,
    )
    import soundfile as sf
    MODELS_LOADED = True
    logging.info("Handler: modèles importés depuis server.py avec succès")
except Exception as e:
    MODELS_LOADED = False
    logging.error(f"Handler: impossible d'importer server.py — {e}")


def wav_to_base64(audio_data: np.ndarray, sample_rate: int) -> str:
    """Convertit un array numpy en WAV base64."""
    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def handle_generate_episode_design(job_input: dict) -> dict:
    """
    Génère un épisode complet avec VoiceDesign sans limite de durée.
    Chunk_max_chars peut être monté jusqu'à ~3000 chars sans timeout
    car on est en mode Jobs asynchrone.
    """
    text = job_input.get("text", "")
    instruct = job_input.get("instruct", "")
    language = job_input.get("language", DEFAULT_LANGUAGE)
    chunk_max = job_input.get("chunk_max_chars", 1600)  # 2× plus grand qu'en HTTP sync

    if not text:
        return {"error": "Le champ 'text' est requis"}
    if not instruct:
        return {"error": "Le champ 'instruct' est requis"}

    enriched_instruct = enrich_french_instruct(instruct)
    logging.info(f"[Jobs] generate_episode_design | {len(text)} chars | instruct: {enriched_instruct[:80]}")

    chunks = split_text_into_chunks(text, max_chars=chunk_max)
    logging.info(f"[Jobs] {len(chunks)} chunks de max {chunk_max} chars")

    audio_chunks = []
    sample_rate = None
    total_start = time.time()

    for i, chunk_text in enumerate(chunks):
        chunk_start = time.time()
        logging.info(f"[Jobs] Chunk {i+1}/{len(chunks)}: '{chunk_text[:60]}...'")

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
        logging.info(f"[Jobs] Chunk {i+1} terminé en {time.time()-chunk_start:.1f}s ({chunk_duration:.1f}s audio)")

    final_audio = assemble_chunks_audio(audio_chunks, sample_rate)
    total_duration = len(final_audio) / sample_rate
    total_elapsed = time.time() - total_start

    logging.info(f"[Jobs] Assemblage terminé : {total_duration:.1f}s audio en {total_elapsed:.1f}s")

    audio_b64 = wav_to_base64(final_audio, sample_rate)

    return {
        "audio_b64": audio_b64,
        "duration": round(total_duration, 2),
        "sample_rate": sample_rate,
        "chunks": len(chunks),
        "elapsed": round(total_elapsed, 2),
        "instruct_enriched": enriched_instruct,
    }


def handle_generate_episode(job_input: dict) -> dict:
    """
    Génère un épisode complet avec Voice Clone (voix de référence) sans limite de durée.
    """
    text = job_input.get("text", "")
    voice = job_input.get("voice", "")
    language = job_input.get("language", DEFAULT_LANGUAGE)
    speed = job_input.get("speed", 0.92)
    chunk_max = job_input.get("chunk_max_chars", 1600)

    if not text:
        return {"error": "Le champ 'text' est requis"}
    if not voice:
        return {"error": "Le champ 'voice' est requis"}

    logging.info(f"[Jobs] generate_episode | voice={voice} | {len(text)} chars")

    # Trouver le fichier voix
    matching_files = [
        f for f in os.listdir(resources_dir)
        if f.startswith(voice) and f.lower().endswith(".wav")
    ]
    if not matching_files:
        return {"error": f"Voix '{voice}' introuvable dans {resources_dir}"}

    reference_file = f"{resources_dir}/{matching_files[0]}"
    cache_data = get_or_create_voice_cache(voice, reference_file)

    chunks = split_text_into_chunks(text, max_chars=chunk_max)
    logging.info(f"[Jobs] {len(chunks)} chunks")

    audio_chunks = []
    sample_rate = None
    total_start = time.time()

    for i, chunk_text in enumerate(chunks):
        chunk_start = time.time()
        audio_data, chunk_sr = generate_speech_with_prompt(
            chunk_text, cache_data["prompt"], speed=speed, language=language
        )
        if sample_rate is None:
            sample_rate = chunk_sr
        audio_chunks.append(audio_data)
        logging.info(f"[Jobs] Chunk {i+1}/{len(chunks)} en {time.time()-chunk_start:.1f}s")

    final_audio = assemble_chunks_audio(audio_chunks, sample_rate)
    total_duration = len(final_audio) / sample_rate
    total_elapsed = time.time() - total_start

    audio_b64 = wav_to_base64(final_audio, sample_rate)

    return {
        "audio_b64": audio_b64,
        "duration": round(total_duration, 2),
        "sample_rate": sample_rate,
        "chunks": len(chunks),
        "elapsed": round(total_elapsed, 2),
    }


def handler(job: dict) -> dict:
    """
    Point d'entrée principal du RunPod Jobs handler.

    RunPod appelle cette fonction avec :
    {
      "id": "job-uuid",
      "input": {
        "action": "generate_episode_design",
        ...
      }
    }

    La fonction retourne un dict qui sera stocké dans output du job.
    """
    job_input = job.get("input", {})
    action = job_input.get("action", "generate_episode_design")

    logging.info(f"[Jobs] Received job | action={action} | id={job.get('id', 'unknown')}")

    if not MODELS_LOADED:
        return {"error": "Modèles non chargés — le serveur FastAPI n'est pas encore prêt"}

    try:
        if action == "generate_episode_design":
            return handle_generate_episode_design(job_input)
        elif action == "generate_episode":
            return handle_generate_episode(job_input)
        elif action == "health":
            return {
                "status": "ok",
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "models_loaded": MODELS_LOADED,
            }
        else:
            return {"error": f"Action inconnue : '{action}'. Actions disponibles : generate_episode_design, generate_episode, health"}

    except Exception as e:
        logging.error(f"[Jobs] Erreur dans handler : {e}", exc_info=True)
        return {"error": str(e)}


# ─── Démarrage du worker RunPod Jobs ─────────────────────────────────────────
if __name__ == "__main__":
    import runpod
    logging.info("[Jobs] Démarrage du worker RunPod Jobs...")
    runpod.serverless.start({"handler": handler})
