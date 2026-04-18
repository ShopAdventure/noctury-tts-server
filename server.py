"""
Noctury TTS Server — Qwen3-TTS Self-Hosted
FastAPI server for text-to-speech with voice cloning, chunking intelligent,
and full French language support. Optimized for RunPod Serverless deployment.
"""

import os
import re
import time
import torch
import numpy as np
import soundfile as sf
import io
import magic
import logging
import json
import base64
import asyncio
from typing import Optional, List, AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment, silence

logging.basicConfig(level=logging.INFO)

# ─── Configuration ───────────────────────────────────────────────────────────
API_KEY = os.environ.get("NOCTURY_TTS_API_KEY", "")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "French")
DEFAULT_SPEED = float(os.environ.get("DEFAULT_SPEED", "0.92"))
CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "800"))
CROSSFADE_MS = int(os.environ.get("CROSSFADE_MS", "200"))

app = FastAPI(
    title="Noctury TTS Server",
    description="Serveur TTS auto-hébergé pour Noctury, basé sur Qwen3-TTS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ─── Model Loading ───────────────────────────────────────────────────────────
from qwen_tts import Qwen3TTSModel

# Prioritize network volume models for fast cold start
_VOLUME_MODELS = "/runpod-volume/models"
_BASE_MODEL_PATH = os.environ.get(
    "NOCTURY_MODEL_BASE",
    os.path.join(_VOLUME_MODELS, "Qwen3-TTS-12Hz-1.7B-Base")
    if os.path.isdir(os.path.join(_VOLUME_MODELS, "Qwen3-TTS-12Hz-1.7B-Base"))
    else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)
_VOICEDESIGN_MODEL_PATH = os.environ.get(
    "NOCTURY_MODEL_VOICEDESIGN",
    os.path.join(_VOLUME_MODELS, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    if os.path.isdir(os.path.join(_VOLUME_MODELS, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
    else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)

logging.info(f"Loading Qwen3-TTS Base model from: {_BASE_MODEL_PATH}")
model = Qwen3TTSModel.from_pretrained(
    _BASE_MODEL_PATH,
    device_map=device,
    dtype=torch.bfloat16,
)
logging.info("Qwen3-TTS Base model loaded successfully")

logging.info(f"Loading Qwen3-TTS VoiceDesign model from: {_VOICEDESIGN_MODEL_PATH}")
voice_design_model = Qwen3TTSModel.from_pretrained(
    _VOICEDESIGN_MODEL_PATH,
    device_map=device,
    dtype=torch.bfloat16,
)
logging.info("Qwen3-TTS VoiceDesign model loaded successfully")

import whisper

logging.info("Loading Whisper model for transcription...")
whisper_model = whisper.load_model("base", device=device)
logging.info("Whisper model loaded successfully")

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

resources_dir = "resources"
os.makedirs(resources_dir, exist_ok=True)

# Voice cache: {voice_name: {"processed_audio": path, "ref_text": str, "prompt": object}}
voice_cache = {}


# ─── Authentication ──────────────────────────────────────────────────────────
def verify_api_key(x_noctury_key: Optional[str] = Header(None)):
    """Verify API key if one is configured (reads from X-Noctury-Key header)."""
    if not API_KEY:
        return True
    if not x_noctury_key:
        raise HTTPException(status_code=401, detail="Missing X-Noctury-Key header")
    if x_noctury_key.strip() != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


# ─── Audio Utilities ─────────────────────────────────────────────────────────
def convert_to_wav(input_path: str, output_path: str):
    """Convert any audio format to WAV using pydub."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(24000)
    audio.export(output_path, format="wav")


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper."""
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()


def detect_leading_silence(audio, silence_threshold=-42, chunk_size=10):
    """Detect silence at the beginning of the audio."""
    trim_ms = 0
    while audio[trim_ms : trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(audio):
        trim_ms += chunk_size
    return trim_ms


def remove_silence_edges(audio, silence_threshold=-42):
    """Remove silence from the beginning and end of the audio."""
    start_trim = detect_leading_silence(audio, silence_threshold)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold)
    duration = len(audio)
    return audio[start_trim : duration - end_trim]


def apply_speed(audio_data: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Apply speed adjustment to audio using time stretching."""
    if speed == 1.0:
        return audio_data
    try:
        import librosa
        return librosa.effects.time_stretch(audio_data, rate=speed)
    except ImportError:
        logging.warning("librosa not installed, speed adjustment not available")
        return audio_data


# ─── Chunking Intelligent ────────────────────────────────────────────────────
def split_text_into_chunks(text: str, max_chars: int = None) -> List[str]:
    """
    Découpe intelligemment le texte en chunks respectant les limites de phrases.
    Chaque chunk fait au maximum max_chars caractères et se termine par une ponctuation.
    """
    if max_chars is None:
        max_chars = CHUNK_MAX_CHARS

    # Nettoyer le texte
    text = text.strip()
    if not text:
        return []

    # Si le texte est assez court, le retourner tel quel
    if len(text) <= max_chars:
        return [text]

    chunks = []

    # Découper d'abord par paragraphes (double saut de ligne)
    paragraphs = re.split(r"\n\s*\n", text)

    current_chunk = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Si le paragraphe entier tient dans le chunk courant
        if len(current_chunk) + len(paragraph) + 1 <= max_chars:
            if current_chunk:
                current_chunk += " " + paragraph
            else:
                current_chunk = paragraph
            continue

        # Si le chunk courant n'est pas vide, le sauvegarder
        if current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        # Si le paragraphe est trop long, le découper par phrases
        if len(paragraph) > max_chars:
            sentences = re.split(r"(?<=[.!?…])\s+", paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

                    # Si une seule phrase dépasse max_chars, la découper aux virgules
                    if len(current_chunk) > max_chars:
                        sub_parts = re.split(r"(?<=[,;:])\s+", current_chunk)
                        current_chunk = ""
                        for part in sub_parts:
                            if len(current_chunk) + len(part) + 1 <= max_chars:
                                if current_chunk:
                                    current_chunk += " " + part
                                else:
                                    current_chunk = part
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = part
        else:
            current_chunk = paragraph

    # Ajouter le dernier chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # S'assurer que chaque chunk se termine par une ponctuation (pour le pacing)
    final_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and not chunk[-1] in ".!?…":
            chunk += "."
        final_chunks.append(chunk)

    return final_chunks


# ─── Voice Processing ────────────────────────────────────────────────────────
def process_reference_audio(reference_file: str) -> tuple:
    """Process reference audio: clip to max 15s and transcribe."""
    temp_short_ref = f"{output_dir}/temp_short_ref.wav"
    aseg = AudioSegment.from_file(reference_file)

    # Try to find long silence for clipping
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
            logging.info("Audio is over 15s, clipping short. (1)")
            break
        non_silent_wave += non_silent_seg

    # Try short silence if first method failed
    if len(non_silent_wave) > 15000:
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                logging.info("Audio is over 15s, clipping short. (2)")
                break
            non_silent_wave += non_silent_seg

    aseg = non_silent_wave

    # Hard clip at 15s
    if len(aseg) > 15000:
        aseg = aseg[:15000]
        logging.info("Audio is over 15s, clipping short. (3)")

    aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
    aseg.export(temp_short_ref, format="wav")

    ref_text = transcribe_audio(temp_short_ref)
    logging.info(f"Reference text transcribed from first 15s: {ref_text}")

    return temp_short_ref, ref_text


def get_or_create_voice_cache(voice: str, reference_file: str) -> dict:
    """Get cached voice data or create new cache entry."""
    global voice_cache

    if voice in voice_cache:
        logging.info(f"Using cached voice data for: {voice}")
        return voice_cache[voice]

    logging.info(f"Creating voice cache for: {voice}")

    processed_ref, ref_text = process_reference_audio(reference_file)

    ref_audio_data, ref_sr = sf.read(processed_ref)
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_audio_data, ref_sr),
        ref_text=ref_text,
    )

    voice_cache[voice] = {
        "processed_audio": processed_ref,
        "ref_text": ref_text,
        "prompt": voice_prompt,
        "audio_data": ref_audio_data,
        "sample_rate": ref_sr,
    }

    logging.info(f"Voice cache created for: {voice} (transcription: '{ref_text[:50]}...')")
    return voice_cache[voice]


# ─── Speech Generation ───────────────────────────────────────────────────────
def generate_speech_with_prompt(
    text: str,
    voice_prompt,
    speed: float = None,
    language: str = None,
    max_tokens: int = None,
) -> tuple:
    """Generate speech using cached voice clone prompt."""
    if speed is None:
        speed = DEFAULT_SPEED
    if language is None:
        language = DEFAULT_LANGUAGE
    if max_tokens is None:
        max_tokens = MAX_NEW_TOKENS

    start_time = time.time()

    torch.manual_seed(42)

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_prompt,
        max_new_tokens=max_tokens,
    )

    audio_data = wavs[0]

    if speed != 1.0:
        audio_data = apply_speed(audio_data, sr, speed)

    generation_time = time.time() - start_time
    audio_duration = len(audio_data) / sr
    logging.info(
        f"Generation completed in {generation_time:.2f}s "
        f"(audio duration: {audio_duration:.2f}s, RTF: {generation_time/audio_duration:.2f}x)"
    )

    return audio_data, sr


def generate_speech(
    text: str, ref_audio_path: str, ref_text: str, speed: float = None
) -> tuple:
    """Generate speech using Qwen3-TTS voice cloning (non-cached fallback)."""
    if speed is None:
        speed = DEFAULT_SPEED

    start_time = time.time()

    torch.manual_seed(42)

    ref_audio_data, ref_sr = sf.read(ref_audio_path)

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=DEFAULT_LANGUAGE,
        ref_audio=(ref_audio_data, ref_sr),
        ref_text=ref_text,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    audio_data = wavs[0]

    if speed != 1.0:
        audio_data = apply_speed(audio_data, sr, speed)

    generation_time = time.time() - start_time
    audio_duration = len(audio_data) / sr
    logging.info(
        f"Generation completed in {generation_time:.2f}s "
        f"(audio duration: {audio_duration:.2f}s, RTF: {generation_time/audio_duration:.2f}x)"
    )

    return audio_data, sr


def audio_to_wav_bytes(audio_data: np.ndarray, sr: int) -> io.BytesIO:
    """Convert numpy audio array to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format="WAV")
    buffer.seek(0)
    return buffer


def voice_seed(instruct: str) -> int:
    """
    Dérive un seed déterministe depuis le prompt instruct.
    Tous les chunks d'un même épisode utilisent le même seed
    → même caractéristiques de voix (timbre, ton, accent) sur tous les chunks.
    """
    import hashlib
    h = int(hashlib.md5(instruct.encode("utf-8")).hexdigest(), 16) % (2**31)
    return h


def set_voice_seed(instruct: str) -> None:
    """Fixe le seed PyTorch/CUDA dérivé du prompt instruct."""
    seed = voice_seed(instruct)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Voice seed set to {seed} (from instruct hash)")


def assemble_chunks_audio(audio_chunks: List[np.ndarray], sr: int) -> np.ndarray:
    """
    Assemble multiple audio chunks into a single audio with crossfade.
    """
    if not audio_chunks:
        return np.array([])

    if len(audio_chunks) == 1:
        return audio_chunks[0]

    crossfade_samples = int(sr * CROSSFADE_MS / 1000)

    result = audio_chunks[0]
    for i in range(1, len(audio_chunks)):
        chunk = audio_chunks[i]

        # Add a small silence between chunks if crossfade is too short
        if crossfade_samples > 0 and len(result) > crossfade_samples and len(chunk) > crossfade_samples:
            # Crossfade: fade out end of result, fade in start of chunk
            fade_out = np.linspace(1.0, 0.0, crossfade_samples)
            fade_in = np.linspace(0.0, 1.0, crossfade_samples)

            overlap = result[-crossfade_samples:] * fade_out + chunk[:crossfade_samples] * fade_in
            result = np.concatenate([result[:-crossfade_samples], overlap, chunk[crossfade_samples:]])
        else:
            # Simple concatenation with small silence gap
            silence_gap = np.zeros(int(sr * 0.3))  # 300ms silence
            result = np.concatenate([result, silence_gap, chunk])

    return result


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    """RunPod health check endpoint (required for Load Balancer)."""
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Detailed health check endpoint."""
    return {
        "status": "ok",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_new_tokens": MAX_NEW_TOKENS,
        "default_language": DEFAULT_LANGUAGE,
        "default_speed": DEFAULT_SPEED,
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "voices_loaded": list(voice_cache.keys()),
        "voice_design_model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }


@app.get("/voices")
async def list_voices(x_noctury_key: Optional[str] = Header(None)):
    """List all available voices."""
    verify_api_key(x_noctury_key)
    voices = []
    for f in os.listdir(resources_dir):
        name = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[1]
        voices.append({"name": name, "file": f, "format": ext, "cached": name in voice_cache})
    return {"voices": voices}


@app.on_event("startup")
async def startup_event():
    """Warmup the model on startup."""
    # Check for pre-loaded voice files
    voice_files = [f for f in os.listdir(resources_dir) if not f.startswith(".")]
    logging.info(f"Found {len(voice_files)} voice files in resources: {voice_files}")

    # Warmup with first available voice
    if voice_files:
        voice_name = os.path.splitext(voice_files[0])[0]
        logging.info(f"Warming up model with {voice_name}...")
        try:
            reference_file = f"{resources_dir}/{voice_files[0]}"
            if not reference_file.lower().endswith(".wav"):
                wav_path = f"{resources_dir}/{voice_name}.wav"
                convert_to_wav(reference_file, wav_path)
                reference_file = wav_path
            get_or_create_voice_cache(voice_name, reference_file)
            logging.info("Warmup complete — voice cached")
        except Exception as e:
            logging.warning(f"Warmup failed: {e}")


@app.get("/synthesize_speech_design/")
async def synthesize_speech_design(
    text: str,
    instruct: str,
    speed: Optional[float] = None,
    language: Optional[str] = None,
    x_noctury_key: Optional[str] = Header(None),
):
    """
    Synthesize speech using a text description (VoiceDesign model).
    No reference audio needed — describe the voice you want.

    Examples:
      instruct="Femme, 30 ans, voix douce et intime, accent parisien, ton narratif chaleureux"
      instruct="Femme, 25 ans, voix sensuelle et profonde, débit lent, murmure complice"
    """
    verify_api_key(x_noctury_key)
    start_time = time.time()
    try:
        lang = language or DEFAULT_LANGUAGE
        spd = speed or DEFAULT_SPEED

        logging.info(f"VoiceDesign generation | lang={lang} | instruct={instruct[:80]}")

        set_voice_seed(instruct)
        wavs, sr = voice_design_model.generate_voice_design(
            text=text,
            language=lang,
            instruct=instruct,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        audio_data = wavs[0]

        save_path = f"{output_dir}/output_voice_design.wav"
        sf.write(save_path, audio_data, sr)

        elapsed = time.time() - start_time
        result = StreamingResponse(open(save_path, "rb"), media_type="audio/wav")
        result.headers["X-Elapsed-Time"] = str(round(elapsed, 2))
        result.headers["X-Device-Used"] = device
        result.headers["X-Audio-Duration"] = str(round(len(audio_data) / sr, 2))
        result.headers["Access-Control-Allow-Origin"] = "*"
        return result

    except Exception as e:
        logging.error(f"Error in synthesize_speech_design: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class EpisodeDesignRequest(BaseModel):
    """Request body for episode generation with VoiceDesign."""
    text: str
    instruct: str
    speed: Optional[float] = None
    language: Optional[str] = None
    chunk_max_chars: Optional[int] = None
    output_format: Optional[str] = "wav"


@app.post("/generate_episode_design/")
async def generate_episode_design(
    request: EpisodeDesignRequest,
    x_noctury_key: Optional[str] = Header(None),
):
    """
    Generate a full episode from long text using VoiceDesign (no reference audio).
    The text is split into chunks, each generated with the described voice,
    then assembled into a single seamless audio file.
    """
    verify_api_key(x_noctury_key)
    total_start = time.time()
    try:
        lang = request.language or DEFAULT_LANGUAGE
        chunk_max = request.chunk_max_chars or CHUNK_MAX_CHARS

        logging.info(f"=== EPISODE DESIGN GENERATION START ===")
        logging.info(f"Instruct: {request.instruct[:80]} | Text: {len(request.text)} chars")

        chunks = split_text_into_chunks(request.text, max_chars=chunk_max)
        logging.info(f"Text split into {len(chunks)} chunks")

        audio_chunks = []
        sr = None

        for i, chunk_text in enumerate(chunks):
            chunk_start = time.time()
            logging.info(f"Chunk {i+1}/{len(chunks)}: '{chunk_text[:60]}...'")

            set_voice_seed(request.instruct)
            wavs, chunk_sr = voice_design_model.generate_voice_design(
                text=chunk_text,
                language=lang,
                instruct=request.instruct,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            if sr is None:
                sr = chunk_sr

            audio_chunks.append(wavs[0])
            logging.info(f"Chunk {i+1} done in {time.time()-chunk_start:.1f}s")

        final_audio = assemble_chunks_audio(audio_chunks, sr)
        total_duration = len(final_audio) / sr

        save_path = f"{output_dir}/episode_design_output.wav"
        sf.write(save_path, final_audio, sr)

        if request.output_format == "mp3":
            mp3_path = f"{output_dir}/episode_design_output.mp3"
            AudioSegment.from_wav(save_path).export(mp3_path, format="mp3", bitrate="192k")
            save_path = mp3_path
            media_type = "audio/mpeg"
        else:
            media_type = "audio/wav"

        total_time = time.time() - total_start
        logging.info(f"=== EPISODE DESIGN COMPLETE: {total_duration:.1f}s audio in {total_time:.1f}s ===")

        result = StreamingResponse(open(save_path, "rb"), media_type=media_type)
        result.headers["X-Total-Time"] = str(round(total_time, 2))
        result.headers["X-Audio-Duration"] = str(round(total_duration, 2))
        result.headers["X-Chunks-Count"] = str(len(chunks))
        result.headers["X-Device-Used"] = device
        result.headers["Access-Control-Allow-Origin"] = "*"
        return result

    except Exception as e:
        logging.error(f"Error in generate_episode_design: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate_episode_design_stream/")
async def generate_episode_design_stream(
    text: str,
    instruct: str,
    speed: Optional[float] = None,
    language: Optional[str] = None,
    chunk_max_chars: Optional[int] = None,
    x_noctury_key: Optional[str] = Header(None),
):
    """
    Streaming SSE endpoint for long-text VoiceDesign generation.
    Sends each audio chunk as base64 via Server-Sent Events as it is generated,
    avoiding RunPod proxy timeouts on long texts.

    Client receives events:
      data: {"chunk": 1, "total": 3, "audio_b64": "...", "done": false}
      data: {"chunk": 3, "total": 3, "audio_b64": "...", "done": true, "duration": 45.2}

    The client assembles the WAV chunks in order.
    """
    verify_api_key(x_noctury_key)

    lang = language or DEFAULT_LANGUAGE
    chunk_max = chunk_max_chars or CHUNK_MAX_CHARS

    chunks = split_text_into_chunks(text, max_chars=chunk_max)
    total_chunks = len(chunks)
    total_duration = 0.0

    logging.info(f"SSE stream: {total_chunks} chunks | instruct={instruct[:60]}")

    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal total_duration
        for i, chunk_text in enumerate(chunks):
            chunk_start = time.time()
            logging.info(f"SSE chunk {i+1}/{total_chunks}: '{chunk_text[:50]}...'")

            # Seed déterministe dérivé du prompt instruct → même voix sur tous les chunks
            set_voice_seed(instruct)

            # Run blocking GPU inference in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            wavs, sr = await loop.run_in_executor(
                None,
                lambda ct=chunk_text: voice_design_model.generate_voice_design(
                    text=ct,
                    language=lang,
                    instruct=instruct,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            )

            audio_data = wavs[0]
            chunk_duration = len(audio_data) / sr
            total_duration += chunk_duration

            # Encode chunk as WAV bytes → base64
            buf = io.BytesIO()
            sf.write(buf, audio_data, sr, format="WAV")
            buf.seek(0)
            audio_b64 = base64.b64encode(buf.read()).decode("utf-8")

            is_done = (i == total_chunks - 1)
            payload = {
                "chunk": i + 1,
                "total": total_chunks,
                "audio_b64": audio_b64,
                "sample_rate": sr,
                "chunk_duration": round(chunk_duration, 2),
                "done": is_done,
            }
            if is_done:
                payload["total_duration"] = round(total_duration, 2)

            logging.info(f"SSE chunk {i+1} done in {time.time()-chunk_start:.1f}s ({chunk_duration:.1f}s audio)")
            yield f"data: {json.dumps(payload)}\n\n"

        # Final keepalive to signal end
        yield "data: {\"event\": \"end\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/upload_audio/")
async def upload_audio(
    audio_file_label: str = Form(...),
    file: UploadFile = File(...),
    x_noctury_key: Optional[str] = Header(None),
):
    """Upload an audio file for later use as the reference audio."""
    verify_api_key(x_noctury_key)
    try:
        contents = await file.read()

        allowed_extensions = {"wav", "mp3", "flac", "ogg"}
        max_file_size = 10 * 1024 * 1024  # 10MB

        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in allowed_extensions:
            return {"error": "Invalid file type. Allowed types are: wav, mp3, flac, ogg"}

        if len(contents) > max_file_size:
            return {"error": "File size is over limit. Max size is 10MB."}

        temp_file = io.BytesIO(contents)
        file_format = magic.from_buffer(temp_file.read(), mime=True)

        if "audio" not in file_format:
            return {"error": "Invalid file content."}

        stored_file_name = f"{audio_file_label}.{file_ext}"

        with open(f"{resources_dir}/{stored_file_name}", "wb") as f:
            f.write(contents)

        # Also create a WAV version
        wav_path = f"{resources_dir}/{audio_file_label}.wav"
        convert_to_wav(f"{resources_dir}/{stored_file_name}", wav_path)

        # Clear cached voice data if it exists
        if audio_file_label in voice_cache:
            del voice_cache[audio_file_label]
            logging.info(f"Cleared voice cache for: {audio_file_label}")

        return {
            "message": f"File {file.filename} uploaded successfully with label {audio_file_label}.",
            "voice_name": audio_file_label,
        }
    except Exception as e:
        logging.error(f"Error in upload_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/synthesize_speech/")
async def synthesize_speech(
    text: str,
    voice: str,
    speed: Optional[float] = None,
    language: Optional[str] = None,
    x_noctury_key: Optional[str] = Header(None),
):
    """Synthesize speech from text using a specified voice (single chunk)."""
    verify_api_key(x_noctury_key)
    start_time = time.time()
    try:
        logging.info(f"Generating speech for voice: {voice}")

        # Find voice file
        matching_files = [
            f for f in os.listdir(resources_dir) if f.startswith(voice) and f.lower().endswith(".wav")
        ]
        if not matching_files:
            matching_files = [f for f in os.listdir(resources_dir) if f.startswith(voice)]
            if not matching_files:
                raise HTTPException(status_code=400, detail="No matching voice found.")
            input_file = f"{resources_dir}/{matching_files[0]}"
            wav_path = f"{output_dir}/ref_converted.wav"
            convert_to_wav(input_file, wav_path)
            reference_file = wav_path
        else:
            reference_file = f"{resources_dir}/{matching_files[0]}"

        cache_data = get_or_create_voice_cache(voice, reference_file)

        audio_data, sr = generate_speech_with_prompt(
            text, cache_data["prompt"], speed=speed, language=language
        )

        save_path = f"{output_dir}/output_synthesized.wav"
        sf.write(save_path, audio_data, sr)

        result = StreamingResponse(open(save_path, "rb"), media_type="audio/wav")

        elapsed_time = time.time() - start_time
        result.headers["X-Elapsed-Time"] = str(elapsed_time)
        result.headers["X-Device-Used"] = device
        result.headers["X-Audio-Duration"] = str(len(audio_data) / sr)
        result.headers["Access-Control-Allow-Origin"] = "*"

        return result
    except Exception as e:
        logging.error(f"Error in synthesize_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class EpisodeRequest(BaseModel):
    """Request body for episode generation."""
    text: str
    voice: str
    speed: Optional[float] = None
    language: Optional[str] = None
    chunk_max_chars: Optional[int] = None
    output_format: Optional[str] = "wav"


@app.post("/generate_episode/")
async def generate_episode(
    request: EpisodeRequest,
    x_noctury_key: Optional[str] = Header(None),
):
    """
    Generate a full episode audio from long text using intelligent chunking.
    This is the main endpoint for Noctury episode generation.
    
    The text is automatically split into optimal chunks, each chunk is generated
    separately with the cloned voice, and all chunks are assembled with crossfade
    into a single seamless audio file.
    """
    verify_api_key(x_noctury_key)
    total_start = time.time()

    try:
        voice = request.voice
        text = request.text
        chunk_max = request.chunk_max_chars or CHUNK_MAX_CHARS

        logging.info(f"=== EPISODE GENERATION START ===")
        logging.info(f"Voice: {voice}, Text length: {len(text)} chars, Max chunk: {chunk_max}")

        # Find voice file
        matching_files = [
            f for f in os.listdir(resources_dir) if f.startswith(voice) and f.lower().endswith(".wav")
        ]
        if not matching_files:
            matching_files = [f for f in os.listdir(resources_dir) if f.startswith(voice)]
            if not matching_files:
                raise HTTPException(status_code=400, detail="No matching voice found.")
            input_file = f"{resources_dir}/{matching_files[0]}"
            wav_path = f"{output_dir}/ref_converted.wav"
            convert_to_wav(input_file, wav_path)
            reference_file = wav_path
        else:
            reference_file = f"{resources_dir}/{matching_files[0]}"

        # Get or create voice cache
        cache_data = get_or_create_voice_cache(voice, reference_file)

        # Split text into chunks
        chunks = split_text_into_chunks(text, max_chars=chunk_max)
        logging.info(f"Text split into {len(chunks)} chunks")

        # Generate audio for each chunk
        audio_chunks = []
        chunk_details = []
        sr = None

        for i, chunk_text in enumerate(chunks):
            chunk_start = time.time()
            logging.info(f"Generating chunk {i+1}/{len(chunks)}: '{chunk_text[:60]}...' ({len(chunk_text)} chars)")

            audio_data, chunk_sr = generate_speech_with_prompt(
                chunk_text,
                cache_data["prompt"],
                speed=request.speed,
                language=request.language,
            )

            if sr is None:
                sr = chunk_sr

            audio_chunks.append(audio_data)
            chunk_duration = len(audio_data) / sr
            chunk_time = time.time() - chunk_start

            chunk_details.append({
                "chunk_index": i,
                "text_length": len(chunk_text),
                "audio_duration_s": round(chunk_duration, 2),
                "generation_time_s": round(chunk_time, 2),
            })

            logging.info(
                f"Chunk {i+1} done: {chunk_duration:.1f}s audio in {chunk_time:.1f}s"
            )

        # Assemble all chunks
        logging.info("Assembling chunks...")
        final_audio = assemble_chunks_audio(audio_chunks, sr)
        total_duration = len(final_audio) / sr

        # Save output
        save_path = f"{output_dir}/episode_output.wav"
        sf.write(save_path, final_audio, sr)

        # Convert to MP3 if requested
        if request.output_format == "mp3":
            mp3_path = f"{output_dir}/episode_output.mp3"
            audio_seg = AudioSegment.from_wav(save_path)
            audio_seg.export(mp3_path, format="mp3", bitrate="192k")
            save_path = mp3_path
            media_type = "audio/mpeg"
        else:
            media_type = "audio/wav"

        total_time = time.time() - total_start

        logging.info(f"=== EPISODE GENERATION COMPLETE ===")
        logging.info(
            f"Total: {total_duration:.1f}s audio, {len(chunks)} chunks, "
            f"{total_time:.1f}s generation time"
        )

        result = StreamingResponse(open(save_path, "rb"), media_type=media_type)
        result.headers["X-Total-Time"] = str(round(total_time, 2))
        result.headers["X-Audio-Duration"] = str(round(total_duration, 2))
        result.headers["X-Chunks-Count"] = str(len(chunks))
        result.headers["X-Text-Length"] = str(len(text))
        result.headers["X-Device-Used"] = device
        result.headers["Access-Control-Allow-Origin"] = "*"

        return result

    except Exception as e:
        logging.error(f"Error in generate_episode: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_episode_json/")
async def generate_episode_json(
    request: EpisodeRequest,
    x_noctury_key: Optional[str] = Header(None),
):
    """
    Same as /generate_episode/ but returns JSON metadata with chunk details.
    The audio file is saved on the server and can be downloaded via /download/.
    """
    verify_api_key(x_noctury_key)
    total_start = time.time()

    try:
        voice = request.voice
        text = request.text
        chunk_max = request.chunk_max_chars or CHUNK_MAX_CHARS

        # Find voice file
        matching_files = [
            f for f in os.listdir(resources_dir) if f.startswith(voice) and f.lower().endswith(".wav")
        ]
        if not matching_files:
            matching_files = [f for f in os.listdir(resources_dir) if f.startswith(voice)]
            if not matching_files:
                raise HTTPException(status_code=400, detail="No matching voice found.")
            input_file = f"{resources_dir}/{matching_files[0]}"
            wav_path = f"{output_dir}/ref_converted.wav"
            convert_to_wav(input_file, wav_path)
            reference_file = wav_path
        else:
            reference_file = f"{resources_dir}/{matching_files[0]}"

        cache_data = get_or_create_voice_cache(voice, reference_file)

        chunks = split_text_into_chunks(text, max_chars=chunk_max)

        audio_chunks = []
        chunk_details = []
        sr = None

        for i, chunk_text in enumerate(chunks):
            chunk_start = time.time()
            audio_data, chunk_sr = generate_speech_with_prompt(
                chunk_text,
                cache_data["prompt"],
                speed=request.speed,
                language=request.language,
            )
            if sr is None:
                sr = chunk_sr

            audio_chunks.append(audio_data)
            chunk_duration = len(audio_data) / sr
            chunk_time = time.time() - chunk_start

            chunk_details.append({
                "chunk_index": i,
                "text": chunk_text,
                "text_length": len(chunk_text),
                "audio_duration_s": round(chunk_duration, 2),
                "generation_time_s": round(chunk_time, 2),
            })

        final_audio = assemble_chunks_audio(audio_chunks, sr)
        total_duration = len(final_audio) / sr

        # Save as WAV
        save_path = f"{output_dir}/episode_output.wav"
        sf.write(save_path, final_audio, sr)

        # Also save as MP3
        mp3_path = f"{output_dir}/episode_output.mp3"
        audio_seg = AudioSegment.from_wav(save_path)
        audio_seg.export(mp3_path, format="mp3", bitrate="192k")

        total_time = time.time() - total_start

        return {
            "status": "success",
            "total_duration_s": round(total_duration, 2),
            "total_generation_time_s": round(total_time, 2),
            "chunks_count": len(chunks),
            "text_length": len(text),
            "device": device,
            "voice": voice,
            "speed": request.speed or DEFAULT_SPEED,
            "language": request.language or DEFAULT_LANGUAGE,
            "download_wav": "/download/episode_output.wav",
            "download_mp3": "/download/episode_output.mp3",
            "chunks": chunk_details,
        }

    except Exception as e:
        logging.error(f"Error in generate_episode_json: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(
    filename: str,
    x_noctury_key: Optional[str] = Header(None),
):
    """Download a generated audio file."""
    verify_api_key(x_noctury_key)
    file_path = f"{output_dir}/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    return StreamingResponse(open(file_path, "rb"), media_type=media_type)


@app.post("/change_voice/")
async def change_voice(
    reference_speaker: str = Form(...),
    file: UploadFile = File(...),
    x_noctury_key: Optional[str] = Header(None),
):
    """Change the voice of an existing audio file."""
    verify_api_key(x_noctury_key)
    try:
        logging.info(f"Changing voice to {reference_speaker}...")

        contents = await file.read()

        input_path = f"{output_dir}/input_audio.wav"
        with open(input_path, "wb") as f:
            f.write(contents)

        matching_files = [f for f in os.listdir(resources_dir) if f.startswith(str(reference_speaker))]
        if not matching_files:
            raise HTTPException(status_code=400, detail="No matching reference speaker found.")

        reference_file = f"{resources_dir}/{matching_files[0]}"

        if not reference_file.lower().endswith(".wav"):
            ref_wav_path = f"{output_dir}/ref_converted.wav"
            convert_to_wav(reference_file, ref_wav_path)
            reference_file = ref_wav_path

        text = transcribe_audio(input_path)
        logging.info(f"Transcribed input audio: {text}")

        cache_data = get_or_create_voice_cache(reference_speaker, reference_file)

        audio_data, sr = generate_speech_with_prompt(text, cache_data["prompt"])

        save_path = f"{output_dir}/output_converted.wav"
        sf.write(save_path, audio_data, sr)

        return StreamingResponse(open(save_path, "rb"), media_type="audio/wav")
    except Exception as e:
        logging.error(f"Error in change_voice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
