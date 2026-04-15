# Noctury TTS Server

Serveur TTS auto-hébergé pour Noctury, basé sur **Qwen3-TTS 1.7B** avec voice cloning, chunking intelligent pour les épisodes longs, et support natif du français. Optimisé pour le déploiement sur **RunPod Serverless**.

## Fonctionnalités

- **Voice Cloning** — Clonage vocal à partir d'un échantillon audio de référence (3-15 secondes)
- **Chunking Intelligent** — Découpe automatique du texte en segments optimaux pour générer des épisodes de durée illimitée
- **Français natif** — Langue par défaut configurée sur le français
- **Contrôle de vitesse** — Ajustement de la vitesse d'élocution (défaut : 0.92x pour un rythme intime)
- **Zéro censure** — Aucune restriction de contenu (auto-hébergé)
- **API REST** — Endpoints simples pour l'intégration avec le pipeline Noctury sur Railway
- **Authentification** — Protection par clé API configurable

## Endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Vérification de l'état du serveur |
| `/voices` | GET | Liste des voix disponibles |
| `/upload_audio/` | POST | Upload d'un fichier audio de référence pour le voice cloning |
| `/synthesize_speech/` | GET | Génération TTS simple (un seul chunk) |
| `/generate_episode/` | POST | Génération d'un épisode complet avec chunking intelligent |
| `/generate_episode_json/` | POST | Idem avec métadonnées JSON détaillées |
| `/download/{filename}` | GET | Téléchargement d'un fichier audio généré |
| `/change_voice/` | POST | Conversion de voix d'un fichier audio existant |

## Déploiement sur RunPod

### 1. Construire l'image Docker

```bash
docker build -t noctury-tts-server .
```

### 2. Tester localement

```bash
docker run --gpus all -p 7860:7860 \
  -e NOCTURY_TTS_API_KEY=your-secret-key \
  noctury-tts-server
```

### 3. Pousser sur Docker Hub

```bash
docker tag noctury-tts-server your-dockerhub/noctury-tts-server:latest
docker push your-dockerhub/noctury-tts-server:latest
```

### 4. Créer un Pod RunPod

1. Aller sur [RunPod](https://www.runpod.io/)
2. Créer un nouveau Pod GPU (A4000 16GB minimum recommandé)
3. Utiliser l'image Docker : `your-dockerhub/noctury-tts-server:latest`
4. Exposer le port 7860
5. Définir la variable d'environnement `NOCTURY_TTS_API_KEY`

## Utilisation

### Upload d'une voix de référence

```bash
curl -X POST "http://YOUR_SERVER:7860/upload_audio/" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "audio_file_label=maxime" \
  -F "file=@/path/to/maxime_sample.mp3"
```

### Génération d'un épisode complet

```bash
curl -X POST "http://YOUR_SERVER:7860/generate_episode/" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Votre texte d épisode complet ici...",
    "voice": "maxime",
    "speed": 0.92,
    "output_format": "mp3"
  }' \
  --output episode.mp3
```

### Synthèse simple (un seul chunk)

```bash
curl "http://YOUR_SERVER:7860/synthesize_speech/?text=Bonjour%20mon%20amour&voice=maxime&speed=0.92" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  --output output.wav
```

## Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `NOCTURY_TTS_API_KEY` | _(vide)_ | Clé API pour l'authentification (vide = pas d'auth) |
| `DEFAULT_LANGUAGE` | `French` | Langue par défaut |
| `DEFAULT_SPEED` | `0.92` | Vitesse d'élocution par défaut |
| `MAX_NEW_TOKENS` | `4096` | Nombre max de tokens audio par chunk |
| `CHUNK_MAX_CHARS` | `400` | Taille max d'un chunk en caractères |
| `CROSSFADE_MS` | `200` | Durée du crossfade entre chunks (ms) |

## Exigences matérielles

- **GPU** : NVIDIA avec 8 Go+ de VRAM (A4000, RTX 3090, A10G recommandés)
- **CUDA** : 12.8+
- **RAM** : 16 Go minimum

## Modèles utilisés

| Modèle | Rôle | Taille |
|--------|------|--------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Génération TTS + Voice Cloning | 1.7B paramètres |
| `openai/whisper-base` | Transcription audio de référence | 74M paramètres |

## Architecture d'intégration Noctury

```
Railway (Serveur Principal)
    │
    ├── Génération du texte (LLM)
    ├── Découpage en chunks
    │
    ▼
RunPod (Ce serveur)
    │
    ├── POST /generate_episode/
    ├── Voice cloning (voix Maxime)
    ├── Chunking + Assemblage
    │
    ▼
Railway (Post-traitement)
    │
    ├── Sound Design
    ├── Upload vers Cloudflare R2
    └── Mise à jour DB
```

## Licence

Apache 2.0 (Qwen3-TTS) + MIT (serveur)
