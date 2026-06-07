# API Reference

Base URL: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Authentication

No authentication required for local deployments. For production, configure `API_KEY` in `.env` and pass:

```
Authorization: Bearer <API_KEY>
```

---

## Endpoints

### `GET /health`

System health check.

**Response 200:**
```json
{
  "status": "ok",
  "version": "0.3.0",
  "model_loaded": true,
  "device": "cpu"
}
```

---

### `POST /classify`

Classify a fauna species from an audio file.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | `File` | Yes | Audio file (.wav, .mp3, .flac, .ogg) |
| `model` | `string` | No | Model name (default: `cnn_baseline`) |
| `top_k` | `int` | No | Number of top predictions (default: 5) |
| `preset` | `string` | No | Taxon preset: `bats`, `birds`, `frogs`, `insects`, `mammals`, `reptiles` |

**Example (curl):**
```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@recording.wav" \
  -F "model=cnn_baseline" \
  -F "top_k=3"
```

**Example (Python):**
```python
import requests

with open("recording.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/classify",
        files={"file": ("recording.wav", f, "audio/wav")},
        data={"top_k": 3, "preset": "frogs"},
    )

result = response.json()
print(result["predictions"])
```

**Response 200:**
```json
{
  "filename": "recording.wav",
  "duration_s": 3.0,
  "sample_rate": 44100,
  "model": "cnn_baseline",
  "predictions": [
    {"rank": 1, "species": "Rana temporaria", "confidence": 0.87, "taxon": "amphibia"},
    {"rank": 2, "species": "Bufo bufo",        "confidence": 0.09, "taxon": "amphibia"},
    {"rank": 3, "species": "Hyla arborea",     "confidence": 0.03, "taxon": "amphibia"}
  ],
  "inference_time_ms": 42.3
}
```

**Response 422:** File format not supported or audio too short.

---

### `POST /classify/url`

Classify from a public audio URL.

**Request JSON:**
```json
{
  "url": "https://xeno-canto.org/sounds/uploaded/ZWAQYZNPND/XC123456.mp3",
  "model": "efficientnet",
  "top_k": 5
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/classify/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/bird.wav", "top_k": 3}'
```

---

### `GET /models`

List available models.

**Response 200:**
```json
{
  "models": [
    {
      "name": "cnn_baseline",
      "type": "BioAcousticCNN",
      "n_classes": 150,
      "loaded": true,
      "description": "CNN residual con atencion de canal"
    },
    {
      "name": "efficientnet",
      "type": "EfficientNetBioAcoustic",
      "n_classes": 150,
      "loaded": false,
      "description": "EfficientNet-B2 con fine-tuning progresivo"
    },
    {
      "name": "panns",
      "type": "PANNSCNN14BioAcoustic",
      "n_classes": 150,
      "loaded": false,
      "description": "PANNs-CNN14 preentrenado en AudioSet"
    }
  ]
}
```

---

### `POST /models/{model_name}/load`

Load a model into memory.

```bash
curl -X POST http://localhost:8000/models/efficientnet/load
```

**Response 200:**
```json
{"status": "loaded", "model": "efficientnet", "device": "cpu"}
```

---

### `GET /species`

List all species in the database.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `taxon` | string | all | Filter by taxonomic group |
| `limit` | int | 100 | Max results |
| `offset` | int | 0 | Pagination offset |
| `search` | string | - | Search by scientific or common name |

**Example:**
```bash
curl "http://localhost:8000/species?taxon=amphibia&limit=20"
```

**Response 200:**
```json
{
  "total": 48,
  "species": [
    {
      "id": 1,
      "scientific_name": "Rana temporaria",
      "common_name": "Common frog",
      "class": "Amphibia",
      "order": "Anura",
      "family": "Ranidae",
      "n_recordings": 124
    }
  ]
}
```

---

### `GET /species/{species_id}`

Get species detail with acoustic profile.

```bash
curl http://localhost:8000/species/1
```

**Response 200:**
```json
{
  "id": 1,
  "scientific_name": "Rana temporaria",
  "acoustic_profile": {
    "freq_range_hz": [100, 4000],
    "dominant_freq_hz": 800,
    "call_type": "advertisement_call",
    "typical_snr_db": 15.0
  }
}
```

---

### `POST /acoustic-indices`

Compute soundscape ecology indices from an audio file.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | `File` | Yes | Audio file |
| `windowed` | `bool` | No | Return per-window indices (default: false) |
| `window_s` | `float` | No | Window duration in seconds (default: 60.0) |

**Example:**
```bash
curl -X POST http://localhost:8000/acoustic-indices \
  -F "file=@soundscape.wav" \
  -F "windowed=true" \
  -F "window_s=60"
```

**Response 200:**
```json
{
  "filename": "soundscape.wav",
  "duration_s": 300.0,
  "indices": {
    "aci": 487.3,
    "adi": 2.14,
    "aei": 0.31,
    "bi": 0.0042,
    "ndsi": 0.72,
    "hf": 0.89,
    "ht": 0.76,
    "h": 0.68
  },
  "interpretation": {
    "ndsi": "Biophony dominante (bajo ruido antropogenico)",
    "adi": "Diversidad acustica alta",
    "aci": "Actividad biologica moderada-alta"
  }
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad request (invalid parameters) |
| 413 | File too large (max 50 MB) |
| 415 | Unsupported audio format |
| 422 | Validation error (audio too short, corrupted) |
| 500 | Internal server error |
| 503 | Model not loaded |

---

## Rate Limits

No rate limits for local deployments.

For production deployments, configure in `.env`:
```env
RATE_LIMIT_PER_MINUTE=60
```
