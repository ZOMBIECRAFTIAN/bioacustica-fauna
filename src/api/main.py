"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
API REST para el sistema de identificación bioacústica.
Framework: FastAPI + Pydantic v2 + asyncpg (PostgreSQL async).

Endpoints:
  POST /classify          → Inferencia sobre audio subido
  POST /classify/url      → Inferencia sobre audio por URL
  GET  /detections        → Listado de detecciones con filtros
  GET  /detections/{id}   → Detección individual
  GET  /species           → Catálogo de especies
  GET  /species/{id}      → Especie individual con estadísticas
  POST /recordings        → Registrar grabación nueva
  GET  /reports/site/{id} → Reporte de biodiversidad por sitio
  GET  /health            → Estado del servicio

Dependencias:
    pip install fastapi uvicorn[standard] python-multipart
                asyncpg sqlalchemy[asyncio] aiofiles pydantic

Ejecución:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Autor: Ian
Versión: 0.3.0
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

# Windows scientific stacks can load duplicate Intel OpenMP runtimes via deps.
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────


def _database_url_from_env() -> str:
    """Resuelve DATABASE_URL aceptando variables sueltas usadas por Docker."""
    raw = os.getenv("DATABASE_URL")
    if raw:
        if raw.startswith("postgresql://"):
            return raw.replace("postgresql://", "postgresql+asyncpg://", 1)
        return raw

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "bioacustica_fauna")
    user = os.getenv("DB_USER", "bioacustica_user")
    password = os.getenv("DB_PASSWORD", "bio")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"


class Settings:
    APP_NAME = "Bioacustica Fauna API"
    VERSION = "0.3.0"
    MODEL_PATH = (
        os.getenv("MODEL_PATH")
        or os.getenv("MODEL_CHECKPOINT_PATH")
        or "models/trained/multitaxon/best_efficientnet.pt"
    )
    MODEL_TYPE = os.getenv("MODEL_TYPE", "auto").lower()
    DB_URL = _database_url_from_env()
    MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "50"))
    TOP_K = int(os.getenv("TOP_K") or os.getenv("MODEL_TOP_K", "5"))
    MIN_PROB = float(os.getenv("MIN_PROB", "0.05"))
    DEVICE = os.getenv("DEVICE") or os.getenv(
        "MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )


settings = Settings()

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELOS PYDANTIC (Schemas)
# ─────────────────────────────────────────────────────────────────────────────


class PredictionItem(BaseModel):
    rank: int
    species_id: int | None = None
    scientific_name: str
    common_name_es: str | None = None
    common_name_en: str | None = None
    acoustic_group: str | None = None
    image_url: str | None = None
    source_url: str | None = None
    probability: float = Field(ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    request_id: str
    filename: str | None = None
    duration_s: float | None = None
    inference_ms: int
    segment_count: int
    predictions: list[PredictionItem]
    model_version: str


class DetectionFilter(BaseModel):
    species_id: int | None = None
    acoustic_group: str | None = None
    min_prob: float = 0.50
    site_id: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class DetectionOut(BaseModel):
    detection_id: str
    recording: str | None
    site: str | None
    t_start_s: float
    t_end_s: float
    scientific_name: str | None
    common_name_es: str | None
    acoustic_group: str | None
    probability: float
    rank: int
    is_correct: bool | None
    model: str
    created_at: str


class SpeciesOut(BaseModel):
    species_id: int
    scientific_name: str
    common_name_es: str | None
    common_name_en: str | None
    acoustic_group: str
    iucn_status: str | None
    family: str | None
    order: str | None
    class_: str | None = Field(None, alias="class")
    freq_min_hz: int | None
    freq_max_hz: int | None
    freq_dom_hz: int | None


class RecordingIn(BaseModel):
    filename: str
    file_path: str
    format: str = "WAV"
    sample_rate: int
    channels: int = 1
    duration_s: float
    source: str = "field"
    site_id: str | None = None
    recorded_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    notes: str | None = None


class RecordingOut(BaseModel):
    recording_id: str
    filename: str
    status: str


class SiteReport(BaseModel):
    site_id: str
    site_name: str
    country: str | None
    month: str
    species_richness: int
    total_detections: int
    avg_confidence: float
    species_list: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
    device: str
    version: str
    uptime_s: float


class FeedbackIn(BaseModel):
    request_id: str
    verdict: str = Field(pattern="^(correct|incorrect|unsure)$")
    expected_species: str | None = None
    notes: str | None = None


class DataEntryBatch(BaseModel):
    rows: list[dict[str, str | int | float | bool | None]]
    mode: str = Field(default="replace", pattern="^(replace|append)$")


class SpeciesInfo(BaseModel):
    scientific_name: str
    common_name_es: str | None = None
    common_name_en: str | None = None
    acoustic_group: str | None = None
    image_url: str | None = None
    source_url: str | None = None


class AudioSourceAPI(BaseModel):
    name: str
    url: str
    coverage: str
    notes: str


SPECIES_CATALOG: dict[str, dict[str, str]] = {
    "campylorhynchus_brunneicapillus": {
        "scientific_name": "Campylorhynchus brunneicapillus",
        "common_name_es": "Matraca del desierto",
        "common_name_en": "Cactus Wren",
        "acoustic_group": "bird",
    },
    "columbina_inca": {
        "scientific_name": "Columbina inca",
        "common_name_es": "Tortolita cola larga",
        "common_name_en": "Inca Dove",
        "acoustic_group": "bird",
    },
    "crotophaga_sulcirostris": {
        "scientific_name": "Crotophaga sulcirostris",
        "common_name_es": "Garrapatero pijuy",
        "common_name_en": "Groove-billed Ani",
        "acoustic_group": "bird",
    },
    "cyanocorax_yncas": {
        "scientific_name": "Cyanocorax yncas",
        "common_name_es": "Chara verde",
        "common_name_en": "Green Jay",
        "acoustic_group": "bird",
    },
    "geococcyx_californianus": {
        "scientific_name": "Geococcyx californianus",
        "common_name_es": "Correcaminos norteno",
        "common_name_en": "Greater Roadrunner",
        "acoustic_group": "bird",
    },
    "glaucidium_brasilianum": {
        "scientific_name": "Glaucidium brasilianum",
        "common_name_es": "Tecolote bajeno",
        "common_name_en": "Ferruginous Pygmy-Owl",
        "acoustic_group": "bird",
    },
    "haemorhous_mexicanus": {
        "scientific_name": "Haemorhous mexicanus",
        "common_name_es": "Pinzon mexicano",
        "common_name_en": "House Finch",
        "acoustic_group": "bird",
    },
    "icterus_pustulatus": {
        "scientific_name": "Icterus pustulatus",
        "common_name_es": "Bolsero dorso rayado",
        "common_name_en": "Streak-backed Oriole",
        "acoustic_group": "bird",
    },
    "melanerpes_aurifrons": {
        "scientific_name": "Melanerpes aurifrons",
        "common_name_es": "Carpintero frente dorada",
        "common_name_en": "Golden-fronted Woodpecker",
        "acoustic_group": "bird",
    },
    "momotus_lessonii": {
        "scientific_name": "Momotus lessonii",
        "common_name_es": "Momoto corona azul",
        "common_name_en": "Lesson's Motmot",
        "acoustic_group": "bird",
    },
    "myiozetetes_similis": {
        "scientific_name": "Myiozetetes similis",
        "common_name_es": "Luis gregario",
        "common_name_en": "Social Flycatcher",
        "acoustic_group": "bird",
    },
    "ortalis_vetula": {
        "scientific_name": "Ortalis vetula",
        "common_name_es": "Chachalaca oriental",
        "common_name_en": "Plain Chachalaca",
        "acoustic_group": "bird",
    },
    "pitangus_sulphuratus": {
        "scientific_name": "Pitangus sulphuratus",
        "common_name_es": "Luis bienteveo",
        "common_name_en": "Great Kiskadee",
        "acoustic_group": "bird",
    },
    "quiscalus_mexicanus": {
        "scientific_name": "Quiscalus mexicanus",
        "common_name_es": "Zanate mexicano",
        "common_name_en": "Great-tailed Grackle",
        "acoustic_group": "bird",
    },
    "setophaga_petechia": {
        "scientific_name": "Setophaga petechia",
        "common_name_es": "Chipe amarillo",
        "common_name_en": "Yellow Warbler",
        "acoustic_group": "bird",
    },
    "thryophilus_sinaloa": {
        "scientific_name": "Thryophilus sinaloa",
        "common_name_es": "Chivirin sinaloense",
        "common_name_en": "Sinaloa Wren",
        "acoustic_group": "bird",
    },
    "toxostoma_curvirostre": {
        "scientific_name": "Toxostoma curvirostre",
        "common_name_es": "Cuitlacoche pico curvo",
        "common_name_en": "Curve-billed Thrasher",
        "acoustic_group": "bird",
    },
    "turdus_grayi": {
        "scientific_name": "Turdus grayi",
        "common_name_es": "Mirlo pardo",
        "common_name_en": "Clay-colored Thrush",
        "acoustic_group": "bird",
    },
    "vireo_hypochryseus": {
        "scientific_name": "Vireo hypochryseus",
        "common_name_es": "Vireo dorado",
        "common_name_en": "Golden Vireo",
        "acoustic_group": "bird",
    },
    "zenaida_asiatica": {
        "scientific_name": "Zenaida asiatica",
        "common_name_es": "Paloma ala blanca",
        "common_name_en": "White-winged Dove",
        "acoustic_group": "bird",
    },
}

SOURCE_APIS = [
    AudioSourceAPI(
        name="Xeno-canto",
        url="https://xeno-canto.org/explore/api",
        coverage="Aves, ranas, murcielagos, mamiferos terrestres y ortopteros.",
        notes="Excelente para bioacustica; API v3 requiere key. Ya esta integrada en el proyecto.",
    ),
    AudioSourceAPI(
        name="iNaturalist / NaturaLista",
        url="https://api.inaturalist.org/v2/docs/",
        coverage="Aves, mamiferos, anfibios, reptiles, insectos y otros grupos.",
        notes="Muy buena para Mexico. Filtra observaciones con audio usando sounds=true.",
    ),
    AudioSourceAPI(
        name="GBIF",
        url="https://techdocs.gbif.org/en/openapi/v1/occurrence",
        coverage="Ocurrencias biologicas con multimedia, incluyendo Sound.",
        notes="Agrega datos de muchas instituciones; util para descubrir audios y licencias.",
    ),
    AudioSourceAPI(
        name="Wikimedia Commons",
        url="https://commons.wikimedia.org/wiki/Category:Audio_files_of_animals",
        coverage="Audios abiertos de animales, con licencias por archivo.",
        notes="Bueno como complemento para especies con pocos ejemplos.",
    ),
    AudioSourceAPI(
        name="EcoSounds",
        url="https://api.ecosounds.org/",
        coverage="Grabaciones ambientales y proyectos ecoacusticos.",
        notes="Mas orientado a soundscapes; revisar permisos por proyecto.",
    ),
    AudioSourceAPI(
        name="audioBlast",
        url="https://audioblast.org/",
        coverage="Busqueda bioacustica y referencias de grabaciones.",
        notes="Util para descubrimiento; revisar licencia antes de entrenar.",
    ),
]

_species_info_cache: dict[str, SpeciesInfo] = {}
APP_DATA_DIR = Path("data/app")
HISTORY_PATH = APP_DATA_DIR / "history.jsonl"
FEEDBACK_PATH = APP_DATA_DIR / "feedback.jsonl"
DATA_ENTRY_DIR = Path(os.getenv("DATA_ENTRY_DIR", "data/app/data_entry"))
DATA_ENTRY_SCHEMAS: dict[str, list[str]] = {
    "dataset": [
        "original_audio_path",
        "class_label",
        "scientific_name",
        "acoustic_group",
        "source",
        "source_url",
        "license",
        "country",
        "state",
        "site_id",
        "date",
        "recordist",
        "sample_rate",
        "duration_s",
        "notes",
    ],
    "field": [
        "recording_id",
        "site_id",
        "date",
        "recorder",
        "microphone",
        "habitat",
        "weather",
        "duration_s",
        "dominant_noise",
        "expected_species_or_group",
        "license_or_permission",
        "notes",
    ],
    "negative": [
        "original_audio_path",
        "class_label",
        "acoustic_group",
        "source",
        "license",
        "site_id",
        "date",
        "duration_s",
        "dominant_noise",
        "notes",
    ],
    "recording": [
        "filename",
        "file_path",
        "format",
        "sample_rate",
        "channels",
        "duration_s",
        "source",
        "site_id",
        "recorded_at",
        "latitude",
        "longitude",
        "notes",
    ],
}


def _species_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _catalog_info(name: str) -> dict[str, str]:
    key = _species_key(name)
    if key in SPECIES_CATALOG:
        return dict(SPECIES_CATALOG[key])
    parts = key.split("_")
    sci = " ".join(parts).capitalize() if parts else name
    return {"scientific_name": sci, "acoustic_group": "unknown"}


def _fetch_inaturalist_info(scientific_name: str) -> dict[str, str]:
    """Consulta iNaturalist para foto y nombre comun; falla silenciosamente offline."""
    try:
        import requests

        resp = requests.get(
            "https://api.inaturalist.org/v1/taxa",
            params={"q": scientific_name, "rank": "species", "locale": "es", "per_page": 5},
            timeout=4,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as e:
        logger.debug(f"iNaturalist no disponible para {scientific_name}: {e}")
        return {}

    target = scientific_name.lower()
    match = None
    for item in results:
        if str(item.get("name", "")).lower() == target:
            match = item
            break
    if match is None and results:
        match = results[0]
    if not match:
        return {}

    photo = match.get("default_photo") or {}
    image_url = photo.get("medium_url") or photo.get("square_url") or photo.get("url")
    info: dict[str, str] = {"source_url": f"https://www.inaturalist.org/taxa/{match.get('id')}"}
    if match.get("preferred_common_name"):
        info["common_name_es"] = str(match["preferred_common_name"])
    if image_url:
        info["image_url"] = str(image_url)
    return info


def species_info_for(name: str) -> SpeciesInfo:
    key = _species_key(name)
    if key in _species_info_cache:
        return _species_info_cache[key]

    merged = _catalog_info(name)
    merged.update(_fetch_inaturalist_info(merged["scientific_name"]))
    info = SpeciesInfo(**merged)
    _species_info_cache[key] = info
    return info


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json_dumps(record) + "\n")


def _data_entry_csv_path(kind: str) -> Path:
    if kind not in DATA_ENTRY_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Tipo de captura no soportado: {kind}")
    DATA_ENTRY_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_ENTRY_DIR / f"{kind}.csv"


def _read_data_entry_rows(kind: str) -> list[dict[str, str]]:
    path = _data_entry_csv_path(kind)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_data_entry_rows(kind: str, rows: list[dict], mode: str) -> int:
    path = _data_entry_csv_path(kind)
    fields = DATA_ENTRY_SCHEMAS[kind]
    existing = _read_data_entry_rows(kind) if mode == "append" else []
    clean_rows = []
    for row in [*existing, *rows]:
        clean_rows.append(
            {field: "" if row.get(field) is None else str(row.get(field, "")) for field in fields}
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean_rows)
    return len(clean_rows)


def _read_jsonl(path: Path, limit: int = 50) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(__import__("json").loads(line))
            except Exception:
                continue
    return list(reversed(rows[-limit:]))


def json_dumps(record: dict) -> str:
    return __import__("json").dumps(record, ensure_ascii=False, separators=(",", ":"))


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL MANAGER (Singleton)
# ─────────────────────────────────────────────────────────────────────────────


class ModelManager:
    """
    Gestiona el ciclo de vida del modelo de clasificación.
    Carga el modelo al arrancar y provee inferencia thread-safe.
    """

    def __init__(self):
        self._model = None
        self._classes: list[str] = []
        self._version: str = "unloaded"
        self._loaded: bool = False

    def _load_class_names(self, checkpoint: dict, model_path: Path, n_classes: int) -> list[str]:
        """Resuelve clases desde checkpoint o class_names.json junto al modelo."""
        raw_classes = checkpoint.get("class_names")
        if isinstance(raw_classes, list) and raw_classes:
            return [str(c) for c in raw_classes]

        class_file = model_path.with_name("class_names.json")
        if class_file.exists():
            try:
                import json

                parsed = json.loads(class_file.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    raw_classes = parsed.get("classes") or parsed.get("class_names")
                elif isinstance(parsed, list):
                    raw_classes = parsed
                if isinstance(raw_classes, list) and raw_classes:
                    return [str(c) for c in raw_classes]
            except Exception as e:
                logger.warning(f"No se pudo leer {class_file}: {e}")

        return [str(i) for i in range(n_classes)]

    def _detect_model_type(self, checkpoint: dict) -> str:
        if settings.MODEL_TYPE != "auto":
            return settings.MODEL_TYPE

        state = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
        if isinstance(state, dict):
            keys = set(state.keys())
            if any(k.startswith("stem.") or k.startswith("layer1.") for k in keys):
                return "cnn_baseline"
            if any("efficientnet" in k or k.startswith("backbone.") for k in keys):
                return "efficientnet"
            if any("spectrogram_extractor" in k or "logmel_extractor" in k for k in keys):
                return "panns"
        return "cnn_baseline"

    def load(self, model_path: str, device: str = "cpu") -> bool:
        """Carga el modelo desde checkpoint. Retorna True si OK."""
        try:
            path = Path(model_path)
            if not path.exists():
                logger.warning(f"Modelo no encontrado: {model_path}. API en modo degradado.")
                return False

            ck = torch.load(model_path, map_location=device)
            n_classes = ck.get("n_classes", 10)
            model_type = self._detect_model_type(ck)

            if model_type in {"efficientnet", "efficientnet_b0"}:
                from src.models.efficientnet_classifier import load_efficientnet

                self._model = load_efficientnet(model_path, device=device)
            elif model_type in {"panns", "panns_cnn14"}:
                from src.models.panns_classifier import load_panns

                self._model = load_panns(model_path, device=device)
            else:
                from src.models.cnn_baseline import load_model

                self._model = load_model(model_path, device=device)

            self._classes = self._load_class_names(ck, path, n_classes)
            val_acc = ck.get("val_acc")
            val_part = f"_val{val_acc:.3f}" if isinstance(val_acc, int | float) else ""
            self._version = f"{model_type}_e{ck.get('epoch', '?')}{val_part}"
            self._loaded = True
            logger.info(
                f"Modelo cargado: {model_path} | type={model_type} | "
                f"classes={len(self._classes)} | device={device}"
            )
            return True

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self._loaded = False
            return False

    @torch.no_grad()
    def predict(
        self,
        audio: np.ndarray,
        sr: int,
        top_k: int = 5,
        preset: str = "birds",
    ) -> list[dict]:
        """
        Inferencia sobre un array de audio crudo.

        Returns:
            Lista de {'rank', 'class', 'probability'}
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Modelo no cargado")

        # Importar procesador en runtime
        from src.audio_processing.preprocessor import PRESETS, AudioConfig, AudioPreprocessor

        cfg = PRESETS.get(preset, AudioConfig(sample_rate=sr))
        proc = AudioPreprocessor(cfg)

        # Preprocesamiento alineado con el preset usado en entrenamiento
        if cfg.apply_bandpass:
            audio = proc.bandpass_filter(audio)
        if cfg.apply_noise_reduction:
            audio = proc.reduce_noise(audio)
        if cfg.normalize:
            audio = proc.normalize(audio)
        mel = proc.mel_spectrogram(audio)  # (n_mels, T)

        # Resize a (1, 1, 128, 128)
        spec_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
        import torch.nn.functional as F

        spec_t = F.interpolate(spec_t, size=(128, 128), mode="bilinear", align_corners=False)

        # Normalizar por canal
        mean, std = spec_t.mean(), spec_t.std() + 1e-8
        spec_t = (spec_t - mean) / std

        probs = self._model.predict_proba(spec_t).squeeze(0).cpu().numpy()
        top_idx = np.argsort(probs)[::-1][:top_k]

        return [
            {
                "rank": int(r + 1),
                "class": self._classes[i] if i < len(self._classes) else str(i),
                "probability": float(probs[i]),
            }
            for r, i in enumerate(top_idx)
            if float(probs[i]) >= settings.MIN_PROB
        ]

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self._version


# Singleton global
model_manager = ModelManager()
_start_time = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# 4. BASE DE DATOS (async)
# ─────────────────────────────────────────────────────────────────────────────

# Motor SQLAlchemy async — se inicializa en startup
_engine = None
_db_ok = False


async def get_db_status() -> bool:
    global _db_ok
    if _engine is None:
        return False
    try:
        async with _engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        _db_ok = True
    except Exception:
        _db_ok = False
    return _db_ok


# ─────────────────────────────────────────────────────────────────────────────
# 5. APLICACIÓN FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown using the modern FastAPI lifespan interface."""
    global _engine, _db_ok
    logger.info(f"Iniciando {settings.APP_NAME} v{settings.VERSION}")

    # Cargar modelo
    model_manager.load(settings.MODEL_PATH, device=settings.DEVICE)

    # Conectar DB
    try:
        from sqlalchemy.ext.asyncio import create_async_engine

        _engine = create_async_engine(settings.DB_URL, pool_size=5, max_overflow=10)
        _db_ok = await get_db_status()
        if _db_ok:
            logger.info("Base de datos conectada.")
        else:
            logger.warning("Base de datos NO disponible. Endpoints de DB deshabilitados.")
    except Exception as e:
        logger.warning(f"DB no disponible: {e}")

    yield  # ← aplicación corriendo

    # Shutdown
    if _engine:
        await _engine.dispose()
    logger.info("API detenida.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "API REST para identificación automática de fauna silvestre mediante "
        "análisis bioacústico e inteligencia artificial. "
        "Soporta: mamíferos, anfibios, reptiles e insectos."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

WEB_APP_FILE = Path(__file__).with_name("static") / "index.html"
DATA_ENTRY_FILE = Path(__file__).with_name("static") / "data_entry.html"


@app.middleware("http")
async def serve_web_ui(request: Request, call_next):
    if request.url.path == "/" and WEB_APP_FILE.exists():
        return HTMLResponse(WEB_APP_FILE.read_text(encoding="utf-8"))
    return await call_next(request)


@app.get("/api/sources", response_model=list[AudioSourceAPI], tags=["Web"])
async def audio_sources():
    return SOURCE_APIS


@app.get("/data-entry", response_class=HTMLResponse, tags=["Web"])
async def data_entry_app():
    if DATA_ENTRY_FILE.exists():
        return HTMLResponse(DATA_ENTRY_FILE.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="data_entry.html no encontrado")


@app.get("/api/data-entry/{kind}", tags=["Web"])
async def get_data_entry_rows(kind: str):
    return {"kind": kind, "rows": _read_data_entry_rows(kind)}


@app.post("/api/data-entry/{kind}", tags=["Web"])
async def save_data_entry_rows(kind: str, payload: DataEntryBatch):
    total = _write_data_entry_rows(kind, payload.rows, payload.mode)
    return {
        "kind": kind,
        "rows": total,
        "path": _data_entry_csv_path(kind).as_posix(),
    }


@app.get("/api/species/{name}", response_model=SpeciesInfo, tags=["Web"])
async def species_lookup(name: str):
    return species_info_for(name)


@app.get("/species-card/{name:path}", response_class=HTMLResponse, tags=["Web"])
async def species_card(name: str):
    html = __import__("html")
    info = species_info_for(name)
    common = info.common_name_es or info.common_name_en or info.scientific_name
    image = (
        f'<img src="{html.escape(info.image_url)}" alt="{html.escape(common)}" />'
        if info.image_url
        else '<div class="fallback">Sin foto</div>'
    )
    source = (
        f'<a href="{html.escape(info.source_url)}" target="_blank" rel="noreferrer">Fuente</a>'
        if info.source_url
        else ""
    )
    return HTMLResponse(
        f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(common)}</title>
  <style>
    body {{ margin:0; font-family:system-ui, sans-serif; background:#eef2ee; color:#14211d; }}
    main {{ width:min(860px, calc(100% - 28px)); margin:0 auto; padding:28px 0; }}
    a {{ color:#07564f; font-weight:700; text-decoration:none; }}
    .card {{ background:#fff; border:1px solid #d5ded6; border-radius:8px; overflow:hidden; box-shadow:0 18px 45px rgba(22,34,28,.1); }}
    img, .fallback {{ width:100%; height:min(54vw, 420px); object-fit:cover; display:grid; place-items:center; background:#dbeaf2; }}
    .body {{ padding:18px; }}
    h1 {{ margin:0; font-size:30px; letter-spacing:0; }}
    .sci {{ margin-top:6px; color:#65736c; font-style:italic; }}
    .meta {{ margin-top:16px; color:#65736c; }}
  </style>
</head>
<body>
  <main>
    <p><a href="/">Volver</a></p>
    <article class="card">
      {image}
      <div class="body">
        <h1>{html.escape(common)}</h1>
        <div class="sci">{html.escape(info.scientific_name)}</div>
        <div class="meta">Grupo: {html.escape(info.acoustic_group or "desconocido")} · {source}</div>
      </div>
    </article>
  </main>
</body>
</html>
    """
    )


@app.get("/api/history", tags=["Web"])
async def classification_history(limit: int = Query(default=25, ge=1, le=200)):
    return _read_jsonl(HISTORY_PATH, limit=limit)


@app.post("/api/feedback", tags=["Web"])
async def save_feedback(feedback: FeedbackIn):
    record = feedback.model_dump()
    record["created_at"] = _utc_now()
    _append_jsonl(FEEDBACK_PATH, record)
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, tags=["Web"])
async def web_app():
    return HTMLResponse(
        """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BioAcoustics Mexico</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f4;
      --ink: #17211b;
      --muted: #657169;
      --line: #d8dfd8;
      --panel: #ffffff;
      --accent: #157a6e;
      --accent-2: #c06722;
      --soft: #e7f3ef;
      --bad: #a13b3b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      padding: 8px 0 18px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.1;
      font-weight: 760;
      letter-spacing: 0;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--bad);
    }
    .dot.ok { background: var(--accent); }
    .layout {
      display: grid;
      grid-template-columns: minmax(280px, 380px) 1fr;
      gap: 18px;
      margin-top: 22px;
      align-items: start;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 16px;
      line-height: 1.2;
      font-weight: 720;
      letter-spacing: 0;
    }
    label {
      display: block;
      margin: 14px 0 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    input[type="file"], select, input[type="number"] {
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    audio {
      width: 100%;
      margin-top: 14px;
    }
    button {
      width: 100%;
      min-height: 42px;
      margin-top: 16px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 760;
      cursor: pointer;
    }
    button:disabled {
      cursor: wait;
      opacity: .65;
    }
    .message {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      min-height: 19px;
    }
    .message.error { color: var(--bad); }
    .summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfb;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric strong {
      font-size: 17px;
      line-height: 1;
    }
    .empty {
      min-height: 280px;
      display: grid;
      place-items: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      text-align: center;
      padding: 24px;
    }
    .prediction {
      display: grid;
      grid-template-columns: 36px 1fr 70px;
      gap: 12px;
      align-items: center;
      padding: 12px 0;
      border-bottom: 1px solid var(--line);
    }
    .prediction:last-child { border-bottom: 0; }
    .rank {
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--soft);
      color: var(--accent);
      font-weight: 780;
    }
    .name {
      font-weight: 760;
      overflow-wrap: anywhere;
    }
    .bar {
      height: 8px;
      margin-top: 7px;
      border-radius: 99px;
      background: #e8ece8;
      overflow: hidden;
    }
    .bar > i {
      display: block;
      height: 100%;
      width: 0;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }
    .prob {
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-weight: 720;
    }
    @media (max-width: 760px) {
      main { width: min(100% - 20px, 1120px); padding-top: 12px; }
      header { align-items: start; flex-direction: column; }
      .layout { grid-template-columns: 1fr; }
      .summary { grid-template-columns: 1fr; }
      .prediction { grid-template-columns: 32px 1fr; }
      .prob { grid-column: 2; text-align: left; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>BioAcoustics Mexico</h1>
      <div class="status"><span id="status-dot" class="dot"></span><span id="status-text">Modelo</span></div>
    </header>

    <div class="layout">
      <section>
        <h2>Audio</h2>
        <form id="form">
          <label for="audio-file">Archivo</label>
          <input id="audio-file" name="file" type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.aif,.aiff" required />

          <div class="row">
            <div>
              <label for="preset">Preset</label>
              <select id="preset" name="preset">
                <option value="birds" selected>Aves</option>
                <option value="mammals">Mamiferos</option>
                <option value="frogs">Anfibios</option>
                <option value="reptiles">Reptiles</option>
                <option value="insects">Insectos</option>
              </select>
            </div>
            <div>
              <label for="top-k">Top K</label>
              <input id="top-k" name="top_k" type="number" min="1" max="10" value="5" />
            </div>
          </div>

          <audio id="player" controls hidden></audio>
          <button id="submit" type="submit">Identificar</button>
          <div id="message" class="message"></div>
        </form>
      </section>

      <section>
        <h2>Resultados</h2>
        <div id="summary" class="summary" hidden>
          <div class="metric"><span>Duracion</span><strong id="duration">-</strong></div>
          <div class="metric"><span>Segmentos</span><strong id="segments">-</strong></div>
          <div class="metric"><span>Tiempo</span><strong id="inference">-</strong></div>
        </div>
        <div id="results" class="empty">Sin resultados</div>
      </section>
    </div>
  </main>

  <script>
    const form = document.getElementById("form");
    const fileInput = document.getElementById("audio-file");
    const player = document.getElementById("player");
    const submit = document.getElementById("submit");
    const message = document.getElementById("message");
    const results = document.getElementById("results");
    const summary = document.getElementById("summary");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");

    const fmtName = (name) => name.replaceAll("_", " ");
    const pct = (value) => `${Math.round(value * 1000) / 10}%`;

    async function loadHealth() {
      try {
        const res = await fetch("/health");
        const data = await res.json();
        statusDot.classList.toggle("ok", Boolean(data.model_loaded));
        statusText.textContent = data.model_loaded ? `Modelo listo · ${data.device}` : "Modelo no cargado";
      } catch {
        statusText.textContent = "API sin respuesta";
      }
    }

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      player.src = URL.createObjectURL(file);
      player.hidden = false;
      message.textContent = file.name;
      message.classList.remove("error");
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const file = fileInput.files[0];
      if (!file) return;

      const body = new FormData();
      body.append("file", file);
      body.append("preset", document.getElementById("preset").value);
      body.append("top_k", document.getElementById("top-k").value);

      submit.disabled = true;
      submit.textContent = "Procesando...";
      message.textContent = "";
      message.classList.remove("error");

      try {
        const res = await fetch("/classify", { method: "POST", body });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Error al clasificar");

        document.getElementById("duration").textContent = `${data.duration_s ?? 0}s`;
        document.getElementById("segments").textContent = data.segment_count;
        document.getElementById("inference").textContent = `${data.inference_ms} ms`;
        summary.hidden = false;

        if (!data.predictions.length) {
          results.className = "empty";
          results.textContent = "Sin predicciones sobre el umbral";
          return;
        }

        results.className = "";
        results.innerHTML = data.predictions.map((p) => `
          <div class="prediction">
            <div class="rank">${p.rank}</div>
            <div>
              <div class="name">${fmtName(p.scientific_name)}</div>
              <div class="bar"><i style="width:${Math.max(2, p.probability * 100)}%"></i></div>
            </div>
            <div class="prob">${pct(p.probability)}</div>
          </div>
        `).join("");
      } catch (err) {
        message.textContent = err.message || String(err);
        message.classList.add("error");
      } finally {
        submit.disabled = false;
        submit.textContent = "Identificar";
      }
    });

    loadHealth();
  </script>
</body>
</html>
        """
    )


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Estado del servicio, modelo y conexión a base de datos."""
    return HealthResponse(
        status="ok" if model_manager.loaded else "degraded",
        model_loaded=model_manager.loaded,
        db_connected=_db_ok,
        device=settings.DEVICE,
        version=settings.VERSION,
        uptime_s=round(time.time() - _start_time, 1),
    )


# ── Clasificación por archivo ─────────────────────────────────────────────────


@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
async def classify_audio(
    file: UploadFile = File(..., description="Archivo de audio: WAV, MP3, FLAC, OGG"),
    top_k: int = Form(default=5, ge=1, le=20, description="Número de predicciones"),
    preset: str = Form(default="birds", description="Preset acústico: birds, mammals, frogs, etc."),
    region: str | None = Form(default=None, description="Región/estado asociado al audio"),
    lat: float | None = Form(default=None, description="Latitud pública/opcional"),
    lng: float | None = Form(default=None, description="Longitud pública/opcional"),
):
    """
    Clasifica el contenido biológico de un archivo de audio.

    Sube un archivo WAV/MP3/FLAC y obtiene las top-K especies más probables
    con su probabilidad de detección.
    """
    # Validar formato
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".webm", ".m4a"}
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Formato no soportado: {ext}. Usar: {allowed}",
        )

    # Validar tamaño
    content = await file.read()
    if len(content) > settings.MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Archivo mayor a {settings.MAX_FILE_MB} MB",
        )

    # Guardar en temporal y procesar
    t0 = time.time()
    try:
        import librosa

        from src.audio_processing.preprocessor import PRESETS

        if preset not in PRESETS:
            raise HTTPException(
                status_code=422,
                detail=f"Preset no reconocido: {preset}. Opciones: {list(PRESETS)}",
            )

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        target_sr = PRESETS[preset].sample_rate
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        duration = len(audio) / sr

        # Inferencia por ventanas de 3 segundos
        window = 3 * sr
        hop = int(1.5 * sr)
        all_preds: list[dict] = []

        if len(audio) <= window:
            preds = model_manager.predict(audio, sr=sr, top_k=top_k, preset=preset)
            all_preds.extend(preds)
            n_segs = 1
        else:
            n_segs = 0
            start = 0
            while start < len(audio):
                seg = audio[start : start + window]
                if len(seg) < window // 4:
                    break
                preds = model_manager.predict(seg, sr=sr, top_k=top_k, preset=preset)
                all_preds.extend(preds)
                start += hop
                n_segs += 1

        # Agregar predicciones: promedio de probabilidades por clase
        agg: dict[str, list[float]] = {}
        for pred in all_preds:
            agg.setdefault(pred["class"], []).append(pred["probability"])

        final = sorted(
            [{"class": k, "probability": float(np.mean(v))} for k, v in agg.items()],
            key=lambda x: -x["probability"],
        )[:top_k]

        inference_ms = int((time.time() - t0) * 1000)
        enriched_predictions: list[PredictionItem] = []
        for i, p in enumerate(final):
            info = species_info_for(p["class"])
            enriched_predictions.append(
                PredictionItem(
                    rank=i + 1,
                    scientific_name=info.scientific_name,
                    common_name_es=info.common_name_es,
                    common_name_en=info.common_name_en,
                    acoustic_group=info.acoustic_group,
                    image_url=info.image_url,
                    source_url=info.source_url,
                    probability=p["probability"],
                )
            )

        response = ClassifyResponse(
            request_id=str(__import__("uuid").uuid4()),
            filename=file.filename,
            duration_s=round(duration, 3),
            inference_ms=inference_ms,
            segment_count=n_segs,
            predictions=enriched_predictions,
            model_version=model_manager.version,
        )
        _append_jsonl(
            HISTORY_PATH,
            {
                "created_at": _utc_now(),
                "request_id": response.request_id,
                "filename": response.filename,
                "duration_s": response.duration_s,
                "inference_ms": response.inference_ms,
                "segment_count": response.segment_count,
                "preset": preset,
                "region": region,
                "lat": lat,
                "lng": lng,
                "model_version": response.model_version,
                "predictions": [p.model_dump() for p in response.predictions],
            },
        )
        return response

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error en /classify")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import os as _os

        try:
            _os.unlink(tmp_path)  # type: ignore[possibly-undefined]
        except Exception:
            pass


# ── Clasificación por URL ─────────────────────────────────────────────────────


@app.post("/classify/url", response_model=ClassifyResponse, tags=["Inference"])
async def classify_audio_url(
    audio_url: str = Query(..., description="URL pública del archivo de audio"),
    top_k: int = Query(default=5, ge=1, le=20),
    preset: str = Query(default="birds"),
):
    """Clasifica audio referenciado por URL pública (Xeno-canto, iNaturalist, etc.)."""
    import requests as req_lib

    try:
        resp = req_lib.get(audio_url, timeout=30, stream=True)
        resp.raise_for_status()

        ext = Path(audio_url.split("?")[0]).suffix.lower() or ".mp3"
        content = resp.content

        # Crear UploadFile virtual y delegar
        from io import BytesIO

        vfile = UploadFile(
            filename=f"remote{ext}",
            file=BytesIO(content),
        )
        return await classify_audio(vfile, top_k=top_k, preset=preset)

    except req_lib.RequestException as e:
        raise HTTPException(status_code=422, detail=f"No se pudo descargar el audio: {e}")


# ── Detecciones ───────────────────────────────────────────────────────────────


@app.get("/detections", response_model=list[DetectionOut], tags=["Data"])
async def list_detections(
    species_id: int | None = Query(None),
    acoustic_group: str | None = Query(None),
    min_prob: float = Query(default=0.5, ge=0.0, le=1.0),
    site_id: str | None = Query(None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Lista detecciones con filtros opcionales por especie, grupo, sitio y probabilidad."""
    if not _db_ok or _engine is None:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")

    import sqlalchemy as sa

    query = """
        SELECT
            detection_id::text,
            recording,
            site,
            t_start_s,
            t_end_s,
            scientific_name,
            common_name_es,
            acoustic_group,
            probability,
            rank,
            is_correct,
            model,
            created_at::text
        FROM detection_summary
        WHERE probability >= :min_prob
    """
    params: dict = {"min_prob": min_prob}

    if species_id:
        query += " AND species_id = :species_id"
        params["species_id"] = species_id
    if acoustic_group:
        query += " AND acoustic_group = :acoustic_group"
        params["acoustic_group"] = acoustic_group
    if site_id:
        query += " AND site_id::text = :site_id"
        params["site_id"] = site_id

    query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
    params["limit"] = limit
    params["offset"] = offset

    async with _engine.connect() as conn:
        rows = (await conn.execute(sa.text(query), params)).mappings().all()

    return [DetectionOut(**dict(row)) for row in rows]


@app.get("/detections/{detection_id}", response_model=DetectionOut, tags=["Data"])
async def get_detection(detection_id: str):
    """Obtiene una detección individual por ID."""
    if not _db_ok or _engine is None:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")

    import sqlalchemy as sa

    async with _engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    sa.text("SELECT * FROM detection_summary WHERE detection_id::text = :id"),
                    {"id": detection_id},
                )
            )
            .mappings()
            .first()
        )

    if not row:
        raise HTTPException(status_code=404, detail=f"Detección {detection_id} no encontrada")
    return DetectionOut(**dict(row))


# ── Especies ──────────────────────────────────────────────────────────────────


@app.get("/species", response_model=list[SpeciesOut], tags=["Taxonomy"])
async def list_species(
    acoustic_group: str | None = Query(
        None, description="Filtrar por grupo: mammal_bat, amphibian_anura, insect_orthoptera, etc."
    ),
    iucn_status: str | None = Query(
        None, description="Filtrar por estado IUCN: LC, NT, VU, EN, CR"
    ),
    search: str | None = Query(None, description="Búsqueda por nombre científico o común"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """Listado del catálogo de especies con filtros taxonómicos."""
    if not _db_ok or _engine is None:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")

    import sqlalchemy as sa

    q = "SELECT * FROM species_full WHERE TRUE"
    params: dict = {}

    if acoustic_group:
        q += " AND acoustic_group = :ag"
        params["ag"] = acoustic_group
    if iucn_status:
        q += " AND iucn_status = :iucn"
        params["iucn"] = iucn_status
    if search:
        q += " AND (scientific_name ILIKE :s OR common_name_es ILIKE :s)"
        params["s"] = f"%{search}%"

    q += " ORDER BY scientific_name LIMIT :limit OFFSET :offset"
    params["limit"] = limit
    params["offset"] = offset

    async with _engine.connect() as conn:
        rows = (await conn.execute(sa.text(q), params)).mappings().all()

    return [
        SpeciesOut(
            species_id=r["species_id"],
            scientific_name=r["scientific_name"],
            common_name_es=r["common_name_es"],
            common_name_en=r["common_name_en"],
            acoustic_group=r["acoustic_group"],
            iucn_status=r["iucn_status"],
            family=r["family"],
            order=r["order"],
            **{"class": r["class"]},
            freq_min_hz=None,
            freq_max_hz=None,
            freq_dom_hz=None,
        )
        for r in rows
    ]


@app.get("/species/{species_id}", response_model=SpeciesOut, tags=["Taxonomy"])
async def get_species(species_id: int):
    """Detalle de una especie con información acústica."""
    if not _db_ok or _engine is None:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")

    import sqlalchemy as sa

    async with _engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    sa.text(
                        """
                SELECT sf.*, s.freq_min_hz, s.freq_max_hz, s.freq_dom_hz
                FROM species_full sf
                JOIN species s ON s.id = sf.species_id
                WHERE sf.species_id = :id
            """
                    ),
                    {"id": species_id},
                )
            )
            .mappings()
            .first()
        )

    if not row:
        raise HTTPException(status_code=404, detail=f"Especie {species_id} no encontrada")
    return SpeciesOut(**{**dict(row), "class": row["class"]})


# ── Grabaciones ───────────────────────────────────────────────────────────────


@app.post("/recordings", response_model=RecordingOut, status_code=201, tags=["Data"])
async def create_recording(recording: RecordingIn, background_tasks: BackgroundTasks):
    """Registra una nueva grabación en la base de datos."""
    if not _db_ok or _engine is None:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")

    import uuid

    import sqlalchemy as sa

    rec_id = str(uuid.uuid4())
    async with _engine.begin() as conn:
        await conn.execute(
            sa.text(
                """
            INSERT INTO recording
              (id, filename, file_path, format, sample_rate, channels,
               duration_s, source, site_id, recorded_at)
            VALUES
              (:id, :filename, :file_path, :format, :sample_rate, :channels,
               :duration_s, :source,
               :site_id::uuid,
               :recorded_at::timestamptz)
        """
            ),
            {
                "id": rec_id,
                "filename": recording.filename,
                "file_path": recording.file_path,
                "format": recording.format,
                "sample_rate": recording.sample_rate,
                "channels": recording.channels,
                "duration_s": recording.duration_s,
                "source": recording.source,
                "site_id": recording.site_id,
                "recorded_at": recording.recorded_at,
            },
        )

    return RecordingOut(
        recording_id=rec_id,
        filename=recording.filename,
        status="registered",
    )


# ── Reportes de biodiversidad ─────────────────────────────────────────────────


@app.get("/reports/site/{site_id}", response_model=list[SiteReport], tags=["Reports"])
async def site_biodiversity_report(
    site_id: str,
    months: int = Query(default=12, ge=1, le=60, description="Últimos N meses"),
):
    """Reporte mensual de biodiversidad acústica para un sitio de monitoreo."""
    if not _db_ok or _engine is None:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")

    import sqlalchemy as sa

    async with _engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    sa.text(
                        """
            SELECT
                site_id::text,
                site_name,
                country,
                to_char(month, 'YYYY-MM') AS month,
                species_richness,
                total_detections,
                avg_confidence,
                species_list
            FROM site_biodiversity_report
            WHERE site_id::text = :sid
              AND month >= NOW() - (INTERVAL '1 month' * :months)
            ORDER BY month DESC
        """
                    ),
                    {"sid": site_id, "months": months},
                )
            )
            .mappings()
            .all()
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Sitio {site_id} no encontrado o sin detecciones en los últimos {months} meses",
        )

    return [SiteReport(**dict(row)) for row in rows]


# ── Modelos disponibles ───────────────────────────────────────────────────────


@app.get("/models", tags=["System"])
async def list_models():
    """Lista los modelos registrados en la base de datos."""
    if not _db_ok or _engine is None:
        return {"models": [], "active_model": model_manager.version}

    import sqlalchemy as sa

    async with _engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    sa.text(
                        """
            SELECT id, name, version, architecture, n_classes,
                   f1_macro, accuracy, is_active, created_at::text
            FROM ml_model ORDER BY created_at DESC LIMIT 20
        """
                    )
                )
            )
            .mappings()
            .all()
        )
    return {"models": [dict(r) for r in rows], "active_model": model_manager.version}


# ─────────────────────────────────────────────────────────────────────────────
# 7. MANEJO DE ERRORES GLOBAL
# ─────────────────────────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Error no manejado en {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor", "type": type(exc).__name__},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=8000,
        reload=True,
        log_level="info",
    )
