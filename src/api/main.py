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
Versión: 1.0.0
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

import numpy as np
import torch
from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, HTTPException,
    Query, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

class Settings:
    APP_NAME     = "BioAcoustics Identification API"
    VERSION      = "1.0.0"
    MODEL_PATH   = os.getenv("MODEL_PATH",  "models/trained/best_efficientnet.pt")
    DB_URL       = os.getenv("DATABASE_URL","postgresql+asyncpg://bio:bio@localhost:5432/bioacoustics")
    MAX_FILE_MB  = int(os.getenv("MAX_FILE_MB", "50"))
    TOP_K        = int(os.getenv("TOP_K",         "5"))
    MIN_PROB     = float(os.getenv("MIN_PROB",  "0.05"))
    DEVICE       = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

settings = Settings()

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELOS PYDANTIC (Schemas)
# ─────────────────────────────────────────────────────────────────────────────

class PredictionItem(BaseModel):
    rank:           int
    species_id:     Optional[int]   = None
    scientific_name: str
    common_name_es: Optional[str]   = None
    acoustic_group: Optional[str]   = None
    probability:    float           = Field(ge=0.0, le=1.0)

class ClassifyResponse(BaseModel):
    request_id:     str
    filename:       Optional[str]   = None
    duration_s:     Optional[float] = None
    inference_ms:   int
    segment_count:  int
    predictions:    List[PredictionItem]
    model_version:  str

class DetectionFilter(BaseModel):
    species_id:   Optional[int]   = None
    acoustic_group: Optional[str] = None
    min_prob:     float           = 0.50
    site_id:      Optional[str]   = None
    limit:        int             = Field(default=50, ge=1, le=500)
    offset:       int             = Field(default=0, ge=0)

class DetectionOut(BaseModel):
    detection_id:     str
    recording:        Optional[str]
    site:             Optional[str]
    t_start_s:        float
    t_end_s:          float
    scientific_name:  Optional[str]
    common_name_es:   Optional[str]
    acoustic_group:   Optional[str]
    probability:      float
    rank:             int
    is_correct:       Optional[bool]
    model:            str
    created_at:       str

class SpeciesOut(BaseModel):
    species_id:     int
    scientific_name: str
    common_name_es: Optional[str]
    common_name_en: Optional[str]
    acoustic_group: str
    iucn_status:    Optional[str]
    family:         Optional[str]
    order:          Optional[str]
    class_:         Optional[str] = Field(None, alias="class")
    freq_min_hz:    Optional[int]
    freq_max_hz:    Optional[int]
    freq_dom_hz:    Optional[int]

class RecordingIn(BaseModel):
    filename:    str
    file_path:   str
    format:      str                    = "WAV"
    sample_rate: int
    channels:    int                    = 1
    duration_s:  float
    source:      str                    = "field"
    site_id:     Optional[str]          = None
    recorded_at: Optional[str]          = None
    latitude:    Optional[float]        = None
    longitude:   Optional[float]        = None
    notes:       Optional[str]          = None

class RecordingOut(BaseModel):
    recording_id: str
    filename:     str
    status:       str

class SiteReport(BaseModel):
    site_id:          str
    site_name:        str
    country:          Optional[str]
    month:            str
    species_richness: int
    total_detections: int
    avg_confidence:   float
    species_list:     List[str]

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    db_connected: bool
    device:       str
    version:      str
    uptime_s:     float

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL MANAGER (Singleton)
# ─────────────────────────────────────────────────────────────────────────────

class ModelManager:
    """
    Gestiona el ciclo de vida del modelo de clasificación.
    Carga el modelo al arrancar y provee inferencia thread-safe.
    """

    def __init__(self):
        self._model      = None
        self._classes: List[str] = []
        self._version:   str     = "unloaded"
        self._loaded:    bool    = False

    def load(self, model_path: str, device: str = "cpu") -> bool:
        """Carga el modelo desde checkpoint. Retorna True si OK."""
        try:
            path = Path(model_path)
            if not path.exists():
                logger.warning(f"Modelo no encontrado: {model_path}. API en modo degradado.")
                return False

            ck = torch.load(model_path, map_location=device)
            n_classes = ck.get("n_classes", 10)

            # Importación lazy para evitar dependencias circulares
            try:
                from src.models.efficientnet_classifier import EfficientNetBioAcoustic, load_efficientnet
                self._model = load_efficientnet(model_path, device=device)
            except ImportError:
                from src.models.cnn_baseline import BioAcousticCNN, load_model
                self._model = load_model(model_path, device=device)

            self._classes  = ck.get("class_names", [str(i) for i in range(n_classes)])
            self._version  = f"v{ck.get('epoch', '?')}_val{ck.get('val_acc', '?'):.3f}"
            self._loaded   = True
            logger.info(f"Modelo cargado: {model_path} | classes={n_classes} | device={device}")
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
    ) -> List[Dict]:
        """
        Inferencia sobre un array de audio crudo.

        Returns:
            Lista de {'rank', 'class', 'probability'}
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Modelo no cargado")

        # Importar procesador en runtime
        from src.audio_processing.preprocessor import AudioPreprocessor, AudioConfig
        cfg  = AudioConfig(sample_rate=sr)
        proc = AudioPreprocessor(cfg)

        # Preprocesamiento básico
        audio = proc.normalize(audio)
        mel   = proc.mel_spectrogram(audio)  # (n_mels, T)

        # Resize a (1, 1, 128, 128)
        spec_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
        import torch.nn.functional as F
        spec_t = F.interpolate(spec_t, size=(128, 128), mode="bilinear", align_corners=False)

        # Normalizar por canal
        mean, std = spec_t.mean(), spec_t.std() + 1e-8
        spec_t    = (spec_t - mean) / std

        probs   = self._model.predict_proba(spec_t).squeeze(0).cpu().numpy()
        top_idx = np.argsort(probs)[::-1][:top_k]

        return [
            {
                "rank":        int(r + 1),
                "class":       self._classes[i] if i < len(self._classes) else str(i),
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
_start_time   = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# 4. BASE DE DATOS (async)
# ─────────────────────────────────────────────────────────────────────────────

# Motor SQLAlchemy async — se inicializa en startup
_engine    = None
_db_ok     = False

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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global _engine, _db_ok
    logger.info(f"Iniciando {settings.APP_NAME} v{settings.VERSION}")

    # Cargar modelo
    model_manager.load(settings.MODEL_PATH, device=settings.DEVICE)

    # Conectar DB
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        _engine = create_async_engine(settings.DB_URL, pool_size=5, max_overflow=10)
        _db_ok  = await get_db_status()
        if _db_ok:
            logger.info("Base de datos conectada.")
        else:
            logger.warning("Base de datos NO disponible. Endpoints de DB deshabilitados.")
    except Exception as e:
        logger.warning(f"DB no disponible: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if _engine:
        await _engine.dispose()
    logger.info("API detenida.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Estado del servicio, modelo y conexión a base de datos."""
    return HealthResponse(
        status       = "ok" if model_manager.loaded else "degraded",
        model_loaded = model_manager.loaded,
        db_connected = _db_ok,
        device       = settings.DEVICE,
        version      = settings.VERSION,
        uptime_s     = round(time.time() - _start_time, 1),
    )


# ── Clasificación por archivo ─────────────────────────────────────────────────

@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
async def classify_audio(
    file: UploadFile = File(..., description="Archivo de audio: WAV, MP3, FLAC, OGG"),
    top_k: int       = Query(default=5, ge=1, le=20, description="Número de predicciones"),
):
    """
    Clasifica el contenido biológico de un archivo de audio.

    Sube un archivo WAV/MP3/FLAC y obtiene las top-K especies más probables
    con su probabilidad de detección.
    """
    # Validar formato
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}
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
        import soundfile as sf
        import librosa

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        duration  = len(audio) / sr

        # Inferencia por ventanas de 3 segundos
        window = 3 * sr
        hop    = int(1.5 * sr)
        all_preds: List[Dict] = []

        if len(audio) <= window:
            preds = model_manager.predict(audio, sr=sr, top_k=top_k)
            all_preds.extend(preds)
            n_segs = 1
        else:
            n_segs = 0
            start  = 0
            while start < len(audio):
                seg = audio[start:start + window]
                if len(seg) < window // 4:
                    break
                preds = model_manager.predict(seg, sr=sr, top_k=top_k)
                all_preds.extend(preds)
                start += hop
                n_segs += 1

        # Agregar predicciones: promedio de probabilidades por clase
        agg: Dict[str, List[float]] = {}
        for pred in all_preds:
            agg.setdefault(pred["class"], []).append(pred["probability"])

        final = sorted(
            [{"class": k, "probability": float(np.mean(v))} for k, v in agg.items()],
            key=lambda x: -x["probability"]
        )[:top_k]

        inference_ms = int((time.time() - t0) * 1000)

        return ClassifyResponse(
            request_id    = str(__import__("uuid").uuid4()),
            filename      = file.filename,
            duration_s    = round(duration, 3),
            inference_ms  = inference_ms,
            segment_count = n_segs,
            predictions   = [
                PredictionItem(
                    rank=i+1,
                    scientific_name=p["class"],
                    probability=p["probability"],
                )
                for i, p in enumerate(final)
            ],
            model_version = model_manager.version,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {e}")
    except Exception as e:
        logger.exception("Error en /classify")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import os as _os
        try:
            _os.unlink(tmp_path)
        except Exception:
            pass


# ── Clasificación por URL ─────────────────────────────────────────────────────

@app.post("/classify/url", response_model=ClassifyResponse, tags=["Inference"])
async def classify_audio_url(
    audio_url: str = Query(..., description="URL pública del archivo de audio"),
    top_k:     int = Query(default=5, ge=1, le=20),
):
    """Clasifica audio referenciado por URL pública (Xeno-canto, iNaturalist, etc.)."""
    import requests as req_lib

    try:
        resp = req_lib.get(audio_url, timeout=30, stream=True)
        resp.raise_for_status()

        ext = Path(audio_url.split("?")[0]).suffix.lower() or ".mp3"
        content = resp.content

        # Crear UploadFile virtual y delegar
        from fastapi import UploadFile as _UF
        from io import BytesIO
        vfile = UploadFile(
            filename=f"remote{ext}",
            file=BytesIO(content),
        )
        return await classify_audio(vfile, top_k=top_k)

    except req_lib.RequestException as e:
        raise HTTPException(status_code=422, detail=f"No se pudo descargar el audio: {e}")


# ── Detecciones ───────────────────────────────────────────────────────────────

@app.get("/detections", response_model=List[DetectionOut], tags=["Data"])
async def list_detections(
    species_id:     Optional[int]   = Query(None),
    acoustic_group: Optional[str]   = Query(None),
    min_prob:       float           = Query(default=0.5, ge=0.0, le=1.0),
    site_id:        Optional[str]   = Query(None),
    limit:          int             = Query(default=50, ge=1, le=500),
    offset:         int             = Query(default=0, ge=0),
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
    params["limit"]  = limit
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
        row = (await conn.execute(
            sa.text("SELECT * FROM detection_summary WHERE detection_id::text = :id"),
            {"id": detection_id},
        )).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail=f"Detección {detection_id} no encontrada")
    return DetectionOut(**dict(row))


# ── Especies ──────────────────────────────────────────────────────────────────

@app.get("/species", response_model=List[SpeciesOut], tags=["Taxonomy"])
async def list_species(
    acoustic_group: Optional[str] = Query(None,
        description="Filtrar por grupo: mammal_bat, amphibian_anura, insect_orthoptera, etc."),
    iucn_status: Optional[str]    = Query(None,
        description="Filtrar por estado IUCN: LC, NT, VU, EN, CR"),
    search: Optional[str]         = Query(None,
        description="Búsqueda por nombre científico o común"),
    limit: int                    = Query(default=100, ge=1, le=1000),
    offset: int                   = Query(default=0, ge=0),
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
    params["limit"]  = limit
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
            freq_min_hz=None, freq_max_hz=None, freq_dom_hz=None,
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
        row = (await conn.execute(
            sa.text("""
                SELECT sf.*, s.freq_min_hz, s.freq_max_hz, s.freq_dom_hz
                FROM species_full sf
                JOIN species s ON s.id = sf.species_id
                WHERE sf.species_id = :id
            """),
            {"id": species_id},
        )).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail=f"Especie {species_id} no encontrada")
    return SpeciesOut(**{**dict(row), "class": row["class"]})


# ── Grabaciones ───────────────────────────────────────────────────────────────

@app.post("/recordings", response_model=RecordingOut, status_code=201, tags=["Data"])
async def create_recording(recording: RecordingIn, background_tasks: BackgroundTasks):
    """Registra una nueva grabación en la base de datos."""
    if not _db_ok or _engine is None:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")

    import sqlalchemy as sa
    import uuid

    rec_id = str(uuid.uuid4())
    async with _engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO recording
              (id, filename, file_path, format, sample_rate, channels,
               duration_s, source, site_id, recorded_at)
            VALUES
              (:id, :filename, :file_path, :format, :sample_rate, :channels,
               :duration_s, :source,
               :site_id::uuid,
               :recorded_at::timestamptz)
        """), {
            "id":          rec_id,
            "filename":    recording.filename,
            "file_path":   recording.file_path,
            "format":      recording.format,
            "sample_rate": recording.sample_rate,
            "channels":    recording.channels,
            "duration_s":  recording.duration_s,
            "source":      recording.source,
            "site_id":     recording.site_id,
            "recorded_at": recording.recorded_at,
        })

    return RecordingOut(
        recording_id=rec_id,
        filename=recording.filename,
        status="registered",
    )


# ── Reportes de biodiversidad ─────────────────────────────────────────────────

@app.get("/reports/site/{site_id}", response_model=List[SiteReport], tags=["Reports"])
async def site_biodiversity_report(
    site_id:  str,
    months:   int = Query(default=12, ge=1, le=60, description="Últimos N meses"),
):
    """Reporte mensual de biodiversidad acústica para un sitio de monitoreo."""
    if not _db_ok or _engine is None:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")

    import sqlalchemy as sa

    async with _engine.connect() as conn:
        rows = (await conn.execute(sa.text("""
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
              AND month >= NOW() - INTERVAL ':months months'
            ORDER BY month DESC
        """), {"sid": site_id, "months": months})).mappings().all()

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
        rows = (await conn.execute(sa.text("""
            SELECT id, name, version, architecture, n_classes,
                   f1_macro, accuracy, is_active, created_at::text
            FROM ml_model ORDER BY created_at DESC LIMIT 20
        """))).mappings().all()
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
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
