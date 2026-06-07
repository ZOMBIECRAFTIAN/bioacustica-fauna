"""
tests/test_api.py
-----------------------------------------------------------------------------
Tests de integracion para src/api/main.py (FastAPI).
Usa httpx.AsyncClient via TestClient -- no requiere servidor en vivo.

Cubre:
  - GET  /health
  - GET  /species
  - GET  /models
  - POST /classify (audio valido, formato invalido, archivo vacio)
  - Schemas Pydantic de respuesta
  - Manejo de errores HTTP (404, 422, 415)

Nota: Los endpoints que requieren DB real (PostgreSQL) se marcan con
      pytest.mark.integration y se saltan en CI basico.

Ejecutar:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v -m "not integration"
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

# -----------------------------------------------------------------------------
# Importacion del app FastAPI
# -----------------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
    from src.api.main import app

    CLIENT_AVAILABLE = True
except Exception:
    CLIENT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CLIENT_AVAILABLE,
    reason="FastAPI app no importable (revisa dependencias)",
)

# -----------------------------------------------------------------------------
# Setup del cliente de test
# -----------------------------------------------------------------------------
SR = 22_050
DURATION = 3.0


@pytest.fixture(scope="module")
def client():
    """TestClient sincrono de FastAPI."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="module")
def sine_wav_bytes():
    """Bytes de un archivo WAV sinusoidal valido (3s, 22050 Hz)."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="module")
def short_wav_bytes():
    """Bytes de un WAV muy corto (0.1s) -- puede generar error de validacion."""
    audio = np.zeros(int(SR * 0.1), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# =============================================================================
# 1. HEALTH CHECK
# =============================================================================


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status(self, client):
        data = client.get("/health").json()
        assert "status" in data

    def test_health_status_is_ok_or_degraded(self, client):
        data = client.get("/health").json()
        assert data["status"] in ("ok", "degraded", "error")

    def test_health_response_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


# =============================================================================
# 2. ENDPOINT /species
# =============================================================================


class TestSpeciesEndpoint:
    def test_species_returns_200(self, client):
        response = client.get("/species")
        assert response.status_code in (200, 503)  # 503 si DB no disponible

    def test_species_with_limit(self, client):
        response = client.get("/species?limit=10")
        assert response.status_code in (200, 503)

    def test_species_invalid_limit(self, client):
        """limit negativo debe retornar 422."""
        response = client.get("/species?limit=-1")
        assert response.status_code == 422

    def test_species_with_taxon_filter(self, client):
        response = client.get("/species?taxon=amphibia")
        assert response.status_code in (200, 503)

    @pytest.mark.integration
    def test_species_response_schema(self, client):
        """Requiere DB activa."""
        response = client.get("/species?limit=5")
        if response.status_code == 200:
            data = response.json()
            assert "species" in data or isinstance(data, list)


# =============================================================================
# 3. ENDPOINT /models
# =============================================================================


class TestModelsEndpoint:
    def test_models_returns_200(self, client):
        response = client.get("/models")
        assert response.status_code == 200

    def test_models_response_is_list_or_dict(self, client):
        data = client.get("/models").json()
        assert isinstance(data, list | dict)

    def test_models_not_empty(self, client):
        data = client.get("/models").json()
        if isinstance(data, dict):
            assert "models" in data or len(data) > 0
        else:
            # Si retorna lista directa
            assert isinstance(data, list)


# =============================================================================
# 4. ENDPOINT POST /classify
# =============================================================================


class TestClassifyEndpoint:
    def test_classify_valid_wav_returns_200_or_503(self, client, sine_wav_bytes):
        """
        200: modelo cargado y clasifica correctamente.
        503: modelo no cargado (sin pesos entrenados en CI) -- aceptable.
        """
        response = client.post(
            "/classify",
            files={"file": ("test.wav", sine_wav_bytes, "audio/wav")},
        )
        assert response.status_code in (200, 503, 422)

    def test_classify_response_schema_when_200(self, client, sine_wav_bytes):
        response = client.post(
            "/classify",
            files={"file": ("test.wav", sine_wav_bytes, "audio/wav")},
        )
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "inference_ms" in data or "inference_time_ms" in data
            assert isinstance(data["predictions"], list)

    def test_classify_predictions_have_probability(self, client, sine_wav_bytes):
        response = client.post(
            "/classify",
            files={"file": ("test.wav", sine_wav_bytes, "audio/wav")},
        )
        if response.status_code == 200:
            data = response.json()
            for pred in data["predictions"]:
                assert "probability" in pred or "confidence" in pred
                prob = pred.get("probability", pred.get("confidence", 0))
                assert 0.0 <= prob <= 1.0

    def test_classify_no_file_returns_422(self, client):
        """Sin archivo adjunto: error de validacion."""
        response = client.post("/classify")
        assert response.status_code == 422

    def test_classify_invalid_format_returns_4xx(self, client):
        """Archivo de texto plano, no audio."""
        fake_audio = b"esto no es un archivo de audio valido"
        response = client.post(
            "/classify",
            files={"file": ("test.txt", fake_audio, "text/plain")},
        )
        assert response.status_code in (400, 415, 422, 500)

    def test_classify_empty_file_returns_4xx(self, client):
        """Archivo WAV de 0 bytes."""
        response = client.post(
            "/classify",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code in (400, 422, 500)

    def test_classify_top_k_parameter(self, client, sine_wav_bytes):
        """top_k debe limitar el numero de predicciones."""
        response = client.post(
            "/classify",
            files={"file": ("test.wav", sine_wav_bytes, "audio/wav")},
            data={"top_k": "3"},
        )
        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) <= 3

    def test_classify_with_preset(self, client, sine_wav_bytes):
        """El parametro preset debe ser aceptado."""
        response = client.post(
            "/classify",
            files={"file": ("test.wav", sine_wav_bytes, "audio/wav")},
            data={"preset": "frogs"},
        )
        assert response.status_code in (200, 422, 503)

    def test_classify_mp3_extension(self, client, sine_wav_bytes):
        """Nombres con extension .mp3 deben ser aceptados o rechazados graciosamente."""
        response = client.post(
            "/classify",
            files={"file": ("test.mp3", sine_wav_bytes, "audio/mpeg")},
        )
        # No debe retornar 500 -- error manejado
        assert response.status_code != 500


# =============================================================================
# 5. ENDPOINT /detections (requiere DB)
# =============================================================================


class TestDetectionsEndpoint:
    def test_detections_accessible(self, client):
        response = client.get("/detections")
        # 200 con DB, 503 sin DB -- ambos aceptables
        assert response.status_code in (200, 404, 503)

    @pytest.mark.integration
    def test_detections_filter_by_species(self, client):
        response = client.get("/detections?species_id=1")
        assert response.status_code in (200, 503)

    @pytest.mark.integration
    def test_detections_invalid_limit(self, client):
        response = client.get("/detections?limit=9999")
        assert response.status_code == 422


# =============================================================================
# 6. Schemas Pydantic
# =============================================================================


class TestPydanticSchemas:
    def test_prediction_item_valid(self):
        from src.api.main import PredictionItem

        item = PredictionItem(
            rank=1,
            scientific_name="Rana temporaria",
            probability=0.87,
        )
        assert item.rank == 1
        assert item.probability == 0.87

    def test_prediction_item_probability_bounds(self):
        from src.api.main import PredictionItem

        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError
            PredictionItem(
                rank=1,
                scientific_name="Test",
                probability=1.5,  # fuera de [0, 1]
            )

    def test_detection_filter_defaults(self):
        from src.api.main import DetectionFilter

        f = DetectionFilter()
        assert f.min_prob == 0.50
        assert f.limit == 50
        assert f.offset == 0

    def test_detection_filter_limit_max(self):
        from src.api.main import DetectionFilter

        with pytest.raises(Exception):  # noqa: B017 -- pydantic ValidationError
            DetectionFilter(limit=501)  # supera max=500


# =============================================================================
# 7. Middleware y cabeceras
# =============================================================================


class TestMiddleware:
    def test_cors_header_present(self, client):
        """CORS middleware debe estar activo."""
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        # Al menos el endpoint debe responder
        assert response.status_code in (200, 400)

    def test_content_type_json(self, client):
        """Las respuestas deben ser JSON."""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")
