"""
data/dataset_builder.py
─────────────────────────────────────────────────────────────────────────────
Descarga y estructuración automatizada de datos bioacústicos desde:
  - Xeno-canto API v2 (aves + referencia cruzada)
  - iNaturalist API v1 (anfibios, insectos, mamíferos, reptiles)
  - FrogID (anuros, Australia) — descarga de registros públicos
  - GBIF Occurrence API (multitaxon)

Genera la estructura de directorios requerida por SpectrogramDataset:
    data/raw/{clase}/{id}.wav

Dependencias:
    pip install requests tqdm soundfile librosa

Autor: Ian
Versión: 1.0.0
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


MEXICO_COUNTRY = "Mexico"
MEXICO_INAT_PLACE_ID = 6793


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLIENTE HTTP CON RETRY Y RATE LIMITING
# ─────────────────────────────────────────────────────────────────────────────


def _make_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    """Session con reintentos automáticos y backoff exponencial."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "BioAcousticsAI/1.0 (research project)"})
    return session


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURACIÓN DE DESCARGA
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DownloadConfig:
    output_dir: str = "data/raw"  # directorio raíz de salida
    max_per_class: int = 500  # máximo de archivos por clase
    min_quality: str = "B"  # calidad mínima (A/B para Xeno-canto)
    min_duration: float = 1.0  # segundos mínimos de audio
    max_duration: float = 60.0  # segundos máximos
    target_sr: int = 44_100  # resamplear a este SR tras descarga
    formats: list[str] = field(default_factory=lambda: ["mp3", "wav", "ogg", "flac"])
    dry_run: bool = False  # si True, solo lista sin descargar
    delay_s: float = 0.5  # pausa entre requests (rate limiting)
    save_metadata: bool = True  # guardar metadata JSON por archivo
    validate_downloads: bool = True  # descartar audios ilegibles tras descarga
    quarantine_dir: str = "data/quarantine"  # destino de audios corruptos
    country: str | None = None  # filtro regional para fuentes que lo soportan
    inat_place_id: int | None = None  # place_id opcional de iNaturalist
    inat_quality_grade: str = "research"  # research, needs_id o casual
    profile: str = "default"  # perfil de targets usado en el build
    xeno_canto_api_key: str | None = None  # requerido por Xeno-canto API v3


def _parse_xc_duration(raw: str | int | float | None) -> float:
    """Convierte duraciones Xeno-canto tipo '1:23' o '12' a segundos."""
    if raw is None:
        return 0.0
    text = str(raw).strip()
    if not text:
        return 0.0
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return 0.0

    parts = text.split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return 0.0
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def _xc_quality_filter(min_quality: str) -> str:
    """Convierte calidad mínima A-E a la sintaxis inclusiva de Xeno-canto v3."""
    if "+" in min_quality or " " in min_quality:
        return min_quality
    order = ["A", "B", "C", "D", "E"]
    quality = min_quality.upper()
    if quality not in order:
        return '">C"'
    if quality == "A":
        return "A"
    if quality == "E":
        return '">F"'
    # En la escala de calidad de XC, A/B se expresa como q:">C".
    return f'">{order[order.index(quality) + 1]}"'


def _xc_country_tag(country: str) -> str:
    """Normaliza el filtro de país para Xeno-canto v3."""
    value = country.strip().lower()
    if " " in value:
        value = f'"{value}"'
    return f"cnt:{value}"


def _xc_group_tag(acoustic_group: str | None) -> str | None:
    """Mapea grupos internos al tag grp de Xeno-canto v3."""
    if not acoustic_group:
        return None
    mapping = {
        "birds": "birds",
        "aves": "birds",
        "frogs": "frogs",
        "bats": "bats",
        "insects": "grasshoppers",
        "mammals": '"land mammals"',
    }
    value = mapping.get(acoustic_group)
    return f"grp:{value}" if value else None


def _xc_species_query(scientific_name: str) -> str:
    """Convierte 'Genus species' a sp:\"Genus species\" para API v3."""
    text = scientific_name.strip()
    if ":" in text:
        return text
    return f'sp:"{text}"'


def _audio_duration(path: Path) -> float:
    """Retorna duración en segundos o lanza excepción si el audio no es legible."""
    import soundfile as sf

    return float(sf.info(str(path)).duration)


def _is_audio_valid(path: Path, min_s: float = 0.0, max_s: float | None = None) -> tuple[bool, str]:
    """Valida que el archivo sea legible y esté dentro del rango de duración."""
    try:
        duration = _audio_duration(path)
    except Exception as e:
        return False, f"read_error: {e}"
    if duration < min_s:
        return False, f"duration={duration:.2f}s < {min_s:.2f}s"
    if max_s is not None and duration > max_s:
        return False, f"duration={duration:.2f}s > {max_s:.2f}s"
    return True, f"duration={duration:.2f}s"


def _discard_invalid_download(path: Path, reason: str) -> None:
    """Elimina una descarga recién creada que no sirve como audio."""
    logger.warning(f"  ✗ Audio inválido descartado: {path.name} ({reason})")
    try:
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"  No se pudo limpiar {path}: {e}")


def _move_with_metadata(path: Path, source_root: Path, quarantine_root: Path) -> Path:
    """Mueve un archivo corrupto y su JSON hermano a cuarentena preservando ruta relativa."""
    rel = path.relative_to(source_root)
    target = quarantine_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        stem = target.stem
        target = target.with_name(f"{stem}_{int(time.time())}{target.suffix}")
    shutil.move(str(path), str(target))

    meta = path.with_suffix(".json")
    if meta.exists():
        meta_target = target.with_suffix(".json")
        shutil.move(str(meta), str(meta_target))
    return target


# ─────────────────────────────────────────────────────────────────────────────
# 3. XENO-CANTO API v2
# ─────────────────────────────────────────────────────────────────────────────


class XenoCantoDownloader:
    """
    Descarga grabaciones desde Xeno-canto API v2.
    Documentación: https://xeno-canto.org/explore/api

    Parámetros de búsqueda soportados:
        q:    calidad (A, B, C, D, E)
        type: tipo de sonido (call, song, alarm, etc.)
        cnt:  país
        len:  duración en segundos (e.g., "5-30")
    """

    BASE_URL = "https://xeno-canto.org/api/3/recordings"

    def __init__(self, config: DownloadConfig):
        self.cfg = config
        self.session = _make_session()
        self.api_key = config.xeno_canto_api_key or os.getenv("XENO_CANTO_API_KEY")
        if self.api_key:
            logger.info("Xeno-canto API key detectada.")
        else:
            logger.warning(
                "Xeno-canto API key no detectada. Define XENO_CANTO_API_KEY "
                "o usa --xeno-canto-api-key."
            )

    def search(
        self,
        query: str,
        quality: str = "A+B",
        type_: str = "",
        country: str | None = None,
        acoustic_group: str | None = None,
        min_len: float = 1.0,
        max_len: float = 60.0,
    ) -> Iterator[dict]:
        """
        Itera sobre todos los resultados de una búsqueda en Xeno-canto.

        Args:
            query:   Consulta libre (e.g., "Bufo bufo" o "genus:Hyla")
            quality: Filtro de calidad ("A", "A+B", "A+B+C")

        Yields:
            Diccionario con metadata de cada grabación.
        """
        query_parts = [query, f"q:{quality}"]
        group_tag = _xc_group_tag(acoustic_group)
        if group_tag:
            query_parts.append(group_tag)
        if country:
            query_parts.append(_xc_country_tag(country))
        params = {"query": " ".join(query_parts)}
        if self.api_key:
            params["key"] = self.api_key
        if type_:
            params["query"] += f" type:{type_}"
        page = 1
        total_pages = 1

        while page <= total_pages:
            params["page"] = page
            url = f"{self.BASE_URL}?{urlencode(params)}"
            logger.debug(f"XenoCanto GET: {url}")

            resp = self.session.get(url, timeout=30)
            if resp.status_code in {401, 403} and not self.api_key:
                raise RuntimeError(
                    "Xeno-canto API v3 requiere API key. Define XENO_CANTO_API_KEY "
                    "o usa --xeno-canto-api-key."
                )
            if resp.status_code == 404:
                logger.warning("Xeno-canto sin resultados o endpoint no disponible: %s", url)
                return
            resp.raise_for_status()
            data = resp.json()

            if page == 1:
                total_pages = int(data.get("numPages", 1))
                total_recs = int(data.get("numRecordings", 0))
                logger.info(
                    f"Xeno-canto '{query}': {total_recs} grabaciones en {total_pages} páginas"
                )

            for rec in data.get("recordings", []):
                dur = _parse_xc_duration(rec.get("length"))
                if min_len <= dur <= max_len:
                    yield rec

            page += 1
            time.sleep(self.cfg.delay_s)

    def download_species(
        self,
        scientific_name: str,
        class_label: str,
        quality: str | None = None,
        country: str | None = None,
        acoustic_group: str | None = None,
    ) -> list[Path]:
        """
        Descarga grabaciones de una especie hasta max_per_class.

        Returns:
            Lista de rutas a archivos descargados.
        """
        out_dir = Path(self.cfg.output_dir) / class_label
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = list(out_dir.glob("*.mp3")) + list(out_dir.glob("*.wav"))
        if len(existing) >= self.cfg.max_per_class:
            logger.info(f"  {class_label}: ya tiene {len(existing)} archivos, saltando.")
            return existing

        downloaded: list[Path] = []
        n_existing = len(existing)

        quality = quality or _xc_quality_filter(self.cfg.min_quality)
        for rec in self.search(
            _xc_species_query(scientific_name),
            quality=quality,
            country=country or self.cfg.country,
            acoustic_group=acoustic_group,
            min_len=self.cfg.min_duration,
            max_len=self.cfg.max_duration,
        ):
            if n_existing + len(downloaded) >= self.cfg.max_per_class:
                break

            rec_id = rec.get("id", "")
            raw_file_url = rec.get("file", "")
            if raw_file_url.startswith("//"):
                file_url = "https:" + raw_file_url
            else:
                file_url = raw_file_url
            ext = rec.get("file-name", "audio.mp3").rsplit(".", 1)[-1].lower()
            out_path = out_dir / f"xc_{rec_id}.{ext}"

            if out_path.exists():
                continue

            if self.cfg.dry_run:
                logger.info(f"  [DRY] {out_path.name} | {rec.get('length')}s")
                downloaded.append(out_path)
                continue

            try:
                download_params = {"key": self.api_key} if self.api_key else None
                resp = self.session.get(
                    file_url,
                    params=download_params,
                    timeout=60,
                    stream=True,
                )
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)

                if self.cfg.validate_downloads:
                    ok, reason = _is_audio_valid(out_path, min_s=self.cfg.min_duration)
                    if not ok:
                        _discard_invalid_download(out_path, reason)
                        continue

                if self.cfg.save_metadata:
                    meta_path = out_path.with_suffix(".json")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "source": "xeno-canto",
                                "id": rec_id,
                                "species": scientific_name,
                                "common_name": rec.get("en", ""),
                                "source_url": f"https://xeno-canto.org/{rec_id}",
                                "download_url": file_url,
                                "author": rec.get("rec", ""),
                                "country": rec.get("cnt", ""),
                                "lat": rec.get("lat", ""),
                                "lng": rec.get("lng", ""),
                                "date": rec.get("date", ""),
                                "quality": rec.get("q", ""),
                                "duration": rec.get("length", ""),
                                "license": rec.get("lic", ""),
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )

                downloaded.append(out_path)
                logger.debug(f"  ✓ {out_path.name}")

            except Exception as e:
                logger.warning(f"  ✗ Error descargando {rec_id}: {e}")

            time.sleep(self.cfg.delay_s)

        action = "candidatos encontrados" if self.cfg.dry_run else "archivos nuevos descargados"
        logger.info(f"  {class_label}: {len(downloaded)} {action}.")
        return downloaded


# ─────────────────────────────────────────────────────────────────────────────
# 4. INATURALIST API v1
# ─────────────────────────────────────────────────────────────────────────────


class iNaturalistDownloader:
    """
    Descarga observaciones con sonido desde iNaturalist API.
    Endpoint: https://api.inaturalist.org/v1/observations

    Taxonomía objetivo:
      - Amphibia: taxon_id=20978
      - Insecta:  taxon_id=47158
      - Mammalia: taxon_id=40151
      - Reptilia: taxon_id=26036
    """

    BASE_URL = "https://api.inaturalist.org/v1/observations"
    TAXON_IDS = {
        "amphibia": 20978,
        "insecta": 47158,
        "mammalia": 40151,
        "reptilia": 26036,
        "aves": 3,
    }

    def __init__(self, config: DownloadConfig, api_token: str | None = None):
        self.cfg = config
        self.session = _make_session()
        if api_token:
            self.session.headers["Authorization"] = f"Bearer {api_token}"

    def search_sounds(
        self,
        taxon_id: int,
        taxon_name: str | None = None,
        place_id: int | None = None,
        quality_grade: str | None = None,
        per_page: int = 200,
    ) -> Iterator[dict]:
        """
        Itera sobre observaciones con sonido en iNaturalist.

        Yields:
            Diccionario de observación con campo 'sounds' no vacío.
        """
        params = {
            "taxon_id": taxon_id,
            "sounds": "true",
            "quality_grade": quality_grade or self.cfg.inat_quality_grade,
            "per_page": per_page,
            "order": "desc",
            "order_by": "id",
        }
        if taxon_name:
            params["taxon_name"] = taxon_name
        if place_id:
            params["place_id"] = place_id

        page = 1
        while True:
            params["page"] = page
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            total = data.get("total_results", 0)
            if page == 1:
                logger.info(f"iNat taxon_id={taxon_id}: {total} observaciones con sonido")

            for obs in results:
                if obs.get("sounds"):
                    yield obs

            if len(results) < per_page:
                break
            page += 1
            time.sleep(self.cfg.delay_s)

    def download_taxon(
        self,
        taxon_group: str,
        taxon_name: str | None = None,
        class_label: str | None = None,
        place_id: int | None = None,
    ) -> list[Path]:
        """
        Descarga sonidos de un grupo taxonómico desde iNaturalist.

        Args:
            taxon_group: Clave en TAXON_IDS ("amphibia", "insecta", etc.)
            taxon_name:  Nombre científico para filtro adicional.
            class_label: Nombre del subdirectorio de clase.
        """
        taxon_id = self.TAXON_IDS.get(taxon_group.lower())
        if taxon_id is None:
            raise ValueError(
                f"Grupo no reconocido: {taxon_group}. Opciones: {list(self.TAXON_IDS)}"
            )

        label = class_label or (taxon_name or taxon_group).replace(" ", "_").lower()
        out_dir = Path(self.cfg.output_dir) / label
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = [
            f for f in out_dir.iterdir() if f.suffix.lower().lstrip(".") in self.cfg.formats
        ]
        if len(existing) >= self.cfg.max_per_class:
            logger.info(f"  {label}: ya tiene {len(existing)} archivos, saltando iNat.")
            return existing

        downloaded: list[Path] = []

        for obs in self.search_sounds(
            taxon_id,
            taxon_name=taxon_name,
            place_id=place_id or self.cfg.inat_place_id,
        ):
            if len(existing) + len(downloaded) >= self.cfg.max_per_class:
                break

            obs_id = obs.get("id", "")
            taxon = obs.get("taxon") or {}
            user = obs.get("user") or {}
            location = obs.get("location") or ""
            lat, lng = "", ""
            if "," in location:
                lat, lng = (p.strip() for p in location.split(",", 1))
            for snd in obs.get("sounds", []):
                url = snd.get("file_url", "")
                if not url:
                    continue
                ext = urlparse(url).path.rsplit(".", 1)[-1].lower() or "mp3"
                snd_id = snd.get("id", obs_id)
                out_path = out_dir / f"inat_{snd_id}.{ext}"

                if out_path.exists() or ext not in self.cfg.formats:
                    continue
                if self.cfg.dry_run:
                    logger.info(f"  [DRY] {out_path.name}")
                    downloaded.append(out_path)
                    continue

                try:
                    resp = self.session.get(url, timeout=60, stream=True)
                    resp.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)

                    if self.cfg.validate_downloads:
                        ok, reason = _is_audio_valid(out_path, min_s=self.cfg.min_duration)
                        if not ok:
                            _discard_invalid_download(out_path, reason)
                            continue

                    if self.cfg.save_metadata:
                        meta_path = out_path.with_suffix(".json")
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "source": "inaturalist",
                                    "obs_id": obs_id,
                                    "sound_id": snd_id,
                                    "species": taxon.get("name", taxon_name or ""),
                                    "common_name": taxon.get("preferred_common_name", ""),
                                    "source_url": obs.get("uri", ""),
                                    "download_url": url,
                                    "author": user.get("login", ""),
                                    "observed_on": obs.get("observed_on", ""),
                                    "country": self.cfg.country or "",
                                    "lat": lat,
                                    "lng": lng,
                                    "quality_grade": obs.get("quality_grade", ""),
                                    "license": snd.get("license_code")
                                    or obs.get("license_code", ""),
                                    "attribution": snd.get("attribution", ""),
                                },
                                f,
                                indent=2,
                                ensure_ascii=False,
                            )

                    downloaded.append(out_path)
                    logger.debug(f"  ✓ {out_path.name}")

                except Exception as e:
                    logger.warning(f"  ✗ iNat {snd_id}: {e}")

            if len(existing) + len(downloaded) >= self.cfg.max_per_class:
                break
            time.sleep(self.cfg.delay_s)

        logger.info(f"  {label}: {len(downloaded)} archivos descargados (iNat)")
        return downloaded


# ─────────────────────────────────────────────────────────────────────────────
# 5. GBIF OCCURRENCE API
# ─────────────────────────────────────────────────────────────────────────────


class GBIFDownloader:
    """
    Accede al catálogo de sonidos en GBIF (Darwin Core multimedia).
    Endpoint: https://api.gbif.org/v1/occurrence/search

    Requiere filtrar por mediaType=Sound y usar los URLs de media.
    """

    BASE_URL = "https://api.gbif.org/v1/occurrence/search"
    CLASS_KEYS = {
        "Aves": 212,
        "Mammalia": 359,
        "Amphibia": 131,
        "Reptilia": 358,
        "Insecta": 216,
    }

    def __init__(self, config: DownloadConfig):
        self.cfg = config
        self.session = _make_session()

    def search_sounds(
        self,
        class_key: int | None = None,
        scientific_name: str | None = None,
        country: str | None = None,
        limit: int = 300,
    ) -> Iterator[dict]:
        """Itera sobre ocurrencias con media de tipo 'Sound'."""
        offset = 0
        while True:
            params = {
                "mediaType": "Sound",
                "limit": min(limit, 300),
                "offset": offset,
            }
            if class_key:
                params["classKey"] = class_key
            if scientific_name:
                params["scientificName"] = scientific_name
            if country:
                params["country"] = "MX" if country.lower() in {"mexico", "méxico"} else country
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            for rec in results:
                media = [m for m in rec.get("media", []) if m.get("type") == "Sound"]
                if media:
                    rec["_sounds"] = media
                    yield rec

            if data.get("endOfRecords", True):
                break
            offset += len(results)
            time.sleep(self.cfg.delay_s)

    def download_species(
        self,
        scientific_name: str,
        class_label: str,
        country: str | None = None,
    ) -> list[Path]:
        """Descarga audios enlazados en GBIF para una especie."""
        out_dir = Path(self.cfg.output_dir) / class_label
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = [
            f for f in out_dir.iterdir() if f.suffix.lower().lstrip(".") in self.cfg.formats
        ]
        if len(existing) >= self.cfg.max_per_class:
            logger.info(f"  {class_label}: ya tiene {len(existing)} archivos, saltando GBIF.")
            return existing

        downloaded: list[Path] = []
        for rec in self.search_sounds(
            scientific_name=scientific_name,
            country=country or self.cfg.country,
        ):
            if len(existing) + len(downloaded) >= self.cfg.max_per_class:
                break

            gbif_id = rec.get("gbifID") or rec.get("key") or ""
            media = rec.get("_sounds", [])
            for idx, item in enumerate(media):
                if len(existing) + len(downloaded) >= self.cfg.max_per_class:
                    break
                url = item.get("identifier") or item.get("references")
                if not url:
                    continue
                path_ext = urlparse(url).path.rsplit(".", 1)[-1].lower()
                ext = path_ext if path_ext in self.cfg.formats else "mp3"
                out_path = out_dir / f"gbif_{gbif_id}_{idx}.{ext}"

                if out_path.exists():
                    continue
                if self.cfg.dry_run:
                    logger.info(f"  [DRY] {out_path.name}")
                    downloaded.append(out_path)
                    continue

                try:
                    resp = self.session.get(url, timeout=60, stream=True)
                    resp.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)

                    if self.cfg.validate_downloads:
                        ok, reason = _is_audio_valid(out_path, min_s=self.cfg.min_duration)
                        if not ok:
                            _discard_invalid_download(out_path, reason)
                            continue

                    if self.cfg.save_metadata:
                        with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "source": "gbif",
                                    "gbif_id": gbif_id,
                                    "species": rec.get("species") or scientific_name,
                                    "source_url": rec.get("references", ""),
                                    "download_url": url,
                                    "dataset": rec.get("datasetName", ""),
                                    "publisher": rec.get("publisher", ""),
                                    "author": item.get("creator") or rec.get("recordedBy", ""),
                                    "license": item.get("license") or rec.get("license", ""),
                                    "country": rec.get(
                                        "country", country or self.cfg.country or ""
                                    ),
                                    "lat": rec.get("decimalLatitude", ""),
                                    "lng": rec.get("decimalLongitude", ""),
                                    "event_date": rec.get("eventDate", ""),
                                },
                                f,
                                indent=2,
                                ensure_ascii=False,
                            )

                    downloaded.append(out_path)
                    logger.debug(f"  ✓ {out_path.name}")
                except Exception as e:
                    logger.warning(f"  ✗ GBIF {gbif_id}: {e}")

                time.sleep(self.cfg.delay_s)

        logger.info(f"  {class_label}: {len(downloaded)} archivos descargados (GBIF)")
        return downloaded


# ─────────────────────────────────────────────────────────────────────────────
# 6. ORQUESTADOR: DatasetBuilder
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TaxonTarget:
    """Define un target de descarga para una clase específica del dataset."""

    class_label: str  # nombre del directorio (e.g., "bufo_bufo")
    scientific_name: str  # nombre científico
    acoustic_group: str  # "frogs", "bats", "insects", "mammals", "reptiles"
    sources: list[str] = field(default_factory=lambda: ["xeno-canto", "inaturalist"])
    xc_query: str | None = None  # query personalizada para XenoCanto
    inat_taxon_name: str | None = None  # nombre para filtro iNat


class DatasetBuilder:
    """
    Orquestador central para construcción del dataset multitaxonómico.

    Usa XenoCanto + iNaturalist en conjunto para maximizar cobertura.
    Registra un manifiesto JSON con la proveniencia de cada archivo.

    Ejemplo de uso:
        cfg     = DownloadConfig(max_per_class=300, output_dir="data/raw")
        builder = DatasetBuilder(cfg)
        targets = [
            TaxonTarget("bufo_bufo",        "Bufo bufo",         "frogs"),
            TaxonTarget("hyla_arborea",     "Hyla arborea",      "frogs"),
            TaxonTarget("myotis_lucifugus", "Myotis lucifugus",  "bats"),
            TaxonTarget("gryllus_campestris","Gryllus campestris","insects"),
        ]
        builder.build(targets)
    """

    def __init__(self, config: DownloadConfig, inat_token: str | None = None):
        self.cfg = config
        self.xc = XenoCantoDownloader(config)
        self.inat = iNaturalistDownloader(config, api_token=inat_token)
        self.gbif = GBIFDownloader(config)
        self.manifest: list[dict] = []
        self.out_dir = Path(config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # Mapa grupo → taxon_id iNat
    _INAT_GROUP_MAP = {
        "frogs": ("amphibia", 20978),
        "bats": ("mammalia", 40151),
        "mammals": ("mammalia", 40151),
        "insects": ("insecta", 47158),
        "reptiles": ("reptilia", 26036),
        "birds": ("aves", 3),
        "aves": ("aves", 3),
    }

    def build(self, targets: list[TaxonTarget]) -> dict:
        """
        Descarga todos los targets y genera el manifiesto del dataset.

        Returns:
            Resumen con conteo por clase.
        """
        summary: dict[str, int] = {}

        for target in targets:
            logger.info(f"\n{'─'*60}")
            logger.info(f"Procesando: {target.scientific_name} → {target.class_label}")
            files: list[Path] = []

            # ── Xeno-canto ────────────────────────────────────
            if "xeno-canto" in target.sources:
                query = target.xc_query or target.scientific_name
                try:
                    new = self.xc.download_species(
                        query,
                        target.class_label,
                        country=self.cfg.country,
                        acoustic_group=target.acoustic_group,
                    )
                    files.extend(new)
                except Exception as e:
                    logger.error(f"XenoCanto error para {target.class_label}: {e}")

            # ── iNaturalist ───────────────────────────────────
            if "inaturalist" in target.sources and len(files) < self.cfg.max_per_class:
                group_key = target.acoustic_group
                if group_key in self._INAT_GROUP_MAP:
                    try:
                        new = self.inat.download_taxon(
                            taxon_group=self._INAT_GROUP_MAP[group_key][0],
                            taxon_name=target.inat_taxon_name or target.scientific_name,
                            class_label=target.class_label,
                            place_id=self.cfg.inat_place_id,
                        )
                        files.extend(new)
                    except Exception as e:
                        logger.error(f"iNat error para {target.class_label}: {e}")

            # ── GBIF ──────────────────────────────────────────
            if "gbif" in target.sources and len(files) < self.cfg.max_per_class:
                try:
                    new = self.gbif.download_species(
                        target.scientific_name,
                        target.class_label,
                        country=self.cfg.country,
                    )
                    files.extend(new)
                except Exception as e:
                    logger.error(f"GBIF error para {target.class_label}: {e}")

            class_dir = self.out_dir / target.class_label
            n = (
                len(
                    [
                        f
                        for f in class_dir.iterdir()
                        if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
                    ]
                )
                if class_dir.exists()
                else 0
            )
            summary[target.class_label] = n
            self.manifest.append(
                {
                    "class_label": target.class_label,
                    "scientific_name": target.scientific_name,
                    "acoustic_group": target.acoustic_group,
                    "n_files": n,
                    "sources": target.sources,
                }
            )

        # Guardar manifiesto
        manifest_path = self.out_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "max_per_class": self.cfg.max_per_class,
                        "min_quality": self.cfg.min_quality,
                        "country": self.cfg.country,
                        "inat_place_id": self.cfg.inat_place_id,
                        "inat_quality_grade": self.cfg.inat_quality_grade,
                        "profile": self.cfg.profile,
                    },
                    "classes": self.manifest,
                    "total_classes": len(self.manifest),
                    "total_files": sum(summary.values()),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Dataset completado: {sum(summary.values())} archivos en {len(summary)} clases"
        )
        logger.info(f"Manifiesto guardado: {manifest_path}")
        return summary

    def validate_dataset(self, quarantine_corrupt: bool = False) -> dict:
        """
        Valida la integridad del dataset: archivos corruptos, clases desbalanceadas.

        Returns:
            Diccionario con estadísticas y lista de archivos con problemas.
        """
        stats: dict[str, dict] = {}
        corrupt: list[str] = []
        quarantined: list[str] = []
        out_of_range: list[dict] = []
        quarantine_root = Path(self.cfg.quarantine_dir) / "corrupt"

        for class_dir in sorted(self.out_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            files = [
                f
                for f in class_dir.iterdir()
                if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
            ]
            durations = []
            for fp in files:
                try:
                    duration = _audio_duration(fp)
                    durations.append(duration)
                    if not (self.cfg.min_duration <= duration <= self.cfg.max_duration):
                        out_of_range.append(
                            {
                                "file": str(fp),
                                "duration_s": round(duration, 2),
                                "min_s": self.cfg.min_duration,
                                "max_s": self.cfg.max_duration,
                            }
                        )
                except Exception:
                    corrupt.append(str(fp))
                    if quarantine_corrupt:
                        try:
                            quarantined_path = _move_with_metadata(
                                fp, self.out_dir, quarantine_root
                            )
                            quarantined.append(str(quarantined_path))
                        except Exception as e:
                            logger.warning(f"No se pudo mover a cuarentena {fp}: {e}")

            if durations:
                stats[class_dir.name] = {
                    "n_files": len(files) - sum(1 for f in corrupt if Path(f).parent == class_dir),
                    "n_valid": len(durations),
                    "total_s": round(sum(durations), 1),
                    "mean_dur_s": round(sum(durations) / len(durations), 2),
                    "min_dur_s": round(min(durations), 2),
                    "max_dur_s": round(max(durations), 2),
                }

        class_counts = [v["n_valid"] for v in stats.values()]
        return {
            "classes": stats,
            "total_classes": len(stats),
            "total_files": sum(class_counts),
            "min_class": min(class_counts) if class_counts else 0,
            "max_class": max(class_counts) if class_counts else 0,
            "corrupt_files": corrupt,
            "n_corrupt": len(corrupt),
            "quarantined_files": quarantined,
            "n_quarantined": len(quarantined),
            "out_of_range_files": out_of_range,
            "n_out_of_range": len(out_of_range),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7. TARGETS PREDEFINIDOS POR GRUPO
# ─────────────────────────────────────────────────────────────────────────────


def get_default_targets() -> list[TaxonTarget]:
    """
    Conjunto de 20 especies representativas como punto de partida del dataset.
    Balanceado entre grupos taxonómicos objetivo.
    """
    return [
        # ── Anfibios (anuros) ─────────────────────────────────────────────────
        TaxonTarget("bufo_bufo", "Bufo bufo", "frogs"),
        TaxonTarget("hyla_arborea", "Hyla arborea", "frogs"),
        TaxonTarget("rana_temporaria", "Rana temporaria", "frogs"),
        TaxonTarget("lithobates_catesbeianus", "Lithobates catesbeianus", "frogs"),
        TaxonTarget("eleutherodactylus_coqui", "Eleutherodactylus coqui", "frogs"),
        # ── Mamíferos (quirópteros) ───────────────────────────────────────────
        TaxonTarget("myotis_lucifugus", "Myotis lucifugus", "bats", sources=["inaturalist"]),
        TaxonTarget(
            "tadarida_brasiliensis", "Tadarida brasiliensis", "bats", sources=["inaturalist"]
        ),
        TaxonTarget("eptesicus_fuscus", "Eptesicus fuscus", "bats", sources=["inaturalist"]),
        # ── Mamíferos (otros) ─────────────────────────────────────────────────
        TaxonTarget(
            "pan_troglodytes", "Pan troglodytes", "mammals", sources=["inaturalist", "xeno-canto"]
        ),
        TaxonTarget("ursus_arctos", "Ursus arctos", "mammals", sources=["inaturalist"]),
        # ── Insectos ──────────────────────────────────────────────────────────
        TaxonTarget(
            "gryllus_campestris",
            "Gryllus campestris",
            "insects",
            sources=["xeno-canto", "inaturalist"],
        ),
        TaxonTarget(
            "tettigonia_viridissima",
            "Tettigonia viridissima",
            "insects",
            sources=["xeno-canto", "inaturalist"],
        ),
        TaxonTarget("cicada_orni", "Cicada orni", "insects", sources=["xeno-canto", "inaturalist"]),
        # ── Reptiles ─────────────────────────────────────────────────────────
        TaxonTarget(
            "crocodylus_niloticus", "Crocodylus niloticus", "reptiles", sources=["inaturalist"]
        ),
        TaxonTarget(
            "gekko_gecko", "Gekko gecko", "reptiles", sources=["xeno-canto", "inaturalist"]
        ),
        # ── Aves (referencia cruzada y validación metodológica) ───────────────
        TaxonTarget("turdus_merula", "Turdus merula", "birds"),
        TaxonTarget("ardea_cinerea", "Ardea cinerea", "birds"),
    ]


def get_mexico_bird_targets() -> list[TaxonTarget]:
    """
    Perfil MVP para identificación acústica de aves en México.

    Criterios:
      - Especies presentes en México y razonablemente vocales.
      - Cobertura esperada en Xeno-canto.
      - Mezcla de ambientes urbanos, secos, tropicales y bosques.
    """
    sources = ["xeno-canto", "inaturalist", "gbif"]
    return [
        TaxonTarget("quiscalus_mexicanus", "Quiscalus mexicanus", "birds", sources=sources),
        TaxonTarget("turdus_grayi", "Turdus grayi", "birds", sources=sources),
        TaxonTarget("pitangus_sulphuratus", "Pitangus sulphuratus", "birds", sources=sources),
        TaxonTarget("myiozetetes_similis", "Myiozetetes similis", "birds", sources=sources),
        TaxonTarget("melanerpes_aurifrons", "Melanerpes aurifrons", "birds", sources=sources),
        TaxonTarget(
            "campylorhynchus_brunneicapillus",
            "Campylorhynchus brunneicapillus",
            "birds",
            sources=sources,
        ),
        TaxonTarget("thryophilus_sinaloa", "Thryophilus sinaloa", "birds", sources=sources),
        TaxonTarget("icterus_pustulatus", "Icterus pustulatus", "birds", sources=sources),
        TaxonTarget("toxostoma_curvirostre", "Toxostoma curvirostre", "birds", sources=sources),
        TaxonTarget("zenaida_asiatica", "Zenaida asiatica", "birds", sources=sources),
        TaxonTarget("columbina_inca", "Columbina inca", "birds", sources=sources),
        TaxonTarget("ortalis_vetula", "Ortalis vetula", "birds", sources=sources),
        TaxonTarget("crotophaga_sulcirostris", "Crotophaga sulcirostris", "birds", sources=sources),
        TaxonTarget("momotus_lessonii", "Momotus lessonii", "birds", sources=sources),
        TaxonTarget("glaucidium_brasilianum", "Glaucidium brasilianum", "birds", sources=sources),
        TaxonTarget("geococcyx_californianus", "Geococcyx californianus", "birds", sources=sources),
        TaxonTarget("haemorhous_mexicanus", "Haemorhous mexicanus", "birds", sources=sources),
        TaxonTarget("setophaga_petechia", "Setophaga petechia", "birds", sources=sources),
        TaxonTarget("vireo_hypochryseus", "Vireo hypochryseus", "birds", sources=sources),
        TaxonTarget("cyanocorax_yncas", "Cyanocorax yncas", "birds", sources=sources),
    ]


def get_mexico_anuran_targets() -> list[TaxonTarget]:
    """
    Perfil piloto para anfibios anuros de México.

    Criterios:
      - Especies vocales o con registros acústicos esperables.
      - Relevancia para monitoreo nocturno y ambientes tropicales/secos.
      - Uso como grupo principal de la tesis de maestría.
    """
    sources = ["xeno-canto", "inaturalist", "gbif"]
    return [
        TaxonTarget("smilisca_baudinii", "Smilisca baudinii", "frogs", sources=sources),
        TaxonTarget("rhinella_horribilis", "Rhinella horribilis", "frogs", sources=sources),
        TaxonTarget("incilius_valliceps", "Incilius valliceps", "frogs", sources=sources),
        TaxonTarget("lithobates_berlandieri", "Lithobates berlandieri", "frogs", sources=sources),
        TaxonTarget("lithobates_forreri", "Lithobates forreri", "frogs", sources=sources),
        TaxonTarget(
            "eleutherodactylus_cystignathoides",
            "Eleutherodactylus cystignathoides",
            "frogs",
            sources=sources,
        ),
        TaxonTarget("exerodonta_smaragdinus", "Exerodonta smaragdinus", "frogs", sources=sources),
        TaxonTarget("craugastor_augusti", "Craugastor augusti", "frogs", sources=sources),
    ]


def get_mexico_bat_targets() -> list[TaxonTarget]:
    """
    Perfil piloto para murciélagos presentes en México.

    Nota metodológica:
      Las grabaciones útiles de murciélagos requieren tasas de muestreo altas.
      Este perfil sirve para descubrimiento inicial; la curación final debe validar
      sample rate, equipo de grabación y tipo de llamada.
    """
    sources = ["inaturalist", "gbif"]
    return [
        TaxonTarget("tadarida_brasiliensis", "Tadarida brasiliensis", "bats", sources=sources),
        TaxonTarget("myotis_velifer", "Myotis velifer", "bats", sources=sources),
        TaxonTarget("eptesicus_fuscus", "Eptesicus fuscus", "bats", sources=sources),
        TaxonTarget("nyctinomops_macrotis", "Nyctinomops macrotis", "bats", sources=sources),
        TaxonTarget("lasiurus_cinereus", "Lasiurus cinereus", "bats", sources=sources),
        TaxonTarget("balantiopteryx_plicata", "Balantiopteryx plicata", "bats", sources=sources),
    ]


def get_mexico_insect_targets() -> list[TaxonTarget]:
    """
    Perfil exploratorio para insectos acústicamente detectables.

    Se considera extensión opcional para maestría si la disponibilidad de audio
    y la separación acústica son suficientes.
    """
    sources = ["xeno-canto", "inaturalist", "gbif"]
    return [
        TaxonTarget("gryllus_assimilis", "Gryllus assimilis", "insects", sources=sources),
        TaxonTarget("neoconocephalus_triops", "Neoconocephalus triops", "insects", sources=sources),
        TaxonTarget("oecanthus_niveus", "Oecanthus niveus", "insects", sources=sources),
        TaxonTarget("cicada_orni", "Cicada orni", "insects", sources=sources),
        TaxonTarget("tettigonia_viridissima", "Tettigonia viridissima", "insects", sources=sources),
    ]


def get_mexico_vocal_mammal_targets() -> list[TaxonTarget]:
    """
    Perfil exploratorio para mamíferos vocales.

    No todos los mamíferos son buenos candidatos acústicos; este perfil prioriza
    especies con vocalizaciones audibles y utilidad ecológica.
    """
    sources = ["xeno-canto", "inaturalist", "gbif"]
    return [
        TaxonTarget("alouatta_palliata", "Alouatta palliata", "mammals", sources=sources),
        TaxonTarget("canis_latrans", "Canis latrans", "mammals", sources=sources),
        TaxonTarget("procyon_lotor", "Procyon lotor", "mammals", sources=sources),
        TaxonTarget("nasua_narica", "Nasua narica", "mammals", sources=sources),
        TaxonTarget("odocoileus_virginianus", "Odocoileus virginianus", "mammals", sources=sources),
    ]


def get_mexico_multitaxon_targets() -> list[TaxonTarget]:
    """
    Perfil piloto multitaxonómico para la tesis de maestría.

    Mantiene el alcance defendible: anuros y murciélagos como grupos centrales,
    aves como grupo comparativo, e insectos/mamíferos como extensión inicial.
    """
    return (
        get_mexico_anuran_targets()[:6]
        + get_mexico_bat_targets()[:4]
        + get_mexico_bird_targets()[:6]
        + get_mexico_insect_targets()[:3]
        + get_mexico_vocal_mammal_targets()[:3]
    )


TARGET_PROFILES = {
    "default": get_default_targets,
    "mexico_anurans": get_mexico_anuran_targets,
    "mexico_bats": get_mexico_bat_targets,
    "mexico_birds": get_mexico_bird_targets,
    "mexico_insects": get_mexico_insect_targets,
    "mexico_mammals": get_mexico_vocal_mammal_targets,
    "mexico_multitaxon": get_mexico_multitaxon_targets,
}


def get_targets_for_profile(profile: str) -> list[TaxonTarget]:
    """Retorna targets predefinidos por perfil de dataset."""
    try:
        return TARGET_PROFILES[profile]()
    except KeyError:
        raise ValueError(f"Perfil no reconocido: {profile}. Opciones: {list(TARGET_PROFILES)}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="BioAcoustics Dataset Builder")
    parser.add_argument(
        "--profile",
        default="mexico_birds",
        choices=sorted(TARGET_PROFILES),
        help="Perfil de especies objetivo",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default="data/raw",
        help="Directorio de salida",
    )
    parser.add_argument("--max-per-class", default=300, type=int, help="Máximo archivos por clase")
    parser.add_argument("--min-quality", default="B", help="Calidad mínima XC: A, B, C, D o E")
    parser.add_argument(
        "--min-duration", default=1.0, type=float, help="Duración mínima en segundos"
    )
    parser.add_argument(
        "--max-duration", default=None, type=float, help="Duración máxima en segundos"
    )
    parser.add_argument("--country", default=None, help="Filtro de país para Xeno-canto")
    parser.add_argument("--inat-place-id", default=None, type=int, help="place_id de iNaturalist")
    parser.add_argument(
        "--inat-quality-grade",
        default="research",
        choices=["research", "needs_id", "casual"],
        help="quality_grade de iNaturalist",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Fuentes separadas por coma: xeno-canto,inaturalist,gbif",
    )
    parser.add_argument(
        "--xeno-canto-api-key",
        default=None,
        help="API key de Xeno-canto v3; también puede venir de XENO_CANTO_API_KEY",
    )
    parser.add_argument("--dry-run", action="store_true", help="Solo listar, no descargar")
    parser.add_argument("--validate", action="store_true", help="Solo validar dataset existente")
    parser.add_argument(
        "--quarantine-corrupt",
        action="store_true",
        help="Durante --validate, mover audios corruptos a data/quarantine/corrupt",
    )
    parser.add_argument(
        "--no-validate-downloads",
        action="store_true",
        help="No validar audios inmediatamente después de descargarlos",
    )
    args = parser.parse_args()

    country = args.country
    if country is None and args.profile.startswith("mexico_"):
        country = MEXICO_COUNTRY
    inat_place_id = args.inat_place_id
    if inat_place_id is None and args.profile.startswith("mexico_"):
        inat_place_id = MEXICO_INAT_PLACE_ID
    max_duration = args.max_duration
    if max_duration is None:
        max_duration = 180.0 if args.profile in {"mexico_birds", "mexico_multitaxon"} else 60.0

    cfg = DownloadConfig(
        output_dir=args.output,
        max_per_class=args.max_per_class,
        min_quality=args.min_quality,
        min_duration=args.min_duration,
        max_duration=max_duration,
        dry_run=args.dry_run,
        validate_downloads=not args.no_validate_downloads,
        country=country,
        inat_place_id=inat_place_id,
        inat_quality_grade=args.inat_quality_grade,
        profile=args.profile,
        xeno_canto_api_key=args.xeno_canto_api_key,
    )
    builder = DatasetBuilder(cfg)

    if args.validate:
        report = builder.validate_dataset(quarantine_corrupt=args.quarantine_corrupt)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        targets = get_targets_for_profile(args.profile)
        if args.sources:
            sources = [s.strip() for s in args.sources.split(",") if s.strip()]
            for target in targets:
                target.sources = sources
        summary = builder.build(targets)
        print(f"\nResumen de descarga ({args.profile}):")
        for cls, n in sorted(summary.items()):
            print(f"  {cls:40s}: {n:4d} archivos")
        print(f"\nTotal: {sum(summary.values())} archivos")
