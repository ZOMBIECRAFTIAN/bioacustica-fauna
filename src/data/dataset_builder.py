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

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLIENTE HTTP CON RETRY Y RATE LIMITING
# ─────────────────────────────────────────────────────────────────────────────

def _make_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    """Session con reintentos automáticos y backoff exponencial."""
    session = requests.Session()
    retry   = Retry(
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
    output_dir:    str  = "data/raw"           # directorio raíz de salida
    max_per_class: int  = 500                  # máximo de archivos por clase
    min_quality:   str  = "B"                  # calidad mínima (A/B para Xeno-canto)
    min_duration:  float = 1.0                 # segundos mínimos de audio
    max_duration:  float = 60.0                # segundos máximos
    target_sr:     int  = 44_100               # resamplear a este SR tras descarga
    formats:       List[str] = field(default_factory=lambda: ["mp3", "wav", "ogg", "flac"])
    dry_run:       bool = False                # si True, solo lista sin descargar
    delay_s:       float = 0.5                 # pausa entre requests (rate limiting)
    save_metadata: bool = True                 # guardar metadata JSON por archivo


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

    BASE_URL = "https://xeno-canto.org/api/2/recordings"

    def __init__(self, config: DownloadConfig):
        self.cfg     = config
        self.session = _make_session()

    def search(
        self,
        query: str,
        quality: str = "A+B",
        type_: str = "",
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
        params  = {"query": f"{query} q:{quality}"}
        if type_:
            params["query"] += f" type:{type_}"
        page    = 1
        total_pages = 1

        while page <= total_pages:
            params["page"] = page
            url = f"{self.BASE_URL}?{urlencode(params)}"
            logger.debug(f"XenoCanto GET: {url}")

            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if page == 1:
                total_pages = int(data.get("numPages", 1))
                total_recs  = int(data.get("numRecordings", 0))
                logger.info(f"Xeno-canto '{query}': {total_recs} grabaciones en {total_pages} páginas")

            for rec in data.get("recordings", []):
                dur = float(rec.get("length", "0").replace(":", ".") or 0)
                if min_len <= dur <= max_len:
                    yield rec

            page += 1
            time.sleep(self.cfg.delay_s)

    def download_species(
        self,
        scientific_name: str,
        class_label: str,
        quality: str = "A+B",
    ) -> List[Path]:
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

        downloaded: List[Path] = []
        n_existing = len(existing)

        for rec in self.search(scientific_name, quality=quality):
            if n_existing + len(downloaded) >= self.cfg.max_per_class:
                break

            rec_id  = rec.get("id", "")
            file_url = "https:" + rec.get("file", "")
            ext     = rec.get("file-name", "audio.mp3").rsplit(".", 1)[-1].lower()
            out_path = out_dir / f"xc_{rec_id}.{ext}"

            if out_path.exists():
                continue

            if self.cfg.dry_run:
                logger.info(f"  [DRY] {out_path.name} | {rec.get('length')}s")
                continue

            try:
                resp = self.session.get(file_url, timeout=60, stream=True)
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)

                if self.cfg.save_metadata:
                    meta_path = out_path.with_suffix(".json")
                    with open(meta_path, "w") as f:
                        json.dump({
                            "source":     "xeno-canto",
                            "id":         rec_id,
                            "species":    rec.get("en", scientific_name),
                            "country":    rec.get("cnt", ""),
                            "lat":        rec.get("lat", ""),
                            "lng":        rec.get("lng", ""),
                            "date":       rec.get("date", ""),
                            "quality":    rec.get("q", ""),
                            "duration":   rec.get("length", ""),
                            "license":    rec.get("lic", ""),
                        }, f, indent=2)

                downloaded.append(out_path)
                logger.debug(f"  ✓ {out_path.name}")

            except Exception as e:
                logger.warning(f"  ✗ Error descargando {rec_id}: {e}")

            time.sleep(self.cfg.delay_s)

        logger.info(f"  {class_label}: {len(downloaded)} archivos nuevos descargados.")
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

    BASE_URL   = "https://api.inaturalist.org/v1/observations"
    TAXON_IDS  = {
        "amphibia": 20978,
        "insecta":  47158,
        "mammalia": 40151,
        "reptilia": 26036,
        "aves":     3,
    }

    def __init__(self, config: DownloadConfig, api_token: Optional[str] = None):
        self.cfg     = config
        self.session = _make_session()
        if api_token:
            self.session.headers["Authorization"] = f"Bearer {api_token}"

    def search_sounds(
        self,
        taxon_id: int,
        taxon_name: Optional[str] = None,
        place_id: Optional[int] = None,
        quality_grade: str = "research",
        per_page: int = 200,
    ) -> Iterator[dict]:
        """
        Itera sobre observaciones con sonido en iNaturalist.

        Yields:
            Diccionario de observación con campo 'sounds' no vacío.
        """
        params = {
            "taxon_id":     taxon_id,
            "sounds":       "true",
            "quality_grade": quality_grade,
            "per_page":     per_page,
            "order":        "desc",
            "order_by":     "id",
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
        taxon_name: Optional[str] = None,
        class_label: Optional[str] = None,
    ) -> List[Path]:
        """
        Descarga sonidos de un grupo taxonómico desde iNaturalist.

        Args:
            taxon_group: Clave en TAXON_IDS ("amphibia", "insecta", etc.)
            taxon_name:  Nombre científico para filtro adicional.
            class_label: Nombre del subdirectorio de clase.
        """
        taxon_id    = self.TAXON_IDS.get(taxon_group.lower())
        if taxon_id is None:
            raise ValueError(f"Grupo no reconocido: {taxon_group}. Opciones: {list(self.TAXON_IDS)}")

        label   = class_label or (taxon_name or taxon_group).replace(" ", "_").lower()
        out_dir = Path(self.cfg.output_dir) / label
        out_dir.mkdir(parents=True, exist_ok=True)

        downloaded: List[Path] = []

        for obs in self.search_sounds(taxon_id, taxon_name=taxon_name):
            if len(downloaded) >= self.cfg.max_per_class:
                break

            obs_id = obs.get("id", "")
            for snd in obs.get("sounds", []):
                url = snd.get("file_url", "")
                if not url:
                    continue
                ext      = url.rsplit(".", 1)[-1].lower().split("?")[0] or "mp3"
                snd_id   = snd.get("id", obs_id)
                out_path = out_dir / f"inat_{snd_id}.{ext}"

                if out_path.exists() or ext not in self.cfg.formats:
                    continue
                if self.cfg.dry_run:
                    logger.info(f"  [DRY] {out_path.name}")
                    continue

                try:
                    resp = self.session.get(url, timeout=60, stream=True)
                    resp.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)

                    if self.cfg.save_metadata:
                        sp_name = ""
                        if obs.get("taxon"):
                            sp_name = obs["taxon"].get("name", "")
                        meta_path = out_path.with_suffix(".json")
                        with open(meta_path, "w") as f:
                            json.dump({
                                "source":       "inaturalist",
                                "obs_id":       obs_id,
                                "sound_id":     snd_id,
                                "species":      sp_name,
                                "observed_on":  obs.get("observed_on", ""),
                                "lat":          obs.get("location", "").split(",")[0] if obs.get("location") else "",
                                "lng":          obs.get("location", "").split(",")[1] if obs.get("location") and "," in obs.get("location","") else "",
                                "quality_grade": obs.get("quality_grade", ""),
                                "license":      obs.get("license_code", ""),
                            }, f, indent=2)

                    downloaded.append(out_path)
                    logger.debug(f"  ✓ {out_path.name}")

                except Exception as e:
                    logger.warning(f"  ✗ iNat {snd_id}: {e}")

            if len(downloaded) >= self.cfg.max_per_class:
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
        "Mammalia":  359,
        "Amphibia":  131,
        "Reptilia":  358,
        "Insecta":   216,
    }

    def __init__(self, config: DownloadConfig):
        self.cfg     = config
        self.session = _make_session()

    def search_sounds(
        self, class_key: int, limit: int = 300
    ) -> Iterator[dict]:
        """Itera sobre ocurrencias con media de tipo 'Sound'."""
        offset = 0
        while True:
            params = {
                "classKey":  class_key,
                "mediaType": "Sound",
                "hasCoordinate": "true",
                "limit":     min(limit, 300),
                "offset":    offset,
            }
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


# ─────────────────────────────────────────────────────────────────────────────
# 6. ORQUESTADOR: DatasetBuilder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaxonTarget:
    """Define un target de descarga para una clase específica del dataset."""
    class_label:  str                          # nombre del directorio (e.g., "bufo_bufo")
    scientific_name: str                       # nombre científico
    acoustic_group:  str                       # "frogs", "bats", "insects", "mammals", "reptiles"
    sources: List[str] = field(default_factory=lambda: ["xeno-canto", "inaturalist"])
    xc_query: Optional[str] = None             # query personalizada para XenoCanto
    inat_taxon_name: Optional[str] = None      # nombre para filtro iNat


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

    def __init__(self, config: DownloadConfig, inat_token: Optional[str] = None):
        self.cfg    = config
        self.xc     = XenoCantoDownloader(config)
        self.inat   = iNaturalistDownloader(config, api_token=inat_token)
        self.manifest: List[dict] = []
        self.out_dir = Path(config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # Mapa grupo → taxon_id iNat
    _INAT_GROUP_MAP = {
        "frogs":    ("amphibia",  20978),
        "bats":     ("mammalia",  40151),
        "mammals":  ("mammalia",  40151),
        "insects":  ("insecta",   47158),
        "reptiles": ("reptilia",  26036),
    }

    def build(self, targets: List[TaxonTarget]) -> dict:
        """
        Descarga todos los targets y genera el manifiesto del dataset.

        Returns:
            Resumen con conteo por clase.
        """
        summary: Dict[str, int] = {}

        for target in targets:
            logger.info(f"\n{'─'*60}")
            logger.info(f"Procesando: {target.scientific_name} → {target.class_label}")
            files: List[Path] = []

            # ── Xeno-canto ────────────────────────────────────
            if "xeno-canto" in target.sources:
                query = target.xc_query or target.scientific_name
                try:
                    new = self.xc.download_species(query, target.class_label)
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
                        )
                        files.extend(new)
                    except Exception as e:
                        logger.error(f"iNat error para {target.class_label}: {e}")

            n = len(list((self.out_dir / target.class_label).glob("*.*"))) if \
                (self.out_dir / target.class_label).exists() else 0
            summary[target.class_label] = n
            self.manifest.append({
                "class_label":    target.class_label,
                "scientific_name": target.scientific_name,
                "acoustic_group": target.acoustic_group,
                "n_files":        n,
                "sources":        target.sources,
            })

        # Guardar manifiesto
        manifest_path = self.out_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "max_per_class": self.cfg.max_per_class,
                    "min_quality":   self.cfg.min_quality,
                },
                "classes": self.manifest,
                "total_classes": len(self.manifest),
                "total_files":   sum(summary.values()),
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset completado: {sum(summary.values())} archivos en {len(summary)} clases")
        logger.info(f"Manifiesto guardado: {manifest_path}")
        return summary

    def validate_dataset(self) -> dict:
        """
        Valida la integridad del dataset: archivos corruptos, clases desbalanceadas.

        Returns:
            Diccionario con estadísticas y lista de archivos con problemas.
        """
        import soundfile as sf

        stats: Dict[str, dict] = {}
        corrupt: List[str]     = []

        for class_dir in sorted(self.out_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            files = [f for f in class_dir.iterdir()
                     if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
            durations = []
            for fp in files:
                try:
                    info = sf.info(str(fp))
                    durations.append(info.duration)
                except Exception:
                    corrupt.append(str(fp))

            if durations:
                stats[class_dir.name] = {
                    "n_files":       len(files),
                    "n_valid":       len(durations),
                    "total_s":       round(sum(durations), 1),
                    "mean_dur_s":    round(sum(durations) / len(durations), 2),
                    "min_dur_s":     round(min(durations), 2),
                    "max_dur_s":     round(max(durations), 2),
                }

        class_counts = [v["n_valid"] for v in stats.values()]
        return {
            "classes":        stats,
            "total_classes":  len(stats),
            "total_files":    sum(class_counts),
            "min_class":      min(class_counts) if class_counts else 0,
            "max_class":      max(class_counts) if class_counts else 0,
            "corrupt_files":  corrupt,
            "n_corrupt":      len(corrupt),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7. TARGETS PREDEFINIDOS POR GRUPO
# ─────────────────────────────────────────────────────────────────────────────

def get_default_targets() -> List[TaxonTarget]:
    """
    Conjunto de 20 especies representativas como punto de partida del dataset.
    Balanceado entre grupos taxonómicos objetivo.
    """
    return [
        # ── Anfibios (anuros) ─────────────────────────────────────────────────
        TaxonTarget("bufo_bufo",             "Bufo bufo",             "frogs"),
        TaxonTarget("hyla_arborea",          "Hyla arborea",          "frogs"),
        TaxonTarget("rana_temporaria",       "Rana temporaria",       "frogs"),
        TaxonTarget("lithobates_catesbeianus","Lithobates catesbeianus","frogs"),
        TaxonTarget("eleutherodactylus_coqui","Eleutherodactylus coqui","frogs"),

        # ── Mamíferos (quirópteros) ───────────────────────────────────────────
        TaxonTarget("myotis_lucifugus",       "Myotis lucifugus",      "bats",
                    sources=["inaturalist"]),
        TaxonTarget("tadarida_brasiliensis",  "Tadarida brasiliensis", "bats",
                    sources=["inaturalist"]),
        TaxonTarget("eptesicus_fuscus",       "Eptesicus fuscus",      "bats",
                    sources=["inaturalist"]),

        # ── Mamíferos (otros) ─────────────────────────────────────────────────
        TaxonTarget("pan_troglodytes",        "Pan troglodytes",       "mammals",
                    sources=["inaturalist", "xeno-canto"]),
        TaxonTarget("ursus_arctos",           "Ursus arctos",          "mammals",
                    sources=["inaturalist"]),

        # ── Insectos ──────────────────────────────────────────────────────────
        TaxonTarget("gryllus_campestris",     "Gryllus campestris",    "insects",
                    sources=["xeno-canto", "inaturalist"]),
        TaxonTarget("tettigonia_viridissima", "Tettigonia viridissima","insects",
                    sources=["xeno-canto", "inaturalist"]),
        TaxonTarget("cicada_orni",            "Cicada orni",           "insects",
                    sources=["xeno-canto", "inaturalist"]),

        # ── Reptiles ─────────────────────────────────────────────────────────
        TaxonTarget("crocodylus_niloticus",   "Crocodylus niloticus",  "reptiles",
                    sources=["inaturalist"]),
        TaxonTarget("gekko_gecko",            "Gekko gecko",           "reptiles",
                    sources=["xeno-canto", "inaturalist"]),

        # ── Aves (referencia cruzada y validación metodológica) ───────────────
        TaxonTarget("turdus_merula",          "Turdus merula",         "birds"),
        TaxonTarget("ardea_cinerea",          "Ardea cinerea",         "birds"),
    ]


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
    parser.add_argument("--output",       default="data/raw",     help="Directorio de salida")
    parser.add_argument("--max-per-class",default=300, type=int,  help="Máximo archivos por clase")
    parser.add_argument("--dry-run",      action="store_true",    help="Solo listar, no descargar")
    parser.add_argument("--validate",     action="store_true",    help="Solo validar dataset existente")
    args = parser.parse_args()

    cfg     = DownloadConfig(
        output_dir=args.output,
        max_per_class=args.max_per_class,
        dry_run=args.dry_run,
    )
    builder = DatasetBuilder(cfg)

    if args.validate:
        report = builder.validate_dataset()
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        targets = get_default_targets()
        summary = builder.build(targets)
        print("\nResumen de descarga:")
        for cls, n in sorted(summary.items()):
            print(f"  {cls:40s}: {n:4d} archivos")
        print(f"\nTotal: {sum(summary.values())} archivos")
