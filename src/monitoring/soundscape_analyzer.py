"""
src/monitoring/soundscape_analyzer.py
─────────────────────────────────────────────────────────────────────────────
Analizador de paisaje sonoro de alto nivel.

Funcionalidades
───────────────
  SoundscapeAnalyzer : procesa múltiples grabaciones → tabla de índices
  Comparación entre sitios / fechas
  Gráficas publicables:
    - Perfil temporal de índices (circadiano / estacional)
    - Heatmap ACI por hora × día
    - Radar chart comparativo entre sitios
    - Boxplot de índices por sitio o grupo taxonómico
    - Espectrograma anotado con índices
  Exportación a CSV / JSON

Uso
───
    from src.monitoring.soundscape_analyzer import SoundscapeAnalyzer

    analyzer = SoundscapeAnalyzer(sample_rate=22050)

    # Analizar directorio de grabaciones
    df = analyzer.analyze_directory(
        audio_dir  = "data/raw/site_A",
        output_dir = "results/soundscape",
        windowed   = True,
        window_s   = 60.0,
    )

    # Comparar dos sitios
    analyzer.plot_site_comparison(
        dfs        = {"Sitio A": df_a, "Sitio B": df_b},
        output_dir = "results/comparison",
    )

Autor: Ian
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap

logger = logging.getLogger(__name__)

# Importaciones opcionales — pandas es deseable pero no obligatorio
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False
    logger.info("pandas no disponible — usando dicts en lugar de DataFrame.")

try:
    import soundfile as sf
    _SF = True
except ImportError:
    _SF = False

from src.monitoring.acoustic_indices import (
    AcousticIndices,
    IndicesConfig,
    IndicesResult,
    compute_indices,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}

# Índices escalares para visualizaciones
SCALAR_INDICES = ["aci", "adi", "aei", "bi", "ndsi", "hf", "ht", "h", "rms", "zcr"]

# Colores por índice para gráficas consistentes
INDEX_COLORS = {
    "aci":  "#2196F3",
    "adi":  "#4CAF50",
    "aei":  "#FF9800",
    "bi":   "#9C27B0",
    "ndsi": "#F44336",
    "hf":   "#00BCD4",
    "ht":   "#009688",
    "h":    "#3F51B5",
    "rms":  "#795548",
    "zcr":  "#607D8B",
}


# ─────────────────────────────────────────────────────────────────────────────
# ANALIZADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class SoundscapeAnalyzer:
    """
    Procesa colecciones de grabaciones y genera análisis de paisaje sonoro.

    Parameters
    ----------
    sample_rate : int — SR por defecto (se sobreescribe al leer archivos)
    cfg         : IndicesConfig — configuración de índices (None → defaults)
    fig_dpi     : int — resolución de figuras exportadas
    fig_format  : str — formato de imagen (png | pdf | svg)
    """

    def __init__(
        self,
        sample_rate: int = 22_050,
        cfg:         Optional[IndicesConfig] = None,
        fig_dpi:     int  = 150,
        fig_format:  str  = "png",
    ):
        self.sample_rate = sample_rate
        self.cfg         = cfg or IndicesConfig(sample_rate=sample_rate)
        self.fig_dpi     = fig_dpi
        self.fig_format  = fig_format

    # ── Carga de audio ────────────────────────────────────────────────────────

    def _load_audio(self, path: Path):
        """Carga archivo de audio → (y, sr)."""
        if not _SF:
            raise ImportError("soundfile requerido: pip install soundfile")
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, sr

    # ── Análisis de un archivo ────────────────────────────────────────────────

    def analyze_file(
        self,
        filepath:  Union[str, Path],
        windowed:  bool  = False,
        window_s:  float = 60.0,
        hop_s:     float = 60.0,
    ) -> Union[Dict, List[Dict]]:
        """
        Calcula índices de un archivo de audio.

        Returns
        -------
        Dict (windowed=False) o List[Dict] (windowed=True)
        cada dict incluye: filename, duration_s, + todos los índices.
        """
        path = Path(filepath)
        y, sr = self._load_audio(path)

        cfg = IndicesConfig(sample_rate=sr, **{
            k: v for k, v in self.cfg.__dict__.items()
            if k != "sample_rate"
        })
        ai = AcousticIndices(cfg)

        if windowed:
            rows = ai.compute_windowed(y, window_s=window_s, hop_s=hop_s)
            for r in rows:
                r["filename"] = path.name
                r["filepath"] = str(path)
            return rows
        else:
            result = ai.compute_all(y)
            row    = result.to_dict()
            row["filename"] = path.name
            row["filepath"] = str(path)
            return row

    # ── Análisis de directorio ────────────────────────────────────────────────

    def analyze_directory(
        self,
        audio_dir:  Union[str, Path],
        output_dir: Union[str, Path] = "results/soundscape",
        windowed:   bool   = False,
        window_s:   float  = 60.0,
        hop_s:      float  = 60.0,
        recursive:  bool   = True,
        save_csv:   bool   = True,
        save_json:  bool   = True,
        plot:       bool   = True,
    ):
        """
        Analiza todos los archivos de audio en un directorio.

        Parameters
        ----------
        audio_dir  : directorio con grabaciones
        output_dir : directorio de salida para resultados
        windowed   : analizar por ventanas temporales
        window_s   : duración de cada ventana (s)
        hop_s      : salto entre ventanas (s)
        recursive  : buscar en subdirectorios
        save_csv   : guardar tabla de índices en CSV
        save_json  : guardar tabla de índices en JSON
        plot       : generar figuras de análisis

        Returns
        -------
        Lista de dicts con índices por archivo (o por ventana si windowed=True).
        Si pandas está instalado, retorna un DataFrame.
        """
        audio_dir  = Path(audio_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Encontrar archivos
        if recursive:
            files = [
                f for f in audio_dir.rglob("*")
                if f.suffix.lower() in AUDIO_EXTENSIONS
            ]
        else:
            files = [
                f for f in audio_dir.glob("*")
                if f.suffix.lower() in AUDIO_EXTENSIONS
            ]

        if not files:
            logger.warning("No se encontraron archivos de audio en %s", audio_dir)
            return [] if not _PANDAS else pd.DataFrame()

        logger.info("Analizando %d archivos en %s ...", len(files), audio_dir)

        all_rows: List[Dict] = []
        errors:   List[str]  = []

        for i, path in enumerate(sorted(files)):
            logger.info("[%d/%d] %s", i + 1, len(files), path.name)
            try:
                result = self.analyze_file(
                    path, windowed=windowed, window_s=window_s, hop_s=hop_s
                )
                if isinstance(result, list):
                    all_rows.extend(result)
                else:
                    all_rows.append(result)
            except Exception as exc:
                logger.warning("Error en %s: %s", path.name, exc)
                errors.append(f"{path.name}: {exc}")

        if errors:
            err_path = output_dir / "errors.txt"
            err_path.write_text("\n".join(errors), encoding="utf-8")
            logger.warning("%d errores — ver %s", len(errors), err_path)

        # Guardar resultados
        if save_json:
            jp = output_dir / "acoustic_indices.json"
            jp.write_text(
                json.dumps(all_rows, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            logger.info("JSON guardado: %s", jp)

        if save_csv and all_rows:
            cp   = output_dir / "acoustic_indices.csv"
            keys = list(all_rows[0].keys())
            with open(cp, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_rows)
            logger.info("CSV guardado: %s", cp)

        # Convertir a DataFrame si disponible
        if _PANDAS:
            df = pd.DataFrame(all_rows)
            # Eliminar columnas anidadas (band_proportions es dict)
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, dict)).any():
                    df = df.drop(columns=[col])
        else:
            df = all_rows

        # Generar figuras
        if plot and all_rows:
            self._plot_directory_analysis(df, output_dir)

        return df

    # ── Visualizaciones ───────────────────────────────────────────────────────

    def _plot_directory_analysis(self, df, output_dir: Path) -> None:
        """Genera el conjunto estándar de figuras para un directorio analizado."""
        figs_dir = output_dir / "figures"
        figs_dir.mkdir(exist_ok=True)

        if _PANDAS and isinstance(df, pd.DataFrame) and len(df) > 0:
            self.plot_indices_overview(df, figs_dir)
            self.plot_ndsi_timeline(df, figs_dir)
            self.plot_band_activity(df, figs_dir)

    def plot_indices_overview(self, df, output_dir: Path) -> Path:
        """
        Panel 2×4 con distribución (violin + scatter) de cada índice.
        """
        indices  = [i for i in SCALAR_INDICES if i in df.columns]
        n        = len(indices)
        ncols    = 4
        nrows    = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.5, nrows * 3.2))
        axes_flat = np.array(axes).flatten()

        for i, idx in enumerate(indices):
            ax     = axes_flat[i]
            values = df[idx].dropna().values
            color  = INDEX_COLORS.get(idx, "#607D8B")

            ax.violinplot(values, positions=[0], showmedians=True,
                          showextrema=True)
            ax.scatter(
                np.random.normal(0, 0.05, len(values)),
                values, alpha=0.4, s=8, color=color,
            )
            ax.set_title(idx.upper(), fontsize=10, fontweight="bold", color=color)
            ax.set_xticks([])
            ax.set_ylabel("Valor", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

            # Estadísticas en el título
            ax.set_xlabel(
                f"μ={values.mean():.3f}  σ={values.std():.3f}",
                fontsize=7,
            )

        # Ocultar ejes sobrantes
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Distribución de Índices Acústicos", fontsize=13,
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        path = output_dir / f"indices_overview.{self.fig_format}"
        fig.savefig(str(path), dpi=self.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figura: %s", path)
        return path

    def plot_ndsi_timeline(self, df, output_dir: Path) -> Path:
        """
        Gráfica temporal de NDSI con banda de color:
          verde  → biofonia (NDSI > 0)
          rojo   → antrofonia (NDSI < 0)
        """
        if "ndsi" not in df.columns:
            return None

        ndsi = df["ndsi"].values
        x    = np.arange(len(ndsi))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.fill_between(x, 0, ndsi,
                        where=ndsi >= 0, alpha=0.5, color="#4CAF50",
                        label="Biofonia (NDSI ≥ 0)")
        ax.fill_between(x, 0, ndsi,
                        where=ndsi < 0, alpha=0.5, color="#F44336",
                        label="Antrofonia (NDSI < 0)")
        ax.plot(x, ndsi, color="black", lw=0.8, alpha=0.7)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("Número de grabación / ventana temporal")
        ax.set_ylabel("NDSI")
        ax.set_title("Normalized Difference Soundscape Index (NDSI) — Perfil Temporal",
                     fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        path = output_dir / f"ndsi_timeline.{self.fig_format}"
        fig.savefig(str(path), dpi=self.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figura: %s", path)
        return path

    def plot_band_activity(self, df, output_dir: Path) -> Path:
        """
        Heatmap de actividad espectral por banda de frecuencia.
        Requiere columna 'band_proportions' (o banda_* columnas).
        """
        band_cols = [c for c in df.columns if "hz" in c.lower()]
        if not band_cols:
            logger.debug("Sin columnas de banda para heatmap.")
            return None

        matrix = df[band_cols].values.T   # (n_bandas, n_grabaciones)

        fig, ax = plt.subplots(figsize=(min(16, len(df) * 0.3 + 4), 5))
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="YlOrRd",
            vmin=0, vmax=1,
        )
        ax.set_yticks(range(len(band_cols)))
        ax.set_yticklabels(band_cols, fontsize=7)
        ax.set_xlabel("Grabación / ventana temporal")
        ax.set_ylabel("Banda de frecuencia")
        ax.set_title("Actividad acústica por banda de frecuencia",
                     fontsize=11)
        plt.colorbar(im, ax=ax, label="Proporción activa")
        fig.tight_layout()
        path = output_dir / f"band_activity_heatmap.{self.fig_format}"
        fig.savefig(str(path), dpi=self.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figura: %s", path)
        return path

    def plot_site_comparison(
        self,
        dfs:        Dict[str, object],   # {"Sitio A": df_a, "Sitio B": df_b}
        output_dir: Union[str, Path],
        indices:    List[str] = None,
    ) -> Path:
        """
        Boxplot comparativo de índices entre múltiples sitios.

        Parameters
        ----------
        dfs        : dict nombre_sitio → DataFrame (o list de dicts)
        output_dir : directorio de salida
        indices    : lista de índices a comparar (None → todos los escalares)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if indices is None:
            indices = SCALAR_INDICES

        n    = len(indices)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 3.5, rows * 3.5))
        axes_flat = np.array(axes).flatten()

        site_names = list(dfs.keys())
        palette    = plt.cm.Set2(np.linspace(0, 1, len(site_names)))

        for i, idx in enumerate(indices):
            ax = axes_flat[i]
            data_by_site = []
            for site in site_names:
                df = dfs[site]
                if _PANDAS and isinstance(df, pd.DataFrame):
                    vals = df[idx].dropna().values if idx in df.columns else np.array([])
                else:
                    vals = np.array([r[idx] for r in df if idx in r])
                data_by_site.append(vals)

            bp = ax.boxplot(
                data_by_site,
                labels=site_names,
                patch_artist=True,
                notch=False,
                showfliers=True,
            )
            for patch, color in zip(bp["boxes"], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

            ax.set_title(idx.upper(), fontsize=10, fontweight="bold",
                         color=INDEX_COLORS.get(idx, "black"))
            ax.set_xticklabels(site_names, rotation=20, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Comparación de Índices Acústicos entre Sitios",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        path = output_dir / f"site_comparison.{self.fig_format}"
        fig.savefig(str(path), dpi=self.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figura comparativa: %s", path)
        return path

    def plot_radar(
        self,
        site_stats: Dict[str, Dict[str, float]],
        output_dir: Union[str, Path],
        indices:    List[str] = None,
        normalize:  bool      = True,
    ) -> Path:
        """
        Radar (spider) chart con valores medios de índices por sitio.

        Parameters
        ----------
        site_stats : {"Sitio A": {"aci": 1200, "adi": 1.5, ...}, ...}
        normalize  : normalizar cada eje a [0, 1] para comparabilidad
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if indices is None:
            indices = ["aci", "adi", "bi", "ndsi", "h"]

        N      = len(indices)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]   # cerrar el polígono

        # Normalizar a [0,1] por índice si se requiere
        if normalize:
            all_vals = {
                idx: [site_stats[s].get(idx, 0) for s in site_stats]
                for idx in indices
            }
            mins = {idx: min(v) for idx, v in all_vals.items()}
            maxs = {idx: max(v) for idx, v in all_vals.items()}
            def _norm(idx, val):
                r = maxs[idx] - mins[idx]
                return (val - mins[idx]) / r if r > 0 else 0.5
        else:
            def _norm(idx, val): return val

        fig, ax = plt.subplots(figsize=(7, 7),
                               subplot_kw=dict(projection="polar"))
        palette = plt.cm.Set1(np.linspace(0, 0.8, len(site_stats)))

        for (site, stats), color in zip(site_stats.items(), palette):
            vals = [_norm(idx, stats.get(idx, 0)) for idx in indices]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", lw=2, color=color, label=site)
            ax.fill(angles, vals, alpha=0.12, color=color)

        ax.set_thetagrids(np.degrees(angles[:-1]), indices)
        ax.set_title(
            "Radar de Índices Acústicos — Comparación entre Sitios",
            size=11, fontweight="bold", pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True, alpha=0.4)

        path = output_dir / f"radar_sites.{self.fig_format}"
        fig.savefig(str(path), dpi=self.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Radar guardado: %s", path)
        return path

    def plot_circadian(
        self,
        df,
        time_col:   str,
        output_dir: Union[str, Path],
        indices:    List[str] = None,
    ) -> Path:
        """
        Perfil circadiano: media de índices por hora del día.

        Requiere columna con timestamps (datetime o string ISO).
        Ideal para análisis de 24h de grabación continua.

        Parameters
        ----------
        df       : DataFrame con columna de tiempo y columnas de índices
        time_col : nombre de la columna con timestamps
        """
        if not _PANDAS:
            logger.warning("pandas requerido para plot_circadian.")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if indices is None:
            indices = ["aci", "adi", "ndsi", "h"]

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df["hour"]   = df[time_col].dt.hour

        hourly = df.groupby("hour")[indices].mean()

        fig, axes = plt.subplots(
            len(indices), 1,
            figsize=(12, len(indices) * 2.5),
            sharex=True,
        )
        if len(indices) == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            color = INDEX_COLORS.get(idx, "#607D8B")
            ax.plot(hourly.index, hourly[idx], "o-", color=color,
                    lw=2, ms=5, label=idx.upper())
            ax.fill_between(hourly.index, hourly[idx], alpha=0.15, color=color)
            ax.set_ylabel(idx.upper(), color=color, fontsize=9)
            ax.tick_params(axis="y", labelcolor=color)
            ax.grid(alpha=0.25)

            # Marcar amanecer/atardecer aproximados
            ax.axvspan(5, 8,   alpha=0.08, color="gold",  label="Amanecer")
            ax.axvspan(17, 20, alpha=0.08, color="orange", label="Atardecer")

        axes[-1].set_xlabel("Hora del día (UTC)", fontsize=10)
        axes[0].set_title(
            "Variación Circadiana de Índices Acústicos",
            fontsize=12, fontweight="bold",
        )
        axes[0].legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        path = output_dir / f"circadian_profile.{self.fig_format}"
        fig.savefig(str(path), dpi=self.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Circadiana: %s", path)
        return path

    # ── Resumen estadístico ───────────────────────────────────────────────────

    def summary_stats(self, df) -> Dict:
        """
        Calcula estadísticas descriptivas de los índices.

        Returns
        -------
        Dict con mean, std, min, max, p25, p50, p75 por índice.
        """
        stats: Dict = {}

        if _PANDAS and isinstance(df, pd.DataFrame):
            for idx in SCALAR_INDICES:
                if idx not in df.columns:
                    continue
                v = df[idx].dropna().values
                if len(v) == 0:
                    continue
                stats[idx] = {
                    "mean": float(np.mean(v)),
                    "std":  float(np.std(v)),
                    "min":  float(np.min(v)),
                    "max":  float(np.max(v)),
                    "p25":  float(np.percentile(v, 25)),
                    "p50":  float(np.percentile(v, 50)),
                    "p75":  float(np.percentile(v, 75)),
                    "n":    int(len(v)),
                }
        else:
            for idx in SCALAR_INDICES:
                v = np.array([r[idx] for r in df if idx in r])
                if len(v) == 0:
                    continue
                stats[idx] = {
                    "mean": float(np.mean(v)),
                    "std":  float(np.std(v)),
                    "min":  float(np.min(v)),
                    "max":  float(np.max(v)),
                    "p50":  float(np.percentile(v, 50)),
                    "n":    int(len(v)),
                }

        return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Análisis de paisaje sonoro — Índices acústicos"
    )
    sub = parser.add_subparsers(dest="command")

    # Analizar un archivo
    f_p = sub.add_parser("file", help="Analizar un archivo de audio")
    f_p.add_argument("input",         help="Ruta al archivo .wav/.flac")
    f_p.add_argument("--windowed",    action="store_true")
    f_p.add_argument("--window-s",    type=float, default=60.0)
    f_p.add_argument("--output",      default=None, help="Guardar JSON")

    # Analizar directorio
    d_p = sub.add_parser("dir", help="Analizar directorio de grabaciones")
    d_p.add_argument("input",         help="Directorio con archivos de audio")
    d_p.add_argument("--output-dir",  default="results/soundscape")
    d_p.add_argument("--windowed",    action="store_true")
    d_p.add_argument("--window-s",    type=float, default=60.0)
    d_p.add_argument("--no-plot",     action="store_true")
    d_p.add_argument("--sample-rate", type=int, default=22_050)

    args = parser.parse_args()

    if args.command == "file":
        from src.monitoring.acoustic_indices import indices_from_file
        result = indices_from_file(
            args.input,
            windowed=args.windowed,
            window_s=args.window_s,
        )

        if isinstance(result, list):
            print(f"\n{len(result)} ventanas procesadas:")
            for r in result[:5]:
                print(f"  [{r['t_start_s']:.1f}s–{r['t_end_s']:.1f}s]"
                      f"  ACI={r['aci']:.1f}  NDSI={r['ndsi']:.3f}"
                      f"  H={r['h']:.3f}")
            if len(result) > 5:
                print(f"  ... ({len(result) - 5} más)")
        else:
            from src.monitoring.acoustic_indices import IndicesResult
            r = result
            print(f"\nACI={r['aci']:.2f}  ADI={r['adi']:.3f}"
                  f"  NDSI={r['ndsi']:.3f}  H={r['h']:.3f}")

        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            print(f"Guardado: {args.output}")

    elif args.command == "dir":
        analyzer = SoundscapeAnalyzer(sample_rate=args.sample_rate)
        df = analyzer.analyze_directory(
            audio_dir  = args.input,
            output_dir = args.output_dir,
            windowed   = args.windowed,
            window_s   = args.window_s,
            plot       = not args.no_plot,
        )

        stats = analyzer.summary_stats(df)
        print("\nResumen estadístico:")
        for idx, s in stats.items():
            print(f"  {idx.upper():<6} μ={s['mean']:.4f}  σ={s['std']:.4f}"
                  f"  [{s['min']:.4f}, {s['max']:.4f}]  n={s['n']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
