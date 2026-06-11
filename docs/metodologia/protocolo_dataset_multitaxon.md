# Protocolo de Dataset Multitaxonómico

## Propósito

Definir los criterios para construir un dataset piloto de fauna silvestre acústicamente detectable para la tesis de maestría.

El dataset debe permitir comparar un enfoque de clasificación único contra un enfoque adaptativo por grupo animal.

## Grupos objetivo iniciales

| Grupo | Prioridad | Uso en tesis |
|---|---:|---|
| Anfibios anuros | Alta | Grupo principal |
| Murciélagos | Alta | Grupo técnicamente contrastante |
| Aves | Media | Grupo comparativo y baseline |
| Insectos | Opcional | Extensión si hay datos suficientes |
| Mamíferos vocales | Opcional | Extensión si hay datos suficientes |

## Criterios de inclusión

Una grabación podrá incluirse si cumple:

1. Tiene licencia compatible con investigación y documentación de fuente.
2. Incluye identificación taxonómica suficientemente confiable.
3. Contiene audio legible y con duración mínima de 1 segundo.
4. Está asociada a una especie o clase objetivo del piloto.
5. No presenta saturación extrema que impida el análisis.
6. Puede transformarse a un formato procesable por el pipeline.

## Criterios de exclusión

Se excluirán grabaciones si:

- No existe licencia o fuente clara.
- El archivo no puede abrirse con `soundfile` o `librosa`.
- La especie no coincide con el perfil objetivo.
- El audio está dominado por voz humana, música o ruido mecánico.
- La duración excede el límite definido y no se puede segmentar adecuadamente.
- La tasa de muestreo es insuficiente para el grupo, especialmente en murciélagos.

## Fuentes de datos

| Fuente | Uso esperado |
|---|---|
| Xeno-canto | Aves, anfibios, algunos insectos y mamíferos |
| iNaturalist / NaturaLista | Multitaxonómico, especialmente registros con sonido |
| GBIF | Descubrimiento de ocurrencias con multimedia |
| ChiroVox u otras bases de murciélagos | Murciélagos si el acceso/licencia lo permite |
| Grabaciones propias de campo | Validación real o semi-real |

## Estructura de carpetas

```text
data/raw/
  anurans/
    especie_1/
    especie_2/
  bats/
    especie_1/
    especie_2/
  birds/
    especie_1/
    especie_2/
  insects/
  mammals/
  unknown/
  noise/
```

La estructura final usada por entrenamiento puede ser plana por clase si lo requiere `SpectrogramDataset`, pero el manifiesto debe preservar el grupo animal.

## Metadatos mínimos

Cada archivo debe tener o derivar metadatos:

- `source`
- `source_url`
- `license`
- `scientific_name`
- `class_label`
- `acoustic_group`
- `country`
- `date`
- `recordist`
- `sample_rate`
- `duration_s`
- `original_format`

## Balance recomendado

Para maestría, un dataset piloto defendible puede iniciar con:

- 3 grupos principales.
- 5 a 10 clases por grupo.
- 50 a 300 grabaciones por clase, según disponibilidad.

Si las clases no alcanzan el mínimo, se podrá trabajar a nivel de género o grupo acústico en vez de especie.

## Clases negativas y desconocidas

El sistema debe incluir ejemplos no objetivo:

- `unknown_biological`
- `rain`
- `wind`
- `human_voice`
- `traffic`
- `silence`

Estas clases ayudan a evitar que el modelo fuerce una especie conocida ante cualquier audio.

## Split experimental

La versión defendible para tesis debe usar split agrupado:

- 70% entrenamiento.
- 15% validación.
- 15% prueba.
- Unidad de separación: archivo original, fuente, sitio o fecha.

Regla central: segmentos derivados del mismo audio original no pueden aparecer
simultáneamente en entrenamiento y prueba. El archivo
`dataset_manifest.csv` debe registrar `original_file_id` y `split_group` para
probar esta separación.

El split estratificado simple solo puede usarse como diagnóstico rápido, no como
resultado principal de tesis.

## Control de sesgo

Se debe vigilar:

- desbalance por especie;
- repetición del mismo grabador o sitio;
- clases con demasiados audios limpios frente a clases ruidosas;
- mezcla de fuentes con calidades muy diferentes;
- posible aprendizaje de artefactos de grabación en vez de señales biológicas.

## Reporte del dataset

Cada experimento debe registrar:

- número de clases;
- número de archivos por clase;
- duración total por grupo;
- duración promedio;
- distribución de fuentes;
- porcentaje de archivos descartados;
- criterios de descarte;
- fecha de construcción del dataset.

## Comandos base

```bash
python -m src.data.dataset_builder --profile mexico_anurans --output data/raw/anurans
python -m src.data.dataset_builder --profile mexico_bats --output data/raw/bats
python -m src.data.dataset_builder --profile mexico_multitaxon --output data/raw/multitaxon
```

Después:

```bash
python -m src.feature_extraction.batch_extractor \
  --input data/raw/multitaxon \
  --output data/spectrograms/multitaxon \
  --mfcc-dir data/features/mfcc/multitaxon \
  --preset adaptive

python -m src.data.manifest \
  --raw-dir data/raw/multitaxon \
  --spectrogram-dir data/spectrograms/multitaxon
```

## Resultado esperado

Un dataset piloto trazable, reproducible y suficientemente documentado para sostener los experimentos de la tesis.
