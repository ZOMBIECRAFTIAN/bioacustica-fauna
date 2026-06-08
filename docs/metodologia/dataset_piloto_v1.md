# Dataset Piloto Multitaxonómico v1

## Fecha de creación

2026-06-07

## Comando ejecutado

```bash
python -m src.data.dataset_builder \
    --profile mexico_multitaxon \
    --output data/raw/multitaxon \
    --max-per-class 10 \
    --min-quality B
```

## Resultado general

- Total descargado y validado: **126 audios**.
- Clases con al menos un archivo válido: **16**.
- Clases vacías tras descarga: **6**.
- Archivos corruptos tras validación: **0**.
- Tamaño aproximado del dataset descargado: **474 MB**.

## Extracción de espectrogramas

Comando ejecutado:

```bash
python -m src.feature_extraction.batch_extractor \
    --input data/raw/multitaxon \
    --output data/spectrograms/multitaxon \
    --mfcc-dir data/features/mfcc/multitaxon \
    --preset multitaxon \
    --workers 1 \
    --overwrite
```

Resultado:

- Audios procesados correctamente: **125/126**.
- Errores de lectura: **1** (`gbif_5291752104_0.mp3` en `incilius_valliceps`).
- Segmentos generados: **2,606**.
- Clases con segmentos: **16**.
- Segmentos mínimos por clase: **10**.
- Segmentos máximos por clase: **559**.
- Razón de desbalance por segmentos: **55.9**.

## Conteo por clase

| Clase | Grupo acústico | Archivos válidos |
|---|---|---:|
| alouatta_palliata | mammals | 10 |
| campylorhynchus_brunneicapillus | birds | 10 |
| canis_latrans | mammals | 10 |
| eleutherodactylus_cystignathoides | frogs | 7 |
| eptesicus_fuscus | bats | 0 |
| gryllus_assimilis | insects | 0 |
| incilius_valliceps | frogs | 9 |
| lithobates_berlandieri | frogs | 3 |
| lithobates_forreri | frogs | 0 |
| melanerpes_aurifrons | birds | 10 |
| myiozetetes_similis | birds | 10 |
| myotis_velifer | bats | 0 |
| neoconocephalus_triops | insects | 4 |
| nyctinomops_macrotis | bats | 0 |
| oecanthus_niveus | insects | 0 |
| pitangus_sulphuratus | birds | 10 |
| procyon_lotor | mammals | 1 |
| quiscalus_mexicanus | birds | 10 |
| rhinella_horribilis | frogs | 10 |
| smilisca_baudinii | frogs | 10 |
| tadarida_brasiliensis | bats | 2 |
| turdus_grayi | birds | 10 |

## Conteo por grupo

| Grupo acústico | Archivos válidos | Observación |
|---|---:|---|
| birds | 60 | Grupo comparativo fuerte |
| frogs | 39 | Grupo principal viable, requiere balance |
| mammals | 21 | Viable para mamíferos vocales seleccionados |
| insects | 4 | Insuficiente para entrenamiento |
| bats | 2 | Insuficiente; requiere fuentes ultrasónicas especializadas |

## Conteo de segmentos por clase

| Clase | Segmentos |
|---|---:|
| tadarida_brasiliensis | 10 |
| lithobates_berlandieri | 15 |
| procyon_lotor | 41 |
| neoconocephalus_triops | 42 |
| eleutherodactylus_cystignathoides | 52 |
| incilius_valliceps | 111 |
| melanerpes_aurifrons | 124 |
| canis_latrans | 126 |
| rhinella_horribilis | 152 |
| campylorhynchus_brunneicapillus | 168 |
| alouatta_palliata | 171 |
| smilisca_baudinii | 192 |
| pitangus_sulphuratus | 252 |
| myiozetetes_similis | 286 |
| turdus_grayi | 305 |
| quiscalus_mexicanus | 559 |

## Interpretación

Este primer corte confirma que el enfoque multitaxonómico es viable, pero también muestra que no todos los grupos pueden construirse con la misma estrategia de descarga.

Las aves tienen alta disponibilidad y pueden funcionar como grupo comparativo. Los anuros tienen suficientes datos iniciales para sostener el eje principal de la tesis, aunque algunas clases deben reemplazarse o ampliarse. Los mamíferos vocales son prometedores si se restringe el alcance a especies con vocalizaciones frecuentes. Los murciélagos no deben entrenarse todavía con este dataset, porque las fuentes generales entregan muy pocos audios y no garantizan ultrasonido útil. Los insectos requieren selección de especies con mejor cobertura acústica.

El conteo por segmentos es útil para entrenamiento, pero no debe confundirse con independencia estadística. Muchos segmentos provienen del mismo archivo original; por lo tanto, la evaluación final de tesis debe evitar que segmentos de una misma grabación aparezcan simultáneamente en entrenamiento y prueba.

## Decisión para el siguiente corte

Para el entrenamiento piloto inicial se recomienda usar solo clases con al menos 5 archivos:

- `smilisca_baudinii`
- `rhinella_horribilis`
- `incilius_valliceps`
- `eleutherodactylus_cystignathoides`
- `quiscalus_mexicanus`
- `turdus_grayi`
- `pitangus_sulphuratus`
- `myiozetetes_similis`
- `melanerpes_aurifrons`
- `campylorhynchus_brunneicapillus`
- `alouatta_palliata`
- `canis_latrans`

Esto produce un primer conjunto entrenable de **12 clases** distribuidas en tres grupos: anuros, aves y mamíferos vocales.

## Acciones siguientes

1. Generar espectrogramas en `data/spectrograms/multitaxon`.
2. Ejecutar un entrenamiento piloto con `configs/train_multitaxon.yaml`.
3. Aumentar anuros hasta al menos 20-50 audios por clase.
4. Reemplazar especies con baja cobertura.
5. Buscar fuentes especializadas para murciélagos antes de incluirlos en entrenamiento.
6. Agregar clases negativas: lluvia, viento, voz humana, tráfico y silencio.
7. Implementar split por archivo/fuente para evitar fuga de segmentos entre entrenamiento y prueba.
