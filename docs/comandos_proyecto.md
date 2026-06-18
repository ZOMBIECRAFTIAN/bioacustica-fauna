# Comandos principales del proyecto

Estos comandos estan pensados para Anaconda Prompt en Windows.

## Activar entorno

```bat
conda activate bioacustica
cd /d D:\bioacustica-fauna
```

## Verificar el proyecto

```bat
scripts\verify_local.bat
```

Equivalente manual:

```bat
black --check --diff src tests scripts
ruff check --no-cache src tests scripts --output-format=github
python scripts\verify_scientific_readiness.py
```

## Ejecutar pipeline piloto de maestria

```bat
scripts\run_master_pipeline.bat
```

## Ver especies y audios objetivo

```bat
python scripts\list_target_species.py
```

Guia detallada:

```text
docs\metodologia\especies_objetivo_maestria.md
```

Pasos manuales:

```bat
python -m src.data.dataset_builder --profile mexico_multitaxon --output data/raw/multitaxon --max-per-class 150

python -m src.feature_extraction.batch_extractor --input data/raw/multitaxon --output data/spectrograms/multitaxon --mfcc-dir data/features/mfcc/multitaxon --preset adaptive --workers 1

python -m src.data.manifest --raw-dir data/raw/multitaxon --spectrogram-dir data/spectrograms/multitaxon

python -m src.models.train --config configs/train_multitaxon.yaml --device cpu --epochs 3
```

## Entrenar experimentos comparativos

Baseline plano:

```bat
python -m src.feature_extraction.batch_extractor --input data/raw/multitaxon --output data/spectrograms/multitaxon_flat --preset multitaxon --workers 1
python -m src.data.manifest --raw-dir data/raw/multitaxon --spectrogram-dir data/spectrograms/multitaxon_flat
python -m src.models.train --config configs/train_multitaxon_flat.yaml --device cpu --epochs 3
```

Adaptativo:

```bat
python -m src.models.train --config configs/train_multitaxon.yaml --device cpu --epochs 3
```

Jerarquico etapa 1:

```bat
python -m src.models.train --config configs/train_multitaxon_stage1_group.yaml --device cpu --epochs 3
```

## Levantar API

```bat
scripts\run_api.bat
```

Abrir:

```text
http://localhost:8000/docs
```

Interfaz principal:

```text
http://localhost:8000/
```

Interfaz para capturar metadatos, validacion de campo y clases negativas:

```text
http://localhost:8000/data-entry
```

En esa pantalla puedes usar:

```text
Descargar CSV      -> baja el CSV visible al navegador
Guardar proyecto   -> guarda en data/app/data_entry/*.csv
Cargar proyecto    -> recarga desde data/app/data_entry/*.csv
Enviar API         -> intenta registrar una grabacion en /recordings
```

Nota: `data/app/` esta ignorado por Git para evitar subir coordenadas, sitios
sensibles o registros de campo privados por accidente.

## Docker

```bat
docker compose up -d
docker compose logs -f api
```

## Subir a GitHub

```bat
git status
git add .gitattributes .gitignore .env.example README.md docker-compose.yml configs docs results scripts src tests
git commit -m "Improve multitaxon thesis reproducibility and run commands"
git push
```
