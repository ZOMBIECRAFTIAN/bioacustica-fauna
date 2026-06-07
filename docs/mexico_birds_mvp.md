# MVP: Aves de Mexico

Este perfil convierte el proyecto en un primer sistema entrenable para identificar aves de Mexico por audio.

## Especies objetivo iniciales

El perfil `mexico_birds` incluye 20 especies:

- Quiscalus mexicanus
- Turdus grayi
- Pitangus sulphuratus
- Myiozetetes similis
- Melanerpes aurifrons
- Campylorhynchus brunneicapillus
- Thryophilus sinaloa
- Icterus pustulatus
- Toxostoma curvirostre
- Zenaida asiatica
- Columbina inca
- Ortalis vetula
- Crotophaga sulcirostris
- Momotus lessonii
- Glaucidium brasilianum
- Geococcyx californianus
- Haemorhous mexicanus
- Setophaga petechia
- Vireo hypochryseus
- Cyanocorax yncas

## Flujo recomendado

```bash
$env:XENO_CANTO_API_KEY="tu_key"

python -m src.data.dataset_builder --profile mexico_birds --output data/raw --max-per-class 300

python -m src.feature_extraction.batch_extractor \
  --input data/raw \
  --output data/spectrograms \
  --preset birds \
  --workers 4

python -m src.models.train --config configs/train_mexico_birds.yaml
```

## Notas de calidad

- Revisa manualmente una muestra por especie antes de entrenar.
- Agrega clases negativas: silencio, viento, lluvia, humanos, ruido urbano y aves fuera del catalogo.
- Para un modelo usable en campo, no aceptes predicciones de baja confianza como especie; tratalas como `unknown`.
- Mantén separados los sitios o autores entre train/test cuando sea posible para evitar fuga de datos.
