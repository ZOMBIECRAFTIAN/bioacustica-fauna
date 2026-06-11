# Diseno experimental para maestria

## Pregunta operativa

Evaluar si un sistema multitaxonomico con preprocesamiento adaptativo y una
estructura jerarquica mejora la identificacion de fauna acusticamente detectable
frente a un modelo unico entrenado sobre todas las clases.

## Experimentos minimos

| Experimento | Configuracion | Proposito |
|---|---|---|
| E1 plano | `configs/train_multitaxon.yaml` con features comunes | Baseline principal |
| E2 adaptativo | `batch_extractor --preset adaptive` + `configs/train_multitaxon.yaml` | Medir efecto del preprocesamiento por grupo |
| E3 jerarquico etapa 1 | `configs/train_multitaxon_stage1_group.yaml` | Clasificar grupo acustico |
| E4 jerarquico etapa 2 | `configs/train_multitaxon_stage2_group_template.yaml` | Especialistas por grupo |
| E5 campo externo | `docs/metodologia/validacion_campo.md` | Medir cambio de dominio |

## Controles obligatorios

1. Usar `dataset_manifest.csv`.
2. Usar split `source_file`, `site` o `source`; no reportar como final un split por segmento.
3. Registrar `split_manifest.csv` por experimento.
4. Incluir clases negativas reales.
5. Reportar metricas globales y por grupo animal.
6. Documentar errores con `error_analysis_template.csv`.

## Metricas principales

- F1-macro global.
- F1-macro por grupo animal.
- Recall por clase.
- Top-3 y top-5 accuracy.
- Matriz de confusion global.
- Matriz de confusion por grupo.
- Tasa de rechazo/unknown.
- Diferencia entre test interno y validacion externa.

## Criterio de interpretacion

El objetivo no es prometer una metrica fija. El resultado defendible es mostrar
si la adaptacion por grupo y la clasificacion jerarquica reducen confusiones,
especialmente entre grupos acusticamente distintos como anuros, murcielagos y
aves.
