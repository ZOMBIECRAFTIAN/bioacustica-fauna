# Plantillas de resultados -- Maestria Multitaxon

Estas plantillas definen la evidencia minima que debe acompanar cada experimento
de tesis. Los resultados reales generados por entrenamiento pueden vivir en
`results/maestria_multitaxon/`, pero esta carpeta versiona la estructura
esperada para que el experimento sea reproducible.

Archivos esperados por corrida:

- `metrics_template.csv`: resumen por experimento.
- `per_group_metrics_template.csv`: desempeno por grupo animal.
- `confusion_by_group_template.csv`: matriz colapsada por grupo.
- `experiment_registry_template.csv`: trazabilidad de configuracion/dataset.
- `field_validation_template.csv`: evaluacion externa en campo o pseudo-campo.
- `error_analysis_template.csv`: analisis cualitativo de errores.

Regla metodologica: no reportar una metrica como resultado final si no puede
rastrearse al commit, configuracion, dataset manifest y split manifest usados.
