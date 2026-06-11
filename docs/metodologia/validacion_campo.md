# Protocolo de validacion en condiciones reales de campo

## Objetivo

Evaluar el desempeno del sistema fuera del conjunto usado para entrenamiento,
con audios de campo o pseudo-campo que representen ruido, variacion de
microfonos y condiciones ambientales reales.

## Diseno minimo para maestria

| Elemento | Minimo | Recomendado |
|---|---:|---:|
| Grabaciones externas | 10 | 30 o mas |
| Sitios | 1 | 3 o mas |
| Noches/dias de muestreo | 1 | 3 o mas |
| Duracion por grabacion | 30 s | 1 a 5 min |
| Clases negativas | 3 | 6 |

## Metadatos requeridos

Cada grabacion debe registrarse en `field_manifest.csv` con:

- `recording_id`
- `site_id`
- `date`
- `recorder`
- `microphone`
- `habitat`
- `weather`
- `duration_s`
- `dominant_noise`
- `expected_species_or_group`
- `license_or_permission`

## Separacion experimental

Estas grabaciones no deben usarse para entrenar ni ajustar hiperparametros.
Funcionan como test externo para medir cambio de dominio.

## Reporte

El informe debe incluir:

1. F1-macro y recall por grupo en datos externos.
2. Falsos positivos sobre ruido.
3. Casos donde el sistema responde `unknown` o baja confianza.
4. Diferencia contra test interno.
5. Analisis cualitativo de errores.

## Interpretacion

Si el desempeno baja en campo, no es fracaso: es evidencia de cambio de dominio.
La tesis debe discutir esa brecha y proponer como reducirla con mas datos,
mejor muestreo, calibracion de umbrales o adaptacion por sitio.
