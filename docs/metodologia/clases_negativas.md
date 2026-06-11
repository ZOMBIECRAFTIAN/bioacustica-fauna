# Protocolo de clases negativas y rechazo

## Objetivo

Evitar que el sistema fuerce una especie conocida cuando el audio contiene
ruido ambiental, actividad humana, silencio o fauna fuera del catalogo.

## Clases negativas minimas

| Clase | Descripcion | Fuente recomendada |
|---|---|---|
| `unknown_biological` | Sonidos biologicos fuera del catalogo | Campo propio, iNaturalist, bancos abiertos |
| `rain` | Lluvia ligera/fuerte, goteo | Campo propio, sound libraries con licencia |
| `wind` | Viento en vegetacion o microfono | Campo propio |
| `human_voice` | Voz humana lejana/cercana | Grabacion propia con consentimiento |
| `traffic` | Vehiculos, maquinaria, motores | Campo urbano/rural |
| `silence` | Segmentos sin energia biologica clara | Extraccion de ventanas silenciosas |

## Criterios de inclusion

1. Licencia o autoria documentada.
2. Etiqueta negativa dominante y verificable.
3. Duracion minima de 1 segundo.
4. Sin informacion personal sensible.
5. Metadatos minimos: fuente, fecha, sitio o descripcion de origen.

## Uso experimental

Las clases negativas deben participar en entrenamiento, validacion y prueba.
Tambien deben reportarse como:

- tasa de rechazo;
- falsos positivos de fauna sobre ruido;
- confusiones entre ruido y grupos biologicos;
- efecto del umbral `unknown_threshold`.

## Riesgo metodologico

No usar clases negativas produce metricas artificialmente altas en datasets
limpios, pero falla en campo. Para una tesis defendible, el modelo debe mostrar
cuando no sabe o cuando el audio no contiene una clase objetivo.
