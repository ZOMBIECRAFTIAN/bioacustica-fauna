# Índice Operativo de Tesis de Maestría

## Título de trabajo

Sistema multitaxonómico para detección e identificación de fauna silvestre mediante monitoreo acústico pasivo, con preprocesamiento adaptativo por grupo animal y evaluación en condiciones reales de campo.

## Capítulo 1. Introducción

### 1.1 Contexto

El monitoreo de fauna silvestre es una actividad central para conservación, manejo ambiental y evaluación de biodiversidad. Sin embargo, gran parte de los métodos tradicionales dependen de observación directa, recorridos de campo, cámaras trampa o revisión manual por especialistas.

El monitoreo acústico pasivo permite registrar actividad biológica de manera continua y no invasiva. Esta tecnología genera grandes volúmenes de audio que pueden contener cantos, llamados, estridulaciones, ecolocalizaciones y otros eventos sonoros producidos por fauna silvestre.

### 1.2 Planteamiento del problema

La revisión manual de grabaciones acústicas es costosa en tiempo y depende de expertos. Aunque existen sistemas avanzados para identificación de aves, la identificación automática multitaxonómica sigue siendo un reto debido a las diferencias acústicas entre grupos animales.

### 1.3 Pregunta de investigación

¿Puede un sistema bioacústico multitaxonómico con preprocesamiento adaptativo por grupo animal mejorar la detección e identificación de fauna silvestre acústicamente detectable frente a un enfoque único de procesamiento y clasificación?

### 1.4 Hipótesis

Un pipeline jerárquico con clasificación inicial de grupo acústico y posterior identificación especializada por grupo animal obtendrá mejor desempeño que un modelo único entrenado sobre todos los sonidos sin adaptación taxonómica.

### 1.5 Objetivos

El objetivo general y los objetivos específicos se mantienen en `docs/tesis_maestria.md`.

### 1.6 Alcance

La tesis se limita a fauna acústicamente detectable y a un dataset piloto. El alcance recomendado para la primera validación es:

- Anfibios anuros.
- Murciélagos.
- Aves como grupo comparativo.

Insectos y mamíferos vocales podrán incorporarse como extensión si existe disponibilidad suficiente de datos.

## Capítulo 2. Marco Teórico

### 2.1 Bioacústica

Estudio de los sonidos producidos por organismos vivos y su relación con comportamiento, comunicación, distribución y ecología.

### 2.2 Ecoacústica y paisaje sonoro

Análisis del ambiente acústico completo, incluyendo biofonía, geofonía y antropofonía.

### 2.3 Monitoreo acústico pasivo

Uso de grabadoras autónomas o estaciones de monitoreo para registrar audio ambiental sin intervención humana continua.

### 2.4 Grupos animales acústicamente detectables

- Anuros: llamados reproductivos y coros.
- Murciélagos: ecolocalización ultrasónica.
- Aves: cantos y llamados.
- Insectos: estridulación y señales repetitivas.
- Mamíferos vocales: llamadas audibles y señales sociales.

### 2.5 Representaciones de audio

- Forma de onda.
- Espectrograma lineal.
- Espectrograma Mel.
- MFCC.
- Índices acústicos.

### 2.6 Aprendizaje profundo aplicado a audio

Uso de CNN, modelos preentrenados y transferencia de aprendizaje sobre representaciones tiempo-frecuencia.

## Capítulo 3. Estado del Arte

### 3.1 Sistemas especializados en aves

Los identificadores de aves sirven como referencia metodológica, pero no resuelven el problema multitaxonómico propuesto.

### 3.2 Identificación acústica de anfibios

Los anuros son un grupo fuerte para monitoreo acústico por la relevancia ecológica de sus vocalizaciones.

### 3.3 Clasificación acústica de murciélagos

Los murciélagos presentan un reto técnico por la frecuencia ultrasónica y por la necesidad de grabadoras de alta tasa de muestreo.

### 3.4 Clasificación de insectos y mamíferos vocales

Son grupos prometedores, aunque con disponibilidad desigual de datos y mayor variabilidad acústica.

### 3.5 Brecha identificada

La mayoría de sistemas se especializan en un grupo. Esta tesis aborda un pipeline adaptativo para varios grupos acústicos.

## Capítulo 4. Metodología

### 4.1 Diseño general

Pipeline jerárquico:

1. Validación de audio.
2. Detección de eventos.
3. Clasificación de grupo acústico.
4. Preprocesamiento específico por grupo.
5. Clasificación de especie o clase taxonómica.
6. Evaluación por grupo.

### 4.2 Dataset

El protocolo detallado se define en `docs/metodologia/protocolo_dataset_multitaxon.md`.

### 4.3 Modelos

- CNN base.
- EfficientNet sobre espectrogramas.
- PANNs o modelo preentrenado de audio si los datos y recursos lo permiten.

### 4.4 Experimentos

- Modelo único.
- Modelo con preprocesamiento adaptativo.
- Modelo jerárquico.
- Evaluación con ruido o fuente no vista.

### 4.5 Métricas

- Accuracy.
- F1-macro.
- Precision y recall.
- Top-k accuracy.
- Matriz de confusión.
- Métricas por grupo animal.
- Latencia.

## Capítulo 5. Implementación

Este capítulo documentará el repositorio:

- `src/audio_processing/preprocessor.py`
- `src/data/dataset_builder.py`
- `src/models/train.py`
- `src/evaluation/evaluator.py`
- `src/api/main.py`
- `configs/train_multitaxon.yaml`

## Capítulo 6. Resultados

Los resultados deberán organizarse por experimento y por grupo animal. No se reportarán métricas sin dataset validado.

## Capítulo 7. Discusión

La discusión debe explicar no solo qué modelo ganó, sino por qué ciertos grupos fueron más difíciles, cómo afectó el ruido y qué limitaciones impone la disponibilidad de datos.

## Capítulo 8. Conclusiones

La conclusión deberá responder directamente la pregunta de investigación y delimitar el paso hacia doctorado.
