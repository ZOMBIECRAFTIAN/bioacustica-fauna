# Tesis de Maestría

## Título de trabajo

**Sistema multitaxonómico para detección e identificación de fauna silvestre mediante monitoreo acústico pasivo, con preprocesamiento adaptativo por grupo animal y evaluación en condiciones reales de campo.**

## Versión corta del título

**Sistema multitaxonómico para identificación bioacústica de fauna silvestre mediante monitoreo acústico pasivo.**

## Enfoque general

Esta tesis propone desarrollar y evaluar un sistema bioacústico multitaxonómico capaz de detectar e identificar fauna silvestre a partir de grabaciones de monitoreo acústico pasivo. El aporte central no es competir con sistemas especializados en aves, sino avanzar hacia una plataforma adaptable a diferentes grupos animales acústicamente detectables, como anfibios anuros, murciélagos, insectos, aves y mamíferos vocales.

El sistema se basa en una estrategia jerárquica: primero detectar eventos acústicos biológicos, luego estimar el grupo animal probable y finalmente aplicar preprocesamiento y clasificación adaptados al grupo acústico correspondiente.

## Documentos asociados

- `docs/tesis_indice.md`: índice operativo por capítulos.
- `docs/metodologia/protocolo_dataset_multitaxon.md`: protocolo inicial de curación del dataset.
- `configs/train_multitaxon.yaml`: configuración base para el primer experimento piloto.

## Problema de investigación

El monitoreo de fauna silvestre mediante métodos tradicionales suele requerir observación directa, captura, cámaras trampa o identificación manual de vocalizaciones. Estos métodos pueden ser costosos, dependientes de especialistas, limitados espacialmente y difíciles de escalar a monitoreos continuos.

El monitoreo acústico pasivo permite registrar paisajes sonoros durante largos periodos sin presencia humana constante. Sin embargo, el análisis manual de grandes volúmenes de audio representa un cuello de botella. Aunque existen herramientas avanzadas para aves, como identificadores especializados basados en audio, persiste una brecha para sistemas multitaxonómicos que integren distintos grupos animales con características acústicas muy diferentes.

Los llamados de anuros, las vocalizaciones audibles de mamíferos, los cantos de aves, las estridulaciones de insectos y las emisiones ultrasónicas de murciélagos presentan rangos de frecuencia, duraciones, patrones temporales y requisitos de muestreo distintos. Un único flujo de preprocesamiento y clasificación puede ser insuficiente para capturar esa diversidad.

## Pregunta de investigación

¿Puede un sistema bioacústico multitaxonómico con preprocesamiento adaptativo por grupo animal mejorar la detección e identificación de fauna silvestre acústicamente detectable frente a un enfoque único de procesamiento y clasificación?

## Hipótesis

Un pipeline jerárquico que primero distingue el grupo acústico y posteriormente aplica preprocesamiento y modelos específicos por grupo animal obtendrá mejor desempeño, especialmente en F1-macro y reducción de confusiones entre grupos, que un modelo único entrenado sobre todos los sonidos sin adaptación taxonómica.

## Objetivo general

Desarrollar y evaluar un sistema multitaxonómico para la detección e identificación automática de fauna silvestre acústicamente detectable mediante monitoreo acústico pasivo, incorporando preprocesamiento adaptativo por grupo animal y validación experimental en condiciones cercanas a campo.

## Objetivos específicos

1. Construir un dataset piloto multitaxonómico con grabaciones acústicas de fauna silvestre, documentando fuentes, licencias, criterios de inclusión y distribución por grupo animal.
2. Diseñar un pipeline de preprocesamiento adaptativo para distintos grupos acústicos, considerando frecuencia, duración de eventos, tasa de muestreo y reducción de ruido.
3. Implementar una estrategia jerárquica de clasificación compuesta por detección de evento, clasificación de grupo acústico e identificación de especie o categoría taxonómica.
4. Entrenar y comparar modelos de aprendizaje profundo, incluyendo una CNN base, EfficientNet sobre espectrogramas y modelos preentrenados para audio cuando sean viables.
5. Evaluar el desempeño del sistema con métricas globales y por grupo animal, incluyendo F1-macro, precisión, recall, top-k accuracy y matrices de confusión.
6. Analizar los principales errores del sistema en términos de ruido ambiental, desbalance de clases, similitud acústica entre especies y diferencias entre fuentes de grabación.
7. Implementar una API o prototipo demostrativo que permita clasificar archivos de audio y visualizar predicciones con sus niveles de confianza.

## Alcance para maestría

La tesis no busca resolver la identificación universal de todos los animales. El alcance adecuado para maestría es construir y evaluar un sistema piloto con grupos acústicamente relevantes y técnicamente contrastantes.

### Grupos recomendados

| Grupo | Rol en la tesis | Razón |
|---|---|---|
| Anfibios anuros | Grupo principal | Alta vocalización, importancia ecológica, buen caso para bioacústica |
| Murciélagos | Grupo técnico avanzado | Ultrasonido, tasas de muestreo altas, procesamiento distinto |
| Aves | Grupo comparativo | Buen baseline por disponibilidad de datos y sistemas existentes |
| Insectos | Extensión opcional | Útiles para paisaje sonoro, pero más complejos por ruido y variabilidad |
| Mamíferos vocales | Extensión opcional | Relevantes para monitoreo, pero con menor disponibilidad de vocalizaciones |

Para mantener una tesis defendible, se recomienda iniciar con tres grupos: anuros, murciélagos y aves. Insectos y mamíferos pueden incluirse como expansión si el dataset y el tiempo lo permiten.

## Fuera de alcance para maestría

- Identificar todas las especies animales posibles.
- Cubrir reptiles como grupo principal, salvo especies vocales específicas.
- Garantizar desempeño en cualquier región geográfica sin validación.
- Entrenar modelos robustos para todas las condiciones ambientales.
- Sustituir la validación de expertos biólogos o bioacústicos.

## Aporte esperado

El aporte principal de la tesis será una arquitectura experimental y reproducible para clasificación bioacústica multitaxonómica. La contribución no se limita a un modelo, sino al diseño completo del flujo:

- Curación de dataset multitaxonómico.
- Preprocesamiento adaptativo por grupo animal.
- Comparación entre clasificación plana y jerárquica.
- Evaluación por grupo taxonómico.
- Prototipo funcional para inferencia.
- Discusión de límites y expansión hacia doctorado.

## Arquitectura propuesta

```text
Audio de campo o repositorio
    ↓
Validación del archivo y normalización básica
    ↓
Detección de evento acústico
    ↓
Clasificación de grupo acústico
    ↓
Selección de preset:
  anuro | murciélago | ave | insecto | mamífero | desconocido
    ↓
Extracción de espectrograma y características
    ↓
Modelo específico o cabeza especializada
    ↓
Predicción top-k + probabilidad + clase unknown
    ↓
Registro de resultado y análisis
```

## Diseño experimental

### Experimento 1: Modelo único

Entrenar un modelo sobre todas las clases mezcladas usando un mismo preprocesamiento.

Propósito: establecer una línea base simple.

### Experimento 2: Preprocesamiento adaptativo

Entrenar modelos usando presets acústicos por grupo animal, manteniendo arquitectura comparable.

Propósito: medir si la adaptación de parámetros acústicos mejora el desempeño.

### Experimento 3: Clasificación jerárquica

Entrenar un primer clasificador de grupo acústico y luego clasificadores especializados por grupo.

Propósito: evaluar si separar grupo y especie reduce confusiones y mejora F1-macro.

### Experimento 4: Robustez ante ruido o campo

Evaluar el sistema con grabaciones más ruidosas, mezclas ambientales o datos de una fuente no vista durante entrenamiento.

Propósito: analizar generalización y limitaciones reales.

## Métricas de evaluación

- Accuracy global.
- F1-macro global.
- F1-macro por grupo animal.
- Precision y recall por clase.
- Top-3 y top-5 accuracy.
- Matriz de confusión global.
- Matriz de confusión por grupo.
- Tasa de rechazo o clase unknown.
- Latencia de inferencia por segmento.

## Criterios de éxito

El proyecto se considerará exitoso para maestría si demuestra:

1. Un dataset piloto documentado y reproducible.
2. Un pipeline funcional desde audio hasta predicción.
3. Comparación experimental entre al menos dos enfoques.
4. Mejora o análisis claro del enfoque adaptativo frente al enfoque único.
5. Resultados honestos, con métricas y discusión de errores.
6. Un prototipo demostrativo integrado al sistema existente.

## Estructura sugerida de capítulos

### Capítulo 1: Introducción

- Contexto de biodiversidad y monitoreo de fauna.
- Monitoreo acústico pasivo.
- Problema del análisis manual de audio.
- Limitación de sistemas especializados en un solo grupo.
- Planteamiento del sistema multitaxonómico.
- Pregunta, hipótesis y objetivos.

### Capítulo 2: Marco teórico

- Bioacústica y ecoacústica.
- Monitoreo acústico pasivo.
- Características acústicas por grupo animal.
- Espectrogramas, MFCC e índices acústicos.
- Aprendizaje profundo para audio.
- Transfer learning en bioacústica.
- Clasificación jerárquica y open-set recognition.

### Capítulo 3: Estado del arte

- Sistemas de identificación de aves.
- Identificación acústica de anuros.
- Clasificación de murciélagos por ultrasonido.
- Clasificación de insectos y mamíferos vocales.
- Modelos preentrenados para audio.
- Datasets bioacústicos existentes.
- Brecha: sistemas multitaxonómicos adaptativos.

### Capítulo 4: Metodología

- Diseño del dataset.
- Fuentes de datos y criterios de inclusión.
- Preprocesamiento por grupo animal.
- Arquitectura del sistema.
- Modelos evaluados.
- Estrategia de entrenamiento.
- Diseño experimental.
- Métricas.

### Capítulo 5: Implementación

- Estructura del proyecto.
- Módulos de audio.
- Pipeline de dataset.
- Modelos implementados.
- API de inferencia.
- Base de datos y registro de detecciones.
- Configuración reproducible.

### Capítulo 6: Resultados

- Descripción del dataset final.
- Resultados del modelo único.
- Resultados con preprocesamiento adaptativo.
- Resultados del enfoque jerárquico.
- Comparación por grupo animal.
- Análisis de errores.
- Latencia y viabilidad de despliegue.

### Capítulo 7: Discusión

- Interpretación de resultados.
- Fortalezas del enfoque multitaxonómico.
- Limitaciones por datos, ruido y sesgo geográfico.
- Comparación con enfoques especializados.
- Implicaciones para monitoreo de fauna.
- Camino hacia doctorado.

### Capítulo 8: Conclusiones

- Respuesta a la pregunta de investigación.
- Cumplimiento de objetivos.
- Contribuciones.
- Trabajo futuro.

## Plan de trabajo

| Fase | Actividad | Resultado |
|---|---|---|
| 1 | Definición final de alcance | Grupos y especies objetivo |
| 2 | Curación de dataset piloto | Dataset documentado |
| 3 | Preprocesamiento adaptativo | Presets validados por grupo |
| 4 | Entrenamiento baseline | Modelo único y métricas iniciales |
| 5 | Entrenamiento adaptativo/jerárquico | Comparación experimental |
| 6 | Evaluación y análisis de errores | Resultados de tesis |
| 7 | Prototipo/API | Demostración funcional |
| 8 | Escritura final | Documento de tesis |

## División futura para doctorado

La maestría debe cerrar con un prototipo validado y resultados medibles. El doctorado puede ampliar la investigación hacia:

- Generalización entre regiones, estaciones y tipos de micrófono.
- Aprendizaje con pocos datos para especies subrepresentadas.
- Modelos fundacionales para bioacústica multitaxonómica.
- Detección de especies fuera del catálogo.
- Estaciones acústicas autónomas con edge AI.
- Dataset abierto de fauna acústicamente detectable en México o Latinoamérica.

## Nombre del proyecto dentro del repositorio

Nombre técnico recomendado:

**Bioacústica Fauna: sistema multitaxonómico de monitoreo acústico pasivo**

Nombre académico recomendado:

**Sistema multitaxonómico para identificación bioacústica de fauna silvestre**

## Próximos pasos inmediatos

1. Reescribir el README para que el enfoque principal sea multitaxonómico y no aves de México.
2. Crear perfiles de dataset por grupo: anuros, murciélagos, aves, insectos y mamíferos vocales.
3. Crear una configuración `train_multitaxon.yaml`.
4. Añadir evaluación por grupo animal.
5. Añadir clase `unknown` y clases de ruido ambiental.
6. Separar el prototipo web de la API científica.
7. Preparar una carpeta de resultados para experimentos de maestría.
