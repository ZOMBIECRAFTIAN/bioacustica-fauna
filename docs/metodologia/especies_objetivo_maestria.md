# Especies objetivo para la maestria

Este documento separa tres listas que suelen confundirse:

1. **Modelo actual en disco**: clases que puede devolver la API si usa
   `models/trained/multitaxon/best_efficientnet.pt`.
2. **Perfil objetivo `mexico_multitaxon`**: especies que el constructor de
   dataset intenta descargar para el piloto.
3. **Clases negativas**: sonidos que conviene subir para que el sistema aprenda
   a rechazar ruido o audios fuera del catalogo.

## Modelo actual en disco

El archivo `models/trained/multitaxon/class_names.json` contiene 16 clases. Esas
son las etiquetas que el modelo multitaxon actual puede predecir.

| Clase | Nombre cientifico | Grupo | Prioridad de audios |
|---|---|---|---|
| `alouatta_palliata` | Alouatta palliata | mammals | Alta |
| `campylorhynchus_brunneicapillus` | Campylorhynchus brunneicapillus | birds | Media |
| `canis_latrans` | Canis latrans | mammals | Alta |
| `eleutherodactylus_cystignathoides` | Eleutherodactylus cystignathoides | frogs | Alta |
| `incilius_valliceps` | Incilius valliceps | frogs | Alta |
| `lithobates_berlandieri` | Lithobates berlandieri | frogs | Alta |
| `melanerpes_aurifrons` | Melanerpes aurifrons | birds | Media |
| `myiozetetes_similis` | Myiozetetes similis | birds | Media |
| `neoconocephalus_triops` | Neoconocephalus triops | insects | Baja, exploratoria |
| `pitangus_sulphuratus` | Pitangus sulphuratus | birds | Media |
| `procyon_lotor` | Procyon lotor | mammals | Alta |
| `quiscalus_mexicanus` | Quiscalus mexicanus | birds | Media |
| `rhinella_horribilis` | Rhinella horribilis | frogs | Alta |
| `smilisca_baudinii` | Smilisca baudinii | frogs | Alta |
| `tadarida_brasiliensis` | Tadarida brasiliensis | bats | Alta, requiere ultrasonido |
| `turdus_grayi` | Turdus grayi | birds | Media |

Nota importante: el entrenamiento actual fue piloto. Su historial muestra baja
validacion inicial, asi que debe considerarse demostracion tecnica, no modelo
final defendible.

## Subconjunto recomendado para primer entrenamiento defendible

Para una primera version de maestria conviene concentrarse en clases con mejor
disponibilidad de audio y balancear antes de ampliar.

| Grupo | Clases recomendadas |
|---|---|
| frogs | `smilisca_baudinii`, `rhinella_horribilis`, `incilius_valliceps`, `eleutherodactylus_cystignathoides`, `lithobates_berlandieri` |
| birds | `quiscalus_mexicanus`, `turdus_grayi`, `pitangus_sulphuratus`, `myiozetetes_similis`, `melanerpes_aurifrons`, `campylorhynchus_brunneicapillus` |
| mammals | `alouatta_palliata`, `canis_latrans`, `procyon_lotor` |

Meta minima recomendada:

- 20 audios por clase para un piloto rapido.
- 50 audios por clase para una defensa mas seria.
- 100 o mas audios por clase para un resultado mas estable.

## Perfil completo `mexico_multitaxon`

Este perfil lo define `src/data/dataset_builder.py` y contiene 22 especies:

| Clase | Nombre cientifico | Grupo |
|---|---|---|
| `smilisca_baudinii` | Smilisca baudinii | frogs |
| `rhinella_horribilis` | Rhinella horribilis | frogs |
| `incilius_valliceps` | Incilius valliceps | frogs |
| `lithobates_berlandieri` | Lithobates berlandieri | frogs |
| `lithobates_forreri` | Lithobates forreri | frogs |
| `eleutherodactylus_cystignathoides` | Eleutherodactylus cystignathoides | frogs |
| `tadarida_brasiliensis` | Tadarida brasiliensis | bats |
| `myotis_velifer` | Myotis velifer | bats |
| `eptesicus_fuscus` | Eptesicus fuscus | bats |
| `nyctinomops_macrotis` | Nyctinomops macrotis | bats |
| `quiscalus_mexicanus` | Quiscalus mexicanus | birds |
| `turdus_grayi` | Turdus grayi | birds |
| `pitangus_sulphuratus` | Pitangus sulphuratus | birds |
| `myiozetetes_similis` | Myiozetetes similis | birds |
| `melanerpes_aurifrons` | Melanerpes aurifrons | birds |
| `campylorhynchus_brunneicapillus` | Campylorhynchus brunneicapillus | birds |
| `gryllus_assimilis` | Gryllus assimilis | insects |
| `neoconocephalus_triops` | Neoconocephalus triops | insects |
| `oecanthus_niveus` | Oecanthus niveus | insects |
| `alouatta_palliata` | Alouatta palliata | mammals |
| `canis_latrans` | Canis latrans | mammals |
| `procyon_lotor` | Procyon lotor | mammals |

## Clases negativas

Sube tambien audios de rechazo. Son tan importantes como las especies positivas.

| Clase | Que audio subir |
|---|---|
| `unknown_biological` | Sonidos biologicos no identificados o fuera del catalogo |
| `rain` | Lluvia leve, media y fuerte |
| `wind` | Viento en vegetacion, rafagas, microfono saturado por viento |
| `human_voice` | Voces humanas o conversaciones de fondo |
| `traffic` | Motores, trafico urbano, maquinaria |
| `silence` | Segmentos sin vocalizacion clara |

## Prioridad practica para subir audios

1. Completar anuros: `smilisca_baudinii`, `rhinella_horribilis`,
   `incilius_valliceps`, `eleutherodactylus_cystignathoides`.
2. Completar mamiferos vocales: `alouatta_palliata`, `canis_latrans`,
   `procyon_lotor`.
3. Mantener aves como grupo comparativo.
4. Subir negativas variadas desde campo.
5. Dejar murcielagos e insectos como exploratorios hasta reunir audios
   suficientes y bien documentados.

## Carpeta esperada

Cada clase debe tener su propia carpeta:

```text
data/raw/multitaxon/
  smilisca_baudinii/
  rhinella_horribilis/
  canis_latrans/
  rain/
  wind/
```

Cada archivo debe registrarse despues en la interfaz de captura:

```text
http://localhost:8000/data-entry
```
