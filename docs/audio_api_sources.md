# Fuentes y APIs para audios bioacusticos

Estas fuentes sirven para ampliar el dataset de aves de Mexico y despues sumar mamiferos, anfibios, reptiles e insectos. Antes de entrenar con audios nuevos, revisar licencia, atribucion y restricciones de uso por archivo.

| Fuente | URL | Uso recomendado |
| --- | --- | --- |
| Xeno-canto | https://xeno-canto.org/explore/api | Principal para aves; tambien puede incluir ranas, murcielagos, mamiferos terrestres y ortopteros. API v3 requiere key. |
| iNaturalist / NaturaLista | https://api.inaturalist.org/v2/docs/ | Muy buena para Mexico y varios grupos. Buscar observaciones con `sounds=true`. |
| GBIF Occurrence API | https://techdocs.gbif.org/en/openapi/v1/occurrence | Agregador de ocurrencias con multimedia. Filtrar por `mediaType=Sound` y `country=MX`. |
| Wikimedia Commons | https://commons.wikimedia.org/wiki/Category:Audio_files_of_animals | Complemento con audios abiertos. Revisar licencia por archivo. |
| EcoSounds | https://api.ecosounds.org/ | Soundscapes y proyectos ecoacusticos. Revisar permisos por proyecto. |
| audioBlast | https://audioblast.org/ | Busqueda y descubrimiento bioacustico. Revisar licencia y fuente original. |
| Animal Sound Archive Berlin | https://www.museumfuernaturkunde.berlin/en/research/animal-sound-archive | Archivo grande de sonidos animales; parte del material puede aparecer via GBIF. |

Ejemplos utiles:

```text
https://api.inaturalist.org/v1/observations?place_id=6793&taxon_id=3&sounds=true&quality_grade=research&per_page=200
https://api.gbif.org/v1/occurrence/search?country=MX&mediaType=Sound&limit=300
https://xeno-canto.org/api/3/recordings?query=grp:birds+cnt:mexico&key=TU_KEY
```

Comandos del proyecto:

```powershell
# Descargar hasta 50 audios por especie usando las tres fuentes configuradas.
python -m src.data.dataset_builder --profile mexico_birds --max-per-class 50 --max-duration 180

# Usar solo iNaturalist y GBIF, sin Xeno-canto.
python -m src.data.dataset_builder --profile mexico_birds --sources inaturalist,gbif --max-per-class 50 --max-duration 180

# Validar dataset descargado.
python -m src.data.dataset_builder --output data/raw --validate
```

Notas:

- `place_id=6793` corresponde a Mexico en iNaturalist.
- `taxon_id=3` corresponde a Aves en iNaturalist.
- Para mamiferos, anfibios y reptiles conviene obtener primero los `taxon_id` de iNaturalist y luego filtrar con `sounds=true`.
- GBIF es excelente para encontrar metadatos, fuente original, licencia y enlaces multimedia, pero no siempre hospeda el audio directamente.
