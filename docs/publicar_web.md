# Publicar la web del identificador

## Local

```powershell
.\scripts\start_web.ps1
```

Abre la URL que imprima el script, normalmente:

```text
http://127.0.0.1:8000/
```

Para detener:

```powershell
.\scripts\stop_web.ps1
```

## En la red local

Para probar desde celular en la misma red Wi-Fi:

```powershell
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Luego abre desde el celular:

```text
http://IP_DE_TU_PC:8000/
```

## En internet

Opciones recomendadas:

- Render, Railway o Fly.io para prototipo.
- VPS con Docker para control total.
- Azure/AWS/GCP si se quiere escalar.

Variables importantes:

```text
MODEL_PATH=models/trained/multitaxon/best_efficientnet.pt
MODEL_TYPE=efficientnet
MODEL_DEVICE=cpu
MAX_FILE_MB=50
```

Notas:

- El modelo `.pt` puede pesar bastante; revisar limites de la plataforma.
- Para uso publico conviene agregar cola de trabajos, limites por IP y almacenamiento de audios separado.
- La base de datos aun es opcional; la web guarda historial y feedback en `data/app/*.jsonl`.
