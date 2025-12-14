# 🐳 Docker Setup - PuLabs

## Construcción y ejecución rápida

### Opción 1: Docker Compose (Recomendado)

```bash
# Construir e iniciar
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener
docker-compose down

# Detener y eliminar volúmenes (⚠️ borra modelos entrenados)
docker-compose down -v
```

### Opción 2: Docker directo

```bash
# Construir imagen
docker build -t voice-controlled-games .

# Ejecutar contenedor
docker run -d \
  --name voice-controlled-games \
  -p 8000:8000 \
  -v $(pwd)/backend/models:/app/backend/models \
  voice-controlled-games

# Ver logs
docker logs -f voice-controlled-games

# Detener
docker stop voice-controlled-games
docker rm voice-controlled-games
```

## Acceso

Una vez iniciado el contenedor:

- **Frontend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/

## Volúmenes persistentes

Docker Compose crea dos volúmenes:

1. `voice_controlled_games_models`: Modelos entrenados y grabaciones
2. `voice_controlled_games_db`: Base de datos SQLite

Para inspeccionar:
```bash
docker volume ls
docker volume inspect voice_controlled_games_models
```

## Desarrollo

El `docker-compose.yml` monta el código como volumen, permitiendo hot-reload:

```bash
# Los cambios en ./backend y ./frontend se reflejan automáticamente
docker-compose up
```

## Variables de entorno

Puedes crear un archivo `.env`:

```env
PYTHONUNBUFFERED=1
LOG_LEVEL=DEBUG
```

Y cargarlo:

```bash
docker-compose --env-file .env up
```

## Troubleshooting

### Audio no funciona en Docker
El audio requiere acceso al dispositivo host. Para desarrollo local, usa el script `start.sh` nativo.

### Puerto 8000 ocupado
Cambia el mapeo de puertos en `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"  # Usar puerto 8080 en el host
```

### Reconstruir imagen
```bash
docker-compose build --no-cache
docker-compose up -d
```
