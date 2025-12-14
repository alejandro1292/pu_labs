"""
Base de datos SQLite para persistencia de keywords y samples.
"""
import aiosqlite
import logging

# Forzar nivel INFO para aiosqlite
logging.getLogger("aiosqlite").setLevel(logging.INFO)
from pathlib import Path
from typing import List, Dict, Optional
import json

DB_PATH = Path("backend/models/keywords.db")


async def init_db():
    """Inicializa la base de datos con las tablas necesarias."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(DB_PATH) as db:
        # Tabla de keywords
        await db.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                n_templates INTEGER DEFAULT 0,
                needs_training BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de samples
        await db.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                duration REAL NOT NULL,
                sample_rate INTEGER DEFAULT 16000,
                file_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (keyword_id) REFERENCES keywords(id) ON DELETE CASCADE
            )
        """)
        
        # Tabla de thresholds
        await db.execute("""
            CREATE TABLE IF NOT EXISTS thresholds (
                keyword_id INTEGER PRIMARY KEY,
                threshold_min REAL NOT NULL,
                threshold_max REAL NOT NULL,
                mean_distance REAL,
                std_distance REAL,
                min_distance REAL,
                max_distance REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (keyword_id) REFERENCES keywords(id) ON DELETE CASCADE
            )
        """)
        
        # Índices
        await db.execute("CREATE INDEX IF NOT EXISTS idx_samples_keyword ON samples(keyword_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_keywords_name ON keywords(name)")
        
        # Migración: agregar columna needs_training si no existe
        try:
            await db.execute("ALTER TABLE keywords ADD COLUMN needs_training BOOLEAN DEFAULT 1")
            await db.commit()
        except Exception:
            # La columna ya existe, no hacer nada
            pass
        
        await db.commit()


async def get_keyword_by_name(name: str) -> Optional[Dict]:
    """Obtiene un keyword por nombre."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM keywords WHERE name = ?", (name,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def create_keyword(name: str) -> Dict:
    """Crea un nuevo keyword."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO keywords (name) VALUES (?)", (name,)
        )
        await db.commit()
        
        return {
            "id": cursor.lastrowid,
            "name": name,
            "n_templates": 0
        }


async def get_all_keywords() -> List[Dict]:
    """Obtiene todos los keywords."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, name, n_templates, needs_training FROM keywords ORDER BY name"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def update_keyword_templates(keyword_id: int, n_templates: int):
    """Actualiza el número de templates de un keyword."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE keywords SET n_templates = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (n_templates, keyword_id)
        )
        await db.commit()


async def delete_keyword(keyword_id: int):
    """Elimina un keyword (CASCADE eliminará samples y thresholds)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM keywords WHERE id = ?", (keyword_id,))
        await db.commit()


async def add_sample(keyword_id: int, filename: str, duration: float, 
                    sample_rate: int, file_path: str) -> Dict:
    """Agrega un sample a la base de datos."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO samples (keyword_id, filename, duration, sample_rate, file_path)
               VALUES (?, ?, ?, ?, ?)""",
            (keyword_id, filename, duration, sample_rate, file_path)
        )
        await db.commit()
        
        return {
            "id": cursor.lastrowid,
            "keyword_id": keyword_id,
            "filename": filename,
            "duration": duration,
            "sample_rate": sample_rate,
            "file_path": file_path
        }


async def get_samples_by_keyword(keyword_id: int) -> List[Dict]:
    """Obtiene todos los samples de un keyword."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM samples WHERE keyword_id = ? ORDER BY created_at",
            (keyword_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def delete_sample(sample_id: int):
    """Elimina un sample."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Obtener datos antes de eliminar
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM samples WHERE id = ?", (sample_id,)
        ) as cursor:
            row = await cursor.fetchone()
            sample_data = dict(row) if row else None
        
        if sample_data:
            await db.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
            await db.commit()
        
        return sample_data


async def save_thresholds(keyword_id: int, thresholds: Dict):
    """Guarda o actualiza los thresholds de un keyword."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO thresholds 
               (keyword_id, threshold_min, threshold_max, mean_distance, 
                std_distance, min_distance, max_distance, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (
                keyword_id,
                thresholds.get("threshold_min"),
                thresholds.get("threshold_max"),
                thresholds.get("mean_distance"),
                thresholds.get("std_distance"),
                thresholds.get("min_distance"),
                thresholds.get("max_distance")
            )
        )
        await db.commit()


async def get_all_thresholds() -> Dict[str, Dict]:
    """Obtiene todos los thresholds."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT k.name, t.threshold_min, t.threshold_max, 
                      t.mean_distance, t.std_distance, t.min_distance, t.max_distance
               FROM thresholds t
               JOIN keywords k ON t.keyword_id = k.id"""
        ) as cursor:
            rows = await cursor.fetchall()
            
            result = {}
            for row in rows:
                name = row[0]
                result[name] = {
                    "threshold_min": row[1],
                    "threshold_max": row[2],
                    "mean_distance": row[3],
                    "std_distance": row[4],
                    "min_distance": row[5],
                    "max_distance": row[6]
                }
            
            return result


async def get_threshold_by_keyword(keyword_id: int) -> Optional[Dict]:
    """Obtiene el threshold de un keyword específico."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM thresholds WHERE keyword_id = ?", (keyword_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def mark_keyword_needs_training(keyword_id: int):
    """Marca un keyword como que necesita entrenamiento."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE keywords SET needs_training = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (keyword_id,)
        )
        await db.commit()


async def mark_keyword_trained(keyword_id: int):
    """Marca un keyword como entrenado."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE keywords SET needs_training = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (keyword_id,)
        )
        await db.commit()
