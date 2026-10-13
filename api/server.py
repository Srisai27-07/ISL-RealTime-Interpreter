import sqlite3
from datetime import datetime

DB_PATH = "data/history.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                sentence TEXT,
                mode TEXT,
                confidence REAL
            )
        """)
init_db()

class HistoryItem(BaseModel):
    sentence: str
    mode: str = "dynamic"  # "static" (letters) or "dynamic" (phrases)
    confidence: float = 0.95

# 1. Get all past conversation transcripts
@app.get("/api/history")
def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM history ORDER BY id DESC LIMIT 50")
        rows = [dict(r) for r in cursor.fetchall()]
    return {"history": rows}

# 2. Save a new conversation sentence
@app.post("/api/history")
def save_history(item: HistoryItem):
    now = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO history (timestamp, sentence, mode, confidence) VALUES (?, ?, ?, ?)",
            (now, item.sentence, item.mode, item.confidence)
        )
        conn.commit()
        new_id = cursor.lastrowid
    return {"status": "saved", "id": new_id, "timestamp": now}

# 3. Clear all history
@app.delete("/api/history")
def clear_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM history")
        conn.commit()
    return {"status": "cleared"}