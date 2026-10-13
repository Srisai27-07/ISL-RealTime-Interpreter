import os
import sys
import json
import time
import pickle
import sqlite3
from datetime import datetime
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.append(".")

app = FastAPI(title="Vaani-ISL Real-Time API")

# 1. CORS Middleware for Frontend (React, Vite, Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Database Initialization
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

# 3. Model State Management
MODEL_STATE = {
    "model": None,
    "classes": [],
    "is_loaded": False
}

def load_model():
    path = "models/checkpoints/isl_static.pkl"
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            MODEL_STATE["model"] = data["model"]
            MODEL_STATE["classes"] = data["classes"]
            MODEL_STATE["is_loaded"] = True
            print("[INFO] Model weights loaded successfully!")
        except Exception as e:
            print(f"[WARN] Error loading model: {e}")

@app.on_event("startup")
async def startup():
    load_model()

# 4. Pydantic Schemas
class LandmarkPayload(BaseModel):
    landmarks: List[float]

class HistoryItem(BaseModel):
    sentence: str
    mode: str = "dynamic"
    confidence: float = 0.95

# 5. Core Endpoints
@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model_loaded": MODEL_STATE["is_loaded"],
        "classes_count": len(MODEL_STATE["classes"])
    }

@app.get("/api/vocabulary")
def vocabulary():
    if MODEL_STATE["is_loaded"]:
        return {"classes": MODEL_STATE["classes"]}
    return {"classes": ["A", "B", "C", "Hello", "Thank You", "Water", "Help"]}

@app.post("/api/predict")
def predict(payload: LandmarkPayload):
    if not MODEL_STATE["is_loaded"]:
        return {"sign": "A", "confidence": 0.95, "timestamp": time.time()}

    arr = np.array(payload.landmarks, dtype=np.float32).reshape(1, -1)
    probas = MODEL_STATE["model"].predict_proba(arr)[0]
    max_idx = int(np.argmax(probas))
    return {
        "sign": str(MODEL_STATE["classes"][max_idx]),
        "confidence": float(probas[max_idx]),
        "timestamp": time.time()
    }

# 6. Conversation History Endpoints
@app.get("/api/history")
def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM history ORDER BY id DESC LIMIT 50")
        rows = [dict(r) for r in cursor.fetchall()]
    return {"history": rows}

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

@app.delete("/api/history")
def clear_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM history")
        conn.commit()
    return {"status": "cleared"}

# 7. WebSocket Streaming Gateway
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Frontend client connected!")
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            landmarks = payload.get("landmarks", [])

            if MODEL_STATE["is_loaded"] and landmarks:
                arr = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                probas = MODEL_STATE["model"].predict_proba(arr)[0]
                max_idx = int(np.argmax(probas))
                sign = str(MODEL_STATE["classes"][max_idx])
                conf = float(probas[max_idx])
            else:
                sign = "A"
                conf = 0.90

            await websocket.send_text(json.dumps({
                "sign": sign,
                "confidence": conf,
                "timestamp": time.time()
            }))
    except WebSocketDisconnect:
        print("[WS] Frontend client disconnected.")