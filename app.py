"""
FastAPI Safety Uniform & Vest Compliance API
"""

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from core.forma_engine import FormaComplianceEngine

app = FastAPI(title="Safety Vest & Uniform Vision AI")
engine = FormaComplianceEngine()

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Safety Vest & Uniform Vision AI</title>
        <meta charset="utf-8">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }
            .card { background: #1e293b; padding: 2rem; border-radius: 12px; max-width: 800px; margin: 0 auto; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
            h1 { color: #10b981; margin-top: 0; }
            .badge { display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; margin-right: 8px; background: #059669; color: white; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🦺 PPE Safety Vest & Uniform Compliance AI</h1>
            <p>Dual-Model Fusion: Dedicated PersonGate Detector + YOLO Forma/Reflective Stripe Classifier.</p>
            <div>
                <span class="badge">PersonGate Model</span>
                <span class="badge">Reflective Stripe Containment</span>
                <span class="badge">Real-Time Violation Alerts</span>
            </div>
            <hr style="border: 0; border-top: 1px solid #334155; margin: 1.5rem 0;">
            <h3>API Endpoints:</h3>
            <ul>
                <li><code>POST /audit</code> - Upload image frame and receive worker compliance audit</li>
                <li><code>GET /health</code> - Check engine status</li>
                <li><code>GET /docs</code> - Swagger UI</li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.post("/audit")
async def audit_frame(file: UploadFile = File(...)):
    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    res = engine.predict_compliance(frame)
    return res

@app.get("/health")
def health():
    return {"status": "ok", "device": str(engine.device)}
