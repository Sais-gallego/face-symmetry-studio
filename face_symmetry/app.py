"""Interactive Web Studio for Face Symmetry Analysis."""

import io
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from .generator import SymmetryStudio
from .cli import create_synthetic_portrait

app = Flask(__name__)
studio = SymmetryStudio()

def mat_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Symmetry Studio | Análisis de Simetría Facial</title>
    <style>
        :root {
            --bg: #09090b;
            --card: #18181b;
            --border: #27272a;
            --primary: #3b82f6;
            --accent: #10b981;
            --text: #f4f4f5;
            --muted: #a1a1aa;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem 1rem;
            min-height: 100vh;
        }
        .container { max-width: 1080px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 2rem; }
        h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; }
        p.subtitle { color: var(--muted); font-size: 1.05rem; }
        .controls {
            display: flex; gap: 1rem; justify-content: center; margin-bottom: 2rem;
            flex-wrap: wrap;
        }
        .btn {
            background: var(--primary); color: white; border: none; padding: 0.75rem 1.5rem;
            border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s;
        }
        .btn:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn-demo { background: var(--border); color: var(--text); }
        .btn-demo:hover { background: #3f3f46; }
        input[type="file"] { display: none; }
        .metrics-card {
            background: var(--card); border: 1px solid var(--border); border-radius: 12px;
            padding: 1.5rem; margin-bottom: 2rem; display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem;
            text-align: center;
        }
        .metric-val { font-size: 2rem; font-weight: 800; color: var(--accent); }
        .metric-lbl { color: var(--muted); font-size: 0.85rem; text-transform: uppercase; margin-top: 0.25rem; }
        .grid-panels {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem; margin-bottom: 2rem;
        }
        .panel {
            background: var(--card); border: 1px solid var(--border); border-radius: 12px;
            overflow: hidden; text-align: center;
        }
        .panel h3 { font-size: 0.95rem; padding: 0.75rem; background: rgba(255,255,255,0.03); border-bottom: 1px solid var(--border); }
        .panel img { width: 100%; height: auto; display: block; }
        .full-collage {
            background: var(--card); border: 1px solid var(--border); border-radius: 12px;
            overflow: hidden; text-align: center; padding: 1rem;
        }
        .full-collage img { max-width: 100%; height: auto; border-radius: 8px; }
        .loader { display: none; text-align: center; color: var(--primary); font-weight: 600; margin: 1rem 0; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛡️ Face Symmetry Studio</h1>
            <p class="subtitle">Análisis Biométrico de Simetría Sagital, Clonación Bilateral y Mapa de Asimetría</p>
        </header>

        <div class="controls">
            <label class="btn">
                📁 Subir Fotografía (JPG/PNG)
                <input type="file" id="fileInput" accept="image/*" onchange="uploadImage(this)">
            </label>
            <button class="btn btn-demo" onclick="loadDemo()">⚡ Cargar Rostro de Demostración</button>
        </div>

        <div id="loader" class="loader">Procesando geometría facial y alineación sagital...</div>

        <div id="resultsArea" style="display: none;">
            <div class="metrics-card">
                <div>
                    <div id="mOverall" class="metric-val">--%</div>
                    <div class="metric-lbl">Índice Global de Simetría</div>
                </div>
                <div>
                    <div id="mPixel" class="metric-val">--%</div>
                    <div class="metric-lbl">Similitud de Píxeles</div>
                </div>
                <div>
                    <div id="mOcular" class="metric-val">--%</div>
                    <div class="metric-lbl">Balance Ocular</div>
                </div>
                <div>
                    <div id="mPhi" class="metric-val">--%</div>
                    <div class="metric-lbl">Proporción Áurea (Phi)</div>
                </div>
            </div>

            <div class="grid-panels">
                <div class="panel">
                    <h3>1. Original (Eje Sagital)</h3>
                    <img id="imgOrig" alt="Original">
                </div>
                <div class="panel">
                    <h3>2. Clon Izquierda-Izquierda</h3>
                    <img id="imgLeft" alt="Left Symmetric">
                </div>
                <div class="panel">
                    <h3>3. Mapa Térmico de Asimetría</h3>
                    <img id="imgHeat" alt="Heatmap">
                </div>
                <div class="panel">
                    <h3>4. Clon Derecha-Derecha</h3>
                    <img id="imgRight" alt="Right Symmetric">
                </div>
            </div>

            <div class="full-collage">
                <h3 style="margin-bottom: 1rem; color: var(--muted);">📊 Reporte Diagnóstico Completo de 4 Paneles</h3>
                <img id="imgCollage" alt="Collage">
            </div>
        </div>
    </div>

    <script>
        function showResults(data) {
            document.getElementById('loader').style.display = 'none';
            document.getElementById('resultsArea').style.display = 'block';
            
            document.getElementById('mOverall').innerText = data.metrics.overall_symmetry_score + '%';
            document.getElementById('mPixel').innerText = data.metrics.pixel_similarity_score + '%';
            document.getElementById('mOcular').innerText = data.metrics.ocular_symmetry_score + '%';
            document.getElementById('mPhi').innerText = data.metrics.golden_ratio_score + '%';

            document.getElementById('imgOrig').src = 'data:image/png;base64,' + data.annotated_original;
            document.getElementById('imgLeft').src = 'data:image/png;base64,' + data.left_symmetric;
            document.getElementById('imgHeat').src = 'data:image/png;base64,' + data.heatmap;
            document.getElementById('imgRight').src = 'data:image/png;base64,' + data.right_symmetric;
            document.getElementById('imgCollage').src = 'data:image/png;base64,' + data.collage;
        }

        async function loadDemo() {
            document.getElementById('loader').style.display = 'block';
            const res = await fetch('/api/demo');
            const data = await res.json();
            showResults(data);
        }

        async function uploadImage(input) {
            if (!input.files || !input.files[0]) return;
            document.getElementById('loader').style.display = 'block';
            const formData = new FormData();
            formData.append('image', input.files[0]);

            const res = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            showResults(data);
        }

        // Auto-load demo on start
        window.addEventListener('DOMContentLoaded', loadDemo);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/demo")
def api_demo():
    synth = create_synthetic_portrait()
    results = studio.process_portrait(synth)
    return jsonify({
        "metrics": {
            "overall_symmetry_score": results["metrics"].overall_symmetry_score,
            "pixel_similarity_score": results["metrics"].pixel_similarity_score,
            "ocular_symmetry_score": results["metrics"].ocular_symmetry_score,
            "golden_ratio_score": results["metrics"].golden_ratio_score
        },
        "annotated_original": mat_to_base64(results["annotated_original"]),
        "left_symmetric": mat_to_base64(results["left_symmetric"]),
        "right_symmetric": mat_to_base64(results["right_symmetric"]),
        "heatmap": mat_to_base64(results["heatmap"]),
        "collage": mat_to_base64(results["collage"])
    })

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400
        
    results = studio.process_portrait(img)
    return jsonify({
        "metrics": {
            "overall_symmetry_score": results["metrics"].overall_symmetry_score,
            "pixel_similarity_score": results["metrics"].pixel_similarity_score,
            "ocular_symmetry_score": results["metrics"].ocular_symmetry_score,
            "golden_ratio_score": results["metrics"].golden_ratio_score
        },
        "annotated_original": mat_to_base64(results["annotated_original"]),
        "left_symmetric": mat_to_base64(results["left_symmetric"]),
        "right_symmetric": mat_to_base64(results["right_symmetric"]),
        "heatmap": mat_to_base64(results["heatmap"]),
        "collage": mat_to_base64(results["collage"])
    })

def run_app(host="127.0.0.1", port=5050):
    print(f"[*] Starting Face Symmetry Studio Web Dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
