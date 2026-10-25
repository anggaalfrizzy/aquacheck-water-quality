import json, os, joblib, numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
art = joblib.load(os.path.join(BASE, "model", "water_lr.joblib"))
METRICS = json.load(open(os.path.join(BASE, "model", "metrics.json")))
FEATURES = art["features"]

@app.route("/")
def index():
    return render_template("index.html", m=METRICS, features=FEATURES)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        row = [float(data[f]) for f in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Semua 20 parameter harus diisi dengan angka."), 400
    if any(v < 0 for v in row):
        return jsonify(error="Konsentrasi tidak boleh bernilai negatif."), 400
    x = art["scaler"].transform(art["imputer"].transform(np.array([row])))
    p = float(art["model"].predict_proba(x)[0, 1])
    contrib = {f: float(c * v) for f, c, v in zip(FEATURES, art["model"].coef_[0], x[0])}
    return jsonify(probability=p, label="Aman Diminum" if p >= 0.5 else "Tidak Aman Diminum",
                   potable=p >= 0.5, contributions=contrib)

if __name__ == "__main__":
    app.run(debug=True)
