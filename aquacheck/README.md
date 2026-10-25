# AquaCheck — Klasifikasi Keamanan Air Minum (Logistic Regression)
Jalankan lokal:
    pip install -r requirements.txt
    python app.py          # buka http://127.0.0.1:5000
Latih ulang model (opsional):
    pip install -r requirements-train.txt && python train.py
Deploy (Render/Railway): Build `pip install -r requirements.txt`, Start `gunicorn app:app`.
