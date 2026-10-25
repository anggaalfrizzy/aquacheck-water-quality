"""Latih model Logistic Regression untuk klasifikasi kelayakan air minum."""
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE

# nama tampilan, batas bahaya (berbahaya jika melebihi) sesuai deskripsi dataset
META = {"aluminium":["Aluminium",2.8],"ammonia":["Amonia",32.5],"arsenic":["Arsenik",0.01],
 "barium":["Barium",2],"cadmium":["Kadmium",0.005],"chloramine":["Kloramin",4],
 "chromium":["Kromium",0.1],"copper":["Tembaga",1.3],"flouride":["Fluorida",1.5],
 "bacteria":["Bakteri",0],"viruses":["Virus",0],"lead":["Timbal",0.015],
 "nitrates":["Nitrat",10],"nitrites":["Nitrit",1],"mercury":["Merkuri",0.002],
 "perchlorate":["Perklorat",56],"radium":["Radium",5],"selenium":["Selenium",0.5],
 "silver":["Perak",0.1],"uranium":["Uranium",0.3]}
FEATURES = list(META)
df = pd.read_csv("data/waterQuality1.csv")
df = df.apply(pd.to_numeric, errors="coerce")       # '#NUM!' -> NaN
df = df.dropna(subset=["is_safe"]).reset_index(drop=True)   # buang baris tanpa label
X, y = df[FEATURES], df["is_safe"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

imp = SimpleImputer(strategy="median").fit(X_tr)           # fit hanya di data latih
sc = StandardScaler().fit(imp.transform(X_tr))
Xtr = sc.transform(imp.transform(X_tr)); Xte = sc.transform(imp.transform(X_te))
Xtr_bal, ytr_bal = SMOTE(random_state=42).fit_resample(Xtr, y_tr)  # SMOTE hanya di data latih

model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42).fit(Xtr_bal, ytr_bal)
proba = model.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)
cm = confusion_matrix(y_te, pred)
fpr, tpr, _ = roc_curve(y_te, proba)
step = max(1, len(fpr)//80)

metrics = {
  "n_total": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
  "class_counts": {"0": int((y==0).sum()), "1": int((y==1).sum())},
  "missing": {c:int(df[c].isna().sum()) for c in FEATURES if df[c].isna().any()},
  "cm": cm.tolist(), "meta": META, "features": FEATURES,
  "accuracy": accuracy_score(y_te,pred), "precision": precision_score(y_te,pred),
  "recall": recall_score(y_te,pred), "f1": f1_score(y_te,pred),
  "auc": roc_auc_score(y_te,proba),
  "roc": {"fpr": fpr[::step].round(4).tolist(), "tpr": tpr[::step].round(4).tolist()},
  "coef": {f: float(c) for f,c in zip(FEATURES, model.coef_[0])},
  "intercept": float(model.intercept_[0]),
  "ranges": {f: [float(df[f].min()), float(df[f].median()), float(df[f].max())] for f in FEATURES},
}
joblib.dump({"imputer":imp,"scaler":sc,"model":model,"features":FEATURES}, "model/water_lr.joblib")
json.dump(metrics, open("model/metrics.json","w"), indent=1)
print(json.dumps({k:metrics[k] for k in ["cm","accuracy","precision","recall","f1","auc"]}, indent=1))
