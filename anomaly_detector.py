import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
df = pd.read_csv('./data/cicids2017_cleaned.csv')
df_benign = df[df['Label'] == 0].drop(columns=['Label'])
if df_benign.empty:
    print(" No benign samples found (label == 0).")
else:
    print(f" Found {len(df_benign)} benign samples.")
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df_benign)
    joblib.dump(model, './data/anomaly_detector.pkl')
    print(" Anomaly detector trained and saved.")
