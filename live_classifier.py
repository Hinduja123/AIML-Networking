import pyshark
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
model = joblib.load('./data/traffic_classifier.pkl')
anomaly_model = joblib.load('./data/anomaly_detector.pkl')
print(" Loaded models!")
df_train = pd.read_csv('./data/cicids2017_cleaned.csv', nrows=10000)
feature_cols = df_train.drop(columns=['Label']).columns.tolist()
label_encoder = LabelEncoder()
label_encoder.fit(df_train['Label'])
label_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
def extract_features(packet):
    try:
        return {
            'Flow Duration': float(packet.frame_info.time_delta),
            'Total Length of Fwd Packets': float(packet.length),
            'Fwd Packet Length Max': float(packet.length),
            'Fwd Packet Length Mean': float(packet.length),
            'Bwd Packet Length Max': float(packet.length),
            'Fwd IAT Total': float(packet.frame_info.time_delta),
            'Flow IAT Mean': float(packet.frame_info.time_delta),
            'Flow IAT Std': 0.0,
            'Flow IAT Max': float(packet.frame_info.time_delta),
            'Flow IAT Min': float(packet.frame_info.time_delta),
        }
    except Exception as e:
        print(" Skipped packet:", e)
        return None
pcap_file = 'capture.pcap'
cap = pyshark.FileCapture(pcap_file)
print("\n Analyzing packets from capture.pcap...\n")
for packet in cap:
    features = extract_features(packet)
    if features:
        df_features = pd.DataFrame([features], columns=feature_cols).fillna(0)
        pred_class = model.predict(df_features)[0]
        class_name = label_map.get(pred_class, "Unknown")
        is_anomaly = anomaly_model.predict(df_features)[0]
        anomaly_status = " YES" if is_anomaly == -1 else " NO"
        print(f" Predicted: {class_name} | Anomaly: {anomaly_status}")
