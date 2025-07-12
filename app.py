import streamlit as st
import pandas as pd
import pyshark
import joblib
from sklearn.preprocessing import LabelEncoder
import tempfile
import os
st.set_page_config(page_title="AI/ML Network Analyzer", layout="wide")
st.title(" AI/ML-based Network Traffic Analyzer with Anomaly Detection")
@st.cache_resource
def load_models():
    clf_model = joblib.load('./data/traffic_classifier.pkl')
    anomaly_model = joblib.load('./data/anomaly_detector.pkl')
    return clf_model, anomaly_model
@st.cache_resource
def load_label_map():
    df = pd.read_csv('./data/cicids2017_cleaned.csv', usecols=['Label'], nrows=10000)
    encoder = LabelEncoder()
    encoder.fit(df['Label'])
    return dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
clf_model, anomaly_model = load_models()
label_map = load_label_map()
df_train = pd.read_csv('./data/cicids2017_cleaned.csv', nrows=1)
feature_cols = df_train.drop(columns=['Label']).columns.tolist()
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
    except:
        return None
pcap_file = st.file_uploader(" Upload a .pcap file", type=["pcap"])
if pcap_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(pcap_file.read())
        tmp_path = tmp.name
    st.success(" File uploaded successfully.")
    st.info(" Processing packets...")
    cap = pyshark.FileCapture(tmp_path)
    results = []
    for packet in cap:
        features = extract_features(packet)
        if features:
            df_feat = pd.DataFrame([features], columns=feature_cols).fillna(0)
            pred_class = int(clf_model.predict(df_feat)[0])
            class_name = label_map.get(pred_class, "Unknown")
            anomaly = anomaly_model.predict(df_feat)[0]
            anomaly_status = " YES" if anomaly == -1 else " NO"
            results.append({
                "Predicted Class": class_name,
                "Anomaly": anomaly_status
            })
    cap.close()
    os.remove(tmp_path)
    if results:
        df_result = pd.DataFrame(results)
        st.subheader(" Prediction Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("** Traffic Type Count**")
            class_counts = df_result["Predicted Class"].value_counts().reset_index().rename(columns={"index": "Class", "Predicted Class": "Count"})
            st.dataframe(class_counts)
        with col2:
            st.markdown("** Anomaly Count**")
            anomaly_counts = df_result["Anomaly"].value_counts().reset_index().rename(columns={"index": "Anomaly", "Anomaly": "Count"})
            st.dataframe(anomaly_counts)
        st.subheader(" Full Packet Analysis")
        st.dataframe(df_result)
        st.download_button(
            label=" Download Results as CSV",
            data=df_result.to_csv(index=False),
            file_name="packet_analysis.csv",
            mime="text/csv"
        )
    else:
        st.warning("No usable packets were found.")
