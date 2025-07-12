import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
df = pd.read_csv('./data/cicids2017_cleaned.csv')
print(f" Dataset loaded: {df.shape}")
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f" Training on: {X_train.shape}, Testing on: {X_test.shape}")
print(" Training XGBoost classifier...")
model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(y.unique()),
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=0
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))
print(f" Accuracy: {accuracy_score(y_test, y_pred):.4f}")
joblib.dump(model, './data/traffic_classifier.pkl')
print("\n Model saved to: ./data/traffic_classifier.pkl")
