import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
DATA_DIR = './data/CICIDS2017'
OUTPUT_FILE = './data/cicids2017_cleaned.csv'
files = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
print(" Scanning for common columns and labels...")
common_cols = None
all_labels = set()
for file in files:
    try:
        path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(path, low_memory=False, nrows=1000)
        all_labels.update(df['Label'].dropna().unique())
        cols = set(df.columns)
        common_cols = cols if common_cols is None else common_cols & cols
    except:
        continue
drop_cols = {'Flow ID', 'Source IP', 'Destination IP', 'Timestamp'}
usable_cols = list(common_cols - drop_cols)
usable_cols.append('Label')
print(f" Found {len(usable_cols)} common usable columns")
print(" Fitting LabelEncoder...")
label_encoder = LabelEncoder()
label_encoder.fit(list(all_labels))
print(" Label mapping:")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
first_file = True
for file in files:
    path = os.path.join(DATA_DIR, file)
    try:
        print(f"\n Processing {file}...")
        for chunk in pd.read_csv(path, usecols=usable_cols, chunksize=50000):
            chunk.dropna(inplace=True)
            chunk['Label'] = label_encoder.transform(chunk['Label'])
            chunk.to_csv(OUTPUT_FILE, mode='w' if first_file else 'a', header=first_file, index=False)
            first_file = False
            print(f" Processed chunk of size: {chunk.shape}")
    except Exception as e:
        print(f" Skipped {file}: {e}")
print(f"\n All files processed & saved to: {OUTPUT_FILE}")
