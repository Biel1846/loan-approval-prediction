import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Set seed agar hasil konsisten
seed = 42

# 1. Load Dataset
# Pastikan file csv ada di folder yang sama
df = pd.read_csv("data/loan_approval_dataset.csv")

# 2. Data Cleaning (PENTING: Dataset ini punya spasi tambahan di nama kolom & isinya)
df.columns = df.columns.str.strip() # Hapus spasi di nama kolom

# Hapus kolom ID
df = df.drop(columns=['loan_id'])

# Bersihkan spasi di data kategori
categorical_cols = ['education', 'self_employed', 'loan_status']
for col in categorical_cols:
    df[col] = df[col].str.strip()

# 3. Encoding (Ubah Teks jadi Angka)
# Education: Graduate -> 1, Not Graduate -> 0
df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})

# Self Employed: Yes -> 1, No -> 0
df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})

# Target (Loan Status): Kita biarkan tetap teks (Approved/Rejected) biar outputnya langsung terbaca
# Pisahkan Fitur (X) dan Target (y)
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# 5. Train Model
clf = RandomForestClassifier(n_estimators=100, random_state=seed)
clf.fit(X_train, y_train)

# 6. Cek Akurasi sebentar
y_pred = clf.predict(X_test)
print(f"Akurasi Model: {accuracy_score(y_test, y_pred)}")

# 7. Simpan Model
joblib.dump(clf, "loan_model.sav")
print("Sukses! Model disimpan sebagai loan_model.sav")