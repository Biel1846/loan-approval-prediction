import joblib

def predict(data):
    # Memuat model yang sudah dilatih
    clf = joblib.load("loan_model.sav")
    return clf.predict(data)