import joblib


def predict(data):
    
    if joblib is None:
        raise RuntimeError(
            "Missing dependency 'joblib'. Make sure you have installed project dependencies (requirements.txt) before running the app."
        )

    # Load the trained model and predict
    clf = joblib.load("loan_model.sav")
    return clf.predict(data)