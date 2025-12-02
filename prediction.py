import joblib


def predict(data):
    """Load model and run prediction.

    Raises a helpful RuntimeError if joblib is not available so the
    user gets a clear build-time/runtime instruction instead of a
    cryptic ImportError on app start.
    """
    if joblib is None:
        raise RuntimeError(
            "Missing dependency 'joblib'. Make sure you have installed project dependencies (requirements.txt) before running the app."
        )

    # Load the trained model and predict
    clf = joblib.load("loan_model.sav")
    return clf.predict(data)