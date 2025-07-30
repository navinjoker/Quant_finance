from tensorflow.keras.models import load_model

def predict_latest(X_seq, model_path):
    model = load_model(model_path, compile=False)
    return model.predict(X_seq)