import pandas as pd
from ms import model


def predict(X, model):
    return model.predict(X)[0]


def get_model_response(input):
    X = pd.json_normalize(input.__dict__)
    prediction = predict(X, model)
    label = "M" if prediction == 1 else "B"
    return {
        'label': label,
        'prediction': int(prediction)
    }
