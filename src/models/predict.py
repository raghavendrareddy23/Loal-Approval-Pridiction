import joblib
import pandas as pd

def predict(input_dict):
    model = joblib.load("model/model.pkl")
    input_df = pd.DataFrame([input_dict])
    return int(model.predict(input_df)[0])