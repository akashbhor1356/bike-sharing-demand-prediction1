# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model
from fastapi import Depends


# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("app")

# Create input/output pydantic models
input_model = create_model("app_input", **{'Hour': 0.0, 'Temperature': 3.200000047683716, 'Humidity': 84.0, 'Wind_speed': 0.6000000238418579, 'Visibility': 785.0, 'Dew_point_temperature': 0.699999988079071, 'Solar_Radiation': 0.0, 'Rainfall': 0.0, 'Snowfall': 0.0, 'Seasons': 1.0, 'Holiday': 0.0, 'Functioning_Day': 1.0})
output_model = create_model("app_output", Rented_Bike_Count_prediction=197)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model = Depends()):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"Rented_Bike_Count_prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
