import dill

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import Optional


app = FastAPI()
with open('models/best_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_keyword: Optional[str] = None
    utm_adcontent: Optional[str] = None
    device_category: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    device_screen_resolution: Optional[str] = None
    device_browser: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None


class Prediction(BaseModel):
    session_id: str
    result: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'result': y[0]
    }


