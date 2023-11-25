from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import torch

from .config import API_CONFIG, MODEL_CONFIG
from .predict import DoubleItStrategy
from .schema import InferenceInput, InferenceOutput, InferenceReponse

# initialize FastAPI
app = FastAPI(
    title="Tenpo Challenge",
    description="API para realizar predicciones con un modelo de PyTorch",
    version="0.0.1",
)


# x-api-key header
api_key_header = APIKeyHeader(name="X-API-Key")


# Upload model on startup
@app.on_event("startup")
async def startup_event():
    # Carga el modelo
    model = DoubleItStrategy(MODEL_CONFIG["model_path"])
    app.package = {"model": model, "api_key_header": api_key_header}


# verified api key
def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_CONFIG["API_KEY"]:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.post("/predict")
async def predict(
    data: InferenceInput, api_key: str = Depends(get_api_key)
) -> InferenceReponse:
    try:
        # get model from package
        result = app.package["model"].main(data.data)
        inference = InferenceOutput(result=result)
        inference_response = InferenceReponse(error=False, results=inference)

        return inference_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
