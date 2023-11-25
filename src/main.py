"""src.main"""
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from fastapi.security import APIKeyHeader
from typing import List
import torch

from .config import API_CONFIG, MODEL_CONFIG
from .predict import DoubleItStrategy
from .schema import InferenceInput, InferenceOutput, InferenceReponse

# x-api-key header
api_key_header = APIKeyHeader(name="X-API-Key")

ml_model = {}


# Upload model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Function to upload the model on startup"""
    model = DoubleItStrategy(MODEL_CONFIG["model_path"])
    ml_model["model"] = model
    yield
    ml_model.clear()


# initialize FastAPI
app = FastAPI(
    title="Tenpo Challenge",
    description="API para realizar predicciones con un modelo de PyTorch",
    version="0.0.1",
    lifespan=lifespan,
)


# verified api key
def get_api_key(api_key: str = Depends(api_key_header)):
    """Function to verify the API key"""
    if api_key == API_CONFIG["API_KEY"]:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.post("/predict")
async def predict(
    data: InferenceInput, api_key: str = Depends(get_api_key)
) -> InferenceReponse:
    """Endpoint to make predictions with the model
    input:
        data: InferenceInput
        api_key: str
    output:
        InferenceReponse
    """
    try:
        # get model from package
        result = ml_model["model"].main(data.data)
        inference = InferenceOutput(result=result)
        inference_response = InferenceReponse(error=False, results=inference)

        return inference_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
