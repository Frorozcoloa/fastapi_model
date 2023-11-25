from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from typing import List
import torch

app = FastAPI(
    title="Tenpo Challenge",
    description="API para realizar predicciones con un modelo de PyTorch",
    version="0.0.1",
)



# Configura la seguridad para la API key
api_key_header = APIKeyHeader(name="X-API-Key")

# Carga el modelo
ts = torch.jit.load('./doubleit_model.pt')

# Función para verificar la clave API en cada solicitud
def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")


# Ruta protegida con autenticación de API key
@app.post("/predict")
async def predict(data: List[float], api_key: str = Depends(get_api_key)):
    try:
        # Convierte la lista de floats en un tensor de PyTorch
        input_tensor = torch.tensor(data, dtype=torch.float32)
        
        # Realiza la inferencia con el modelo
        result = ts(input_tensor)
        
        # Convierte el resultado a una lista de floats
        result_list = result.tolist()
        
        return {"result": result_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
