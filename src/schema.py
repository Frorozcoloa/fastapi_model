from typing import List, Optional
from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """Clase para definir el esquema de entrada de la API"""

    data: List[float] = Field(..., example=[1.0, 2.0, 3.0])


class InferenceOutput(BaseModel):
    """Clase para definir el esquema de salida de la API"""

    result: List[float] = Field(..., example=[2.0, 4.0, 6.0])


class InferenceReponse(BaseModel):
    """Clase para definir el esquema de respuesta de la API"""

    error: bool = Field(..., example=False)
    results: Optional[InferenceOutput] = Field(..., example={"result": [2.0, 4.0, 6.0]})
