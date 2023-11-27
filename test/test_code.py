"""test.test_code"""
import torch
from src.predict import InfereceStrategy, DoubleItStrategy
import pytest


def test_inference_strategy_interface():
    # Test that the InferenceStrategy interface methods are abstract
    with pytest.raises(TypeError):
        strategy = InfereceStrategy()
        strategy.upload_model("model_path")
        strategy.preprocess("data")
        strategy.predict("data")
        strategy.postprocess("prediction")
        strategy.main("data")


def test_double_it_strategy_upload_model():
    # Test that the model is successfully loaded
    double_it_strategy = DoubleItStrategy("doubleit_model.pt")
    model = double_it_strategy.upload_model("doubleit_model.pt")
    assert isinstance(model, torch.jit.ScriptModule)


def test_double_it_strategy_preprocess():
    # Test that data is preprocessed correctly
    double_it_strategy = DoubleItStrategy("doubleit_model.pt")
    preprocessed_data = double_it_strategy.preprocess([1, 2, 3])
    assert isinstance(preprocessed_data, torch.Tensor)


def test_double_it_strategy_predict():
    # Test that prediction is obtained from the model
    double_it_strategy = DoubleItStrategy("doubleit_model.pt")
    prediction = double_it_strategy.predict(
        torch.tensor([1, 2, 3], dtype=torch.float32)
    )
    assert isinstance(prediction, torch.Tensor)


def test_double_it_strategy_postprocess():
    # Test that the prediction is postprocessed correctly
    double_it_strategy = DoubleItStrategy("doubleit_model.pt")
    prediction = torch.tensor([1, 2, 3], dtype=torch.float32)
    postprocessed_result = double_it_strategy.postprocess(prediction)
    assert isinstance(postprocessed_result, list)


def test_double_it_strategy_main():
    # Test the main method
    double_it_strategy = DoubleItStrategy("doubleit_model.pt")
    result = double_it_strategy.main([1, 2, 3])
    assert isinstance(result, list)
    assert result == [2, 4, 6]

# Importar las bibliotecas necesarias para las pruebas
import pytest
from pydantic import ValidationError
from src.schema import InferenceInput, InferenceOutput, InferenceReponse

# Pruebas para el esquema InferenceInput
def test_inference_input_valid():
    input_data = {"data": [1.0, 2.0, 3.0]}
    assert InferenceInput(**input_data)

def test_inference_input_missing_data():
    with pytest.raises(ValidationError):
        InferenceInput()

def test_inference_input_invalid_data_type():
    with pytest.raises(ValidationError):
        InferenceInput(data="invalid_data")

# Pruebas para el esquema InferenceOutput
def test_inference_output_valid():
    output_data = {"result": [2.0, 4.0, 6.0]}
    assert InferenceOutput(**output_data)

def test_inference_output_missing_result():
    with pytest.raises(ValidationError):
        InferenceOutput()

def test_inference_output_invalid_result_type():
    with pytest.raises(ValidationError):
        InferenceOutput(result="invalid_result")

# Pruebas para el esquema InferenceReponse
def test_inference_response_valid():
    response_data = {"error": False, "results": {"result": [2.0, 4.0, 6.0]}}
    assert InferenceReponse(**response_data)

def test_inference_response_missing_error():
    with pytest.raises(ValidationError):
        InferenceReponse(results={"result": [2.0, 4.0, 6.0]})

def test_inference_response_invalid_error_type():
    with pytest.raises(ValidationError):
        InferenceReponse(error="invalid_error", results={"result": [2.0, 4.0, 6.0]})

def test_inference_response_missing_results():
    with pytest.raises(ValidationError):
        InferenceReponse(error=False)

def test_inference_response_invalid_results_type():
    with pytest.raises(ValidationError):
        InferenceReponse(error=False, results="invalid_results")
