import torch
from src.predict import InfereciaStrategy, DoubleItStrategy
import pytest


def test_inference_strategy_interface():
    # Test that the InferenceStrategy interface methods are abstract
    with pytest.raises(TypeError):
        strategy = InfereciaStrategy()
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
