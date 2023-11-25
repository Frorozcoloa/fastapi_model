import unittest
import torch
from src.predict import InfereciaStrategy, DoubleItStrategy

class TestInferenceStrategy(unittest.TestCase):
    def test_inference_strategy_interface(self):
        # Test that the InferenceStrategy interface methods are abstract
        with self.assertRaises(TypeError):
            strategy = InfereciaStrategy()
            strategy.upload_model('model_path')
            strategy.preprocess('data')
            strategy.predict('data')
            strategy.postprocess('prediction')
            strategy.main('data')

class TestDoubleItStrategy(unittest.TestCase):
    def setUp(self):
        # Set up a DoubleItStrategy instance for testing
        self.model_path = 'doubleit_model.pt'
        self.test_data = [1, 2, 3]
        self.double_it_strategy = DoubleItStrategy(self.model_path)

    def test_upload_model(self):
        # Test that the model is successfully loaded
        model = self.double_it_strategy.upload_model(self.model_path)
        self.assertIsInstance(model, torch.jit.ScriptModule)

    def test_preprocess(self):
        # Test that data is preprocessed correctly
        preprocessed_data = self.double_it_strategy.preprocess(self.test_data)
        self.assertIsInstance(preprocessed_data, torch.Tensor)

    def test_predict(self):
        # Test that prediction is obtained from the model
        prediction = self.double_it_strategy.predict(torch.tensor(self.test_data, dtype=torch.float32))
        self.assertIsInstance(prediction, torch.Tensor)
        

    def test_postprocess(self):
        # Test that the prediction is postprocessed correctly
        prediction = torch.tensor(self.test_data, dtype=torch.float32)
        postprocessed_result = self.double_it_strategy.postprocess(prediction)
        self.assertIsInstance(postprocessed_result, list)

    def test_main(self):
        # Test the main method
        result = self.double_it_strategy.main(self.test_data)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [2, 4, 6])

