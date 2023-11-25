from abc import ABC, abstractmethod
import torch


# We are going to use a design pattern called strategy pattern (https://refactoring.guru/es/design-patterns/strategy)
# The idea is that each model that you want to use for prediction, should implement this interface
class InfereciaStrategy(ABC):
    @abstractmethod
    def upload_model(self, model_path):
        """This method should load the model from the path and store it in the class instance"""
        pass

    @abstractmethod
    def preprocess(self, data):
        """This method should preprocess the data before feeding it to the model"""
        pass

    @abstractmethod
    def predict(self, data):
        """This method should return the prediction of the model"""
        pass

    @abstractmethod
    def postprocess(self, prediction):
        """This method should postprocess the prediction before returning it"""
        pass

    @abstractmethod
    def main(self, data):
        """This method should call the other methods in the correct order"""
        pass


class DoubleItStrategy(InfereciaStrategy):
    """This class implements the inference strategy for the doubleit model"""

    def __init__(self, model_path):
        self.model = self.upload_model(model_path)

    def upload_model(self, model_path):
        return torch.jit.load(model_path)

    def preprocess(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def predict(self, data):
        return self.model(data)

    def postprocess(self, prediction):
        return prediction.tolist()

    def main(self, data):
        data = self.preprocess(data)
        prediction = self.predict(data)
        prediction = self.postprocess(prediction)
        return prediction
