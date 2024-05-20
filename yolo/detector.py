from ultralytics import YOLO
import torch
import logging

class YOLODetector:
  def __init__(self, model_config) -> None:
    self.logger = logging.getLogger('ObjectDetectionLogger')
    self.model_config = model_config
    self.model = self._init_model(self.model_config)

    
    
    
  def _init_model(self, model_config: dict) -> YOLO:
    """
    Initialise the YOLO model.

    This function initialises the YOLO model by extracting the model path from the configuration file.

    Args:
        model_config (dict[str, str]): The model configuration dictionary.
    """
    self.model_path: str = model_config['model_path']
    self.logger.info(f"Model path: {self.model_path}")
    model = YOLO(self.model_path)
    model.fuse() 
    self.logger.info(f"Model loaded. Cuda available: {torch.cuda.is_available()}")
    return model


  def train(self):
    """ Should handle training of the model using the provided data specified in the configuration file. """
    raise NotImplementedError
  

  def predict(self, frame) -> list:
    if not self.model:
      self.logger.error("Model not loaded.")
      return []
    
    results = self.model(frame)
    return results
