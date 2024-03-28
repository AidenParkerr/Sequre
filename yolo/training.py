

class YOLOTrainer:
  def __init__(self, model, config):
    self.model = model
    self.config = config

  def train(self):
    raise NotImplementedError("Method not implemented")
