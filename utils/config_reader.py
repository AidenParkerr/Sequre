import json

class ConfigReader:
  def __init__(self, config_path: str) -> None:
    self.config_path = config_path
    self.config: dict[str, dict] = self._read_config()
    
  def _read_config(self) -> dict[str, dict]:
    try:
      with open(self.config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    except FileNotFoundError:
      raise FileNotFoundError(f"Config file not found at {self.config_path}")
    except json.JSONDecodeError:
      raise json.JSONDecodeError(f"Error reading config file at {self.config_path}")
    except Exception as e:
      raise Exception(f"Error reading config file at {self.config_path}: {e}")
    
  def get(self, key: str) -> dict:
    return self.config[key]