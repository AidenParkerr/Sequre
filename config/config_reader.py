import json


class ConfigReader:
  """
  The ConfigReader class reads the configuration file and extracts the required information from it. The
  configuration file is a JSON file containing data ranging from the paths to the video, model,
  and region of interest points numpy file.

  Attributes
  ----------
  config_path : str
    The path to the configuration file.
  config : dict[str, dict]
    The contents of the configuration file.

  Methods
  -------
  _read_config() -> dict[str, dict]
    Read the configuration file and return the contents as a dictionary.
  get(key: str) -> dict
    Return the value of the key from the configuration file.
  """

  def __init__(self, config_path: str) -> None:
    self.config_path = config_path
    self.config: dict[str, dict] = self._read_config()

  def _read_config(self) -> dict[str, dict]:
    """
    Read the configuration file and return the contents as a dictionary.

    Returns
    -------
    dict[str, dict]
      The contents of the configuration file.

    Raises
    ------
    FileNotFoundError
      If the configuration file is not found.
    json.JSONDecodeError
      If there is an error reading the configuration file.
    Exception
      If there is any other error reading the configuration file.
    """
    try:
      with open(self.config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    except FileNotFoundError:
      raise FileNotFoundError(f"Config file not found at {self.config_path}")
    except json.JSONDecodeError as e:
      raise json.JSONDecodeError(
          f"Error reading config file at {self.config_path}",
          doc=e.doc,
          pos=e.pos)
    except Exception as e:
      raise Exception(f"Error reading config file at {self.config_path}: {e}")

  def get(self, key: str) -> dict:
    """
    Return the value of the key from the configuration file.

    The base structure of the configuration file is a dictionary containing keys for each
    configuration section, such as `data_config`, or `video_config`. The value of each key
    is another dictionary containing the configuration values for that section.

    Parameters
    ----------
    key : str
      The key to extract from the configuration file.

    Returns
    -------
    dict
      The value of the key from the configuration file.
    """
    return self.config[key]
  
  def get_config(self) -> dict:
    """
    Return the entire configuration from the configuration file.

    Returns
    -------
    dict
      The entire configuration from the configuration file.
    """
    return self.config
  
  def get_model_config(self) -> dict:
    """
    Return the model configuration from the configuration file.

    The model configuration is a dictionary containing the model path, model name, and model type.

    Parameters
    ----------
    model_name : str
      The name of the model configuration to extract from the configuration file.

    Returns
    -------
    dict
      The model configuration from the configuration file.
    """
    return self.config['model_config']


  def get_video_config(self) -> dict:
    """
    Return the video configuration from the configuration file.

    The video configuration is a dictionary containing the video path and the region of interest points path.

    Returns
    -------
    dict
      The video configuration from the configuration file.
    """
    return self.config['video_config']
  
  def get_data_config(self) -> dict:
    """
    Return the data configuration from the configuration file.

    The data configuration is a dictionary containing the data path and the data type.

    Returns
    -------
    dict
      The data configuration from the configuration file.
    """
    return self.config['data_config']