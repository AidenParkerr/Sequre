import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime

def setup_logging():
  log_directory = "logs"
  if not os.path.exists(log_directory):
    os.makedirs(log_directory)
  
  datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_file_name = f"{log_directory}/{datetime_str}.log"
  
  logger = logging.getLogger('ObjectDetectionLogger')
  logger.setLevel(logging.DEBUG)  # Adjust as needed

  if not logger.handlers:  # Prevent adding duplicate handlers if already exists
    file_handler = TimedRotatingFileHandler(log_file_name, when="midnight", interval=1, backupCount=7)
    file_handler.setLevel(logging.DEBUG)  # Adjust as per requirement
    
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    
    logger.addHandler(file_handler)
