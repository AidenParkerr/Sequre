import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime


def setup_logging():
  log_directory = "logs"
  if not os.path.exists(log_directory):
    os.makedirs(log_directory)

  # Generate log file name with the current datetime
  datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_file_name = f"{log_directory}/{datetime_str}.log"

  # Create a custom logger
  logger = logging.getLogger('ObjectDetectionLogger')
  logger.setLevel(logging.DEBUG)  # Adjust this as needed

  # Create handlers
  file_handler = TimedRotatingFileHandler(
      log_file_name, when="midnight", interval=1, backupCount=7)
  file_handler.setLevel(logging.DEBUG)  # Adjust as per requirement

  # Create formatters and add it to handlers
  log_format = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(log_format)

  # Add handlers to the logger
  logger.addHandler(file_handler)

  return logger
