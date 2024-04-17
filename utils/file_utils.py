
import numpy as np
import logging
logger = logging.getLogger('ObjectDetectionLogger')

def load_roi_points(roi_points_path: str) -> np.ndarray:
  """
  Load the region of interest points from the file.

  This function loads the region of interest points from the file
  and returns the points as a numpy array.

  Parameters
  ----------
  roi_points_path : str
    The path to the region of interest points file.

  Returns
  -------
  np.ndarray
    The region of interest points.
  """
  try:
    roi_points = np.load(roi_points_path)
    logger.info(
        f"Region of interest points loaded successfully. Num points: {len(roi_points)}")
    return roi_points
  except OSError as e:
    logger.warning(f"Input file does not exist or cannot be read: {e}")
    return np.array([])
  except EOFError as e:
    logger.warning(
        f"Calling np.load multiple times on the same file handle: {e}")
    return np.array([])
  except Exception as e:
    logger.warning(f"Error loading the region of interest points: {e}")
    return np.array([])