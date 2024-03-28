import cv2
import numpy as np


class ROIInitialiser:
  def __init__(self, roi_points: np.ndarray) -> None:
    self.roi_points = np.array(
        roi_points) if roi_points is not None else np.array([], dtype=np.int32)
    self.done: bool = False
    self.current: tuple = (0, 0)
    self.temp_points: list = []

  def _save_roi_points(self, save_path: str) -> None:
    """
    Save the ROI points to a numpy file.

    Parameters
    ----------
    save_path : str
      The path to save the ROI points to.
    """
    np.save(save_path, self.roi_points)

  def _load_roi_points(self, load_path: str) -> np.ndarray:
    """
    Load the ROI points from a numpy file.

    Parameters
    ----------
    load_path : str
      The path to load the ROI points from.
    """
    return np.load(load_path)

  def handle_mouse_events(self, event, x, y, flags, param) -> None:
    """
    Handle mouse events for selecting ROI points in a frame.

    Parameters
    ----------
    event : int
      The type of mouse event.
    x : int
      The x-coordinate of the mouse event.
    y : int
      The y-coordinate of the mouse event.
    flags : int
      The flags for the mouse event.
    param : dict
      A dictionary containing the ROI points and the current point.
    """
    if self.done:
      return

    if event == cv2.EVENT_MOUSEMOVE:
      # On mouse move, update the current point
      self.current = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
      # On left mouse button click, add the point to the ROI points
      self.temp_points.append((x, y))
    elif event == cv2.EVENT_MBUTTONDOWN:
      # On middle mouse button click, finish selecting the ROI points if > 3
      # points
      if len(self.temp_points) > 3:
        self.done = True
        self.roi_points = np.array(self.temp_points, dtype=np.int32)
        self._save_roi_points('data/roi_points.npy')
    elif event == cv2.EVENT_RBUTTONDOWN:
      # On right mouse button click, clear the ROI points
      self.temp_points = []

    if not self.done :
      # update the param with current list of points
      param['roi_points'] = self.temp_points
      param['current_point'] = self.current

  def _handle_roi_setup(self, frame, window_name, params) -> None:
    """
    Handle the setup of the region of interest.

    Parameters
    ----------
    frame : np.ndarray
      The frame to display.
    window_name : str
      The name of the window.
    params : dict
      A dictionary containing the ROI points and the current point.
    """
    while not self.done:
      if len(params['roi_points']) > 0:
        # If the user has selected points, display the ROI points and the
        # current point
        updated_frame = frame.copy()
        cv2.polylines(
            updated_frame, [
                np.array(
                    params['roi_points'])], False, (0, 255, 0), 2)
        cv2.line(updated_frame, params['roi_points']
                 [-1], params['current_point'], (0, 255, 0), 2)
        cv2.imshow(window_name, updated_frame)
      else:
        cv2.imshow(window_name, frame)
      if cv2.waitKey(10) & 0xFF == ord('q') or cv2.getWindowProperty(
              window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

  def set_roi(self, frame,
              window_name: str = 'Define Region of Interest') -> np.ndarray:
    """
    Set the region of interest in a frame.

    Parameters
    ----------
    frame : np.ndarray
      The frame to select the region of interest.
    window_name : str, optional
      The name of the window, by default 'Define Region of Interest'.

    Returns
    -------
    np.ndarray
      The region of interest points.
    """
    cv2.namedWindow(window_name)
    cv2.putText(frame, "Choose the area to monitor for intrusions. \
                        - Left click to place a point \
                        - Right click to clear points \
                        - Middle mouse button to finish.",
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2)

    if len(self.roi_points) > 0:
      # If ROI points are provided, display them on the frame
      cv2.polylines(frame, [np.array(self.roi_points)], False, (0, 255, 0), 2)

    params = {'roi_points': [], 'current_point': (0, 0)}
    cv2.setMouseCallback(window_name, self.handle_mouse_events, params)

    self._handle_roi_setup(frame, window_name, params)

    cv2.destroyWindow(window_name)
    # Save the ROI points to a file
    self._save_roi_points('data/roi_points.npy')

    return np.array(self.roi_points, dtype=int)
