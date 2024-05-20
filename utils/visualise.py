import cv2
import numpy as np


def display(frame: np.ndarray, window_name: str = 'frame',
            delay: int = 0) -> bool:
  """
  Display the frame in a window.

  Display the frame in a window with the name `window_name`. The function waits for a key press
  for 10 milliseconds and checks if the window is closed. If the window is closed or the key 'q' is
  pressed, the function returns False, otherwise it returns True.

  Parameters
  ----------
  frame : np.ndarray
    The frame to display.
  window_name : str, optional
    The name of the window, by default 'frame'.
  delay : int, optional
    The number of milliseconds to wait for a key press, by default 0.

  Returns
  -------
  bool
    True if the window is open, False otherwise.
  """
  cv2.imshow(window_name, frame)
  if cv2.waitKey(delay) & 0xFF == ord('q') or cv2.getWindowProperty(
          window_name, cv2.WND_PROP_VISIBLE) < 1:
    return False
  return True


def draw_grid(frame: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
              box_coords: np.ndarray, num_segments: int = 10) -> None:
  """
  Draw the grid on the frame.

  The x and y coordinates of the grid passed as `grid_x` and `grid_y` are used to draw the grid
  on the frame. The bounding box coordinates are used to draw the grid inside the bounding box
  container.


  Parameters
  ----------
  frame : np.ndarray
    The frame to draw the grid on.
  grid_x : np.ndarray
    The grid x coordinates.
  grid_y : np.ndarray
    The grid y coordinates.
  box_coords : np.ndarray
    The bounding box coordinates.

  """
  left, top, right, bottom = np.array(box_coords, dtype=np.int32).squeeze()
  for i in range(num_segments):
    cv2.line(
        frame, (int(
            grid_x[0][i]), top), (int(
                grid_x[0][i]), bottom), (0, 0, 255), 1)
    cv2.line(
        frame, (left, int(
            grid_y[i][0])), (right, int(
                grid_y[i][0])), (0, 0, 255), 1)


def retrieve_cropped_box(frame: np.ndarray, box_coords: np.ndarray, box_class: str,
                         desired_class: str = 'person') -> np.ndarray:
  """
  Retrieve the cropped box from the frame.

  Retrieve the cropped box from the frame using the bounding box coordinates. The bounding box
  coordinates are used to crop the frame and return the cropped box. The function checks if the
  class of the object is the desired class, if not, it returns an empty array.

  Parameters
  ----------
  frame : np.ndarray
    The frame to crop the bounding box from.
  box_coords : np.ndarray
    The bounding box coordinates.
  box_class : str
    The class of the bounding box.
  desired_class : str, optional
    The desired class of the bounding box, by default 'person'.

  Returns
  -------
  np.ndarray
    The cropped box from the frame.
  """
  if box_class != desired_class:
    return np.array([], dtype=np.int32)
  left, top, right, bottom = np.array(box_coords, dtype=int).squeeze()
  cropped_frame = frame[top:bottom, left:right]
  return cropped_frame
