from email.mime import image
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from deskew import determine_skew

def crop_largest_rect(img, angle_deg, orig_w, orig_h):
    """Calculates the maximum interior rectangle inside a rotated image to remove black corners."""
    angle_rad = np.abs(np.radians(angle_deg))
    
    # Calculate dimensions of largest inner rectangle
    if orig_w <= orig_h:
        side_short, side_long = orig_w, orig_h
    else:
        side_short, side_long = orig_h, orig_w

    sin = np.sin(angle_rad)
    cos = np.cos(angle_rad)
    
    # Determine maximum inside bounding box width and height
    if side_short <= 2 * sin * cos * side_long:
        x = 0.5 * side_short
        crop_w = int(x / sin) if sin != 0 else orig_w
        crop_h = int(x / cos) if cos != 0 else orig_h
    else:
        cos2 = cos * cos - sin * sin
        crop_w = int((orig_w * cos - orig_h * sin) / cos2)
        crop_h = int((orig_h * cos - orig_w * sin) / cos2)

    # Clamp to sane, positive dimensions so downstream cv2 calls (e.g. VideoWriter)
    # never see a zero/negative frame size
    crop_w = max(1, min(int(crop_w), orig_w))
    crop_h = max(1, min(int(crop_h), orig_h))

    # Perform center crop
    h, w = img.shape[:2]
    start_x = max(0, (w - crop_w) // 2)
    start_y = max(0, (h - crop_h) // 2)
    
    return img[start_y:start_y + crop_h, start_x:start_x + crop_w]

class VisionModel2():
    def crop_based_angle(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Determine angle using Radon transform
        angle = determine_skew(gray)
        if angle is None:
            # deskew couldn't confidently find an angle on this frame
            # (flat lighting, low texture, etc.) - fall back to "no rotation"
            angle = 0.0
        # Rotate image using OpenCV
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        straightened = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cropped = crop_largest_rect(straightened, angle, w, h)
        return cropped, angle
    def crop_post_angle(self, image,angle):
        # Rotate image using OpenCV
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        straightened = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cropped = crop_largest_rect(straightened, angle, w, h)
        return cropped

    def detect_court_line_segments(self,bgr_img: np.ndarray):
      # 1. Smooth slightly to suppress surface grain while keeping crisp line edges
      blurred = cv2.GaussianBlur(bgr_img, (5, 5), 0)

      # 2. Extract bright line candidates in HSV (low saturation, high brightness)
      hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
      white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 65, 255]))

      # Morphologically close small gaps caused by surface texture or shadows
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
      white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

      # 3. Edge detection + Probabilistic Hough Transform
      edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
      lines = cv2.HoughLinesP(
          edges,
          rho=1,
          theta=np.pi / 180,
          threshold=80,
          minLineLength=60,
          maxLineGap=25,
      )
      return lines if lines is not None else np.empty((0, 1, 4), dtype=np.int32)

    def overlay_court_lines(self,img,lines,color=(0, 255, 0),thickness=3,return_pil: bool = True):
        canvas = img.copy()
        for seg in lines:
          x1, y1, x2, y2 = map(int, seg.reshape(4))
          cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        if return_pil:
          rgb_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
          return Image.fromarray(rgb_out)
        return canvas