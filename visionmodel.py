import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
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

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    