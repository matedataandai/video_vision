import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
from ultralytics import YOLO
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
    def __init__(self, model_path='keypoints_model.pth'):
        self.model = models.resnet50(pretrained=True)
        self.model_path = model_path
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
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

class PlayerDetector:
    def __init__(self):
        self.model_name = 'yolo11n.pt'
        self.detector = YOLO(self.model_name)

    def predict(self, image):
        results = self.detector.predict(image,verbose=False)
        player_shoes = []
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = box.cpu()
                player_shoes.append(((x1+x2)/2,y2))
        return player_shoes

class DrawCourt:
    # ---- reference court (top-down units) ----
    L, R = 286, 1379          # doubles sidelines
    Li, Ri = 423, 1242        # singles sidelines
    T, B = 561, 2935          # baselines
    NET = 1748
    TS, BS = 1110, 2386       # service lines
    MID = 832                 # centre service line x
    PAD = 400

    REF_PTS = np.array([
        (L, T), (R, T), (L, B), (R, B),
        (Li, T), (Li, B), (Ri, T), (Ri, B),
        (Li, TS), (Ri, TS), (Li, BS), (Ri, BS),
        (MID, TS), (MID, BS),
    ], dtype=np.float32)

    def __init__(self, keypoints, player_positions):
        """
        keypoints:        (14, 2) for a static camera, or (F, 14, 2) one set per frame.
                          A flat 28-vector (or (F, 28)) is also accepted.
        player_positions: (F, 2) player foot position per frame, image pixels
                          (e.g. bbox bottom-centre). NaN / None rows are skipped.
        """
        kp = np.asarray(keypoints, dtype=np.float32)
        self.keypoints = kp.reshape(-1, 14, 2) if kp.ndim != 2 or kp.shape != (14, 2) else kp

        pos = np.array([[np.nan, np.nan] if p is None else p for p in player_positions],
                       dtype=np.float32).reshape(-1, 2)
        self.player_positions = pos

        self.W = self.R + self.PAD
        self.H_px = self.B + self.PAD

        self.per_frame = self.keypoints.ndim == 3
        if self.per_frame and len(self.keypoints) not in (1, len(pos)):
            raise ValueError("keypoints and player_positions must have the same number of frames")

        self.court_points = self._project_all()   # (N, 2) in court coordinates

    # ------------------------------------------------------------------
    # homography / projection
    # ------------------------------------------------------------------
    def _homography(self, kp):
        H, _ = cv2.findHomography(kp, self.REF_PTS, cv2.RANSAC, 5.0)
        return H

    def _project_all(self):
        if not self.per_frame or len(self.keypoints) == 1:
            kp = self.keypoints[0] if self.per_frame else self.keypoints
            H = self._homography(kp)
            Hs = [H] * len(self.player_positions)
        else:
            Hs, last = [], None
            for kp in self.keypoints:
                H = self._homography(kp)
                last = H if H is not None else last      # reuse last good H on failure
                Hs.append(last)

        out = np.full_like(self.player_positions, np.nan)
        for i, (p, H) in enumerate(zip(self.player_positions, Hs)):
            if H is None or np.isnan(p).any():
                continue
            out[i] = cv2.perspectiveTransform(p.reshape(1, 1, 2), H)[0, 0]
        return out

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def court_lines(self, line=6, color=(255, 255, 255)):
        """Transparent BGRA canvas with court lines only."""
        court = np.zeros((self.H_px, self.W, 4), np.uint8)
        c = (*color, 255)
        L, R, Li, Ri, T, B = self.L, self.R, self.Li, self.Ri, self.T, self.B

        def ln(p, q):
            cv2.line(court, p, q, c, line, cv2.LINE_AA)

        ln((L, T), (R, T));   ln((L, B), (R, B))
        ln((L, T), (L, B));   ln((R, T), (R, B))
        ln((Li, T), (Li, B)); ln((Ri, T), (Ri, B))
        ln((Li, self.TS), (Ri, self.TS)); ln((Li, self.BS), (Ri, self.BS))
        ln((self.MID, self.TS), (self.MID, self.BS))
        ln((L, self.NET), (R, self.NET))
        return court

    def density(self, sigma=45):
        pts = self.court_points[~np.isnan(self.court_points).any(axis=1)]
        xs = np.round(pts[:, 0]).astype(int)
        ys = np.round(pts[:, 1]).astype(int)
        ok = (xs >= 0) & (xs < self.W) & (ys >= 0) & (ys < self.H_px)
        acc = np.zeros((self.H_px, self.W), np.float32)
        np.add.at(acc, (ys[ok], xs[ok]), 1.0)
        return cv2.GaussianBlur(acc, (0, 0), sigma)

    def heatmap(self, sigma=45, bg_color=(40, 90, 40), line_color=(255, 255, 255),
                max_alpha=0.8, min_level=0.04, crop=True):
        """Colourised density over a court background, lines on top. Returns BGR image."""
        dens = self.density(sigma)
        norm = dens / (dens.max() + 1e-9)

        bg = np.full((self.H_px, self.W, 3), bg_color, np.uint8)
        colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        alpha = (np.clip((norm - min_level) / (1 - min_level), 0, 1) ** 0.6 * max_alpha)[..., None]
        out = (bg * (1 - alpha) + colour * alpha).astype(np.uint8)

        lines = self.court_lines(color=line_color)
        a = lines[..., 3:4].astype(np.float32) / 255.0
        out = (out * (1 - a) + lines[..., :3] * a).astype(np.uint8)
        return self._crop(out) if crop else out

    def positions_plot(self, radius=10, color=(0, 0, 255), bg_color=(40, 90, 40), crop=True):
        """Every projected position as a dot (plus the trail line) on the court."""
        out = np.full((self.H_px, self.W, 3), bg_color, np.uint8)
        lines = self.court_lines()
        a = lines[..., 3:4].astype(np.float32) / 255.0
        out = (out * (1 - a) + lines[..., :3] * a).astype(np.uint8)

        prev = None
        for p in self.court_points:
            if np.isnan(p).any():
                prev = None
                continue
            q = (int(p[0]), int(p[1]))
            if prev is not None:
                cv2.line(out, prev, q, color, 2, cv2.LINE_AA)
            cv2.circle(out, q, radius, color, -1, cv2.LINE_AA)
            prev = q
        return self._crop(out) if crop else out

    def _crop(self, img, margin=150):
        return img[self.T - margin:self.B + margin, self.L - margin:self.R + margin]
