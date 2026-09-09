import cv2
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
from visionmodel import VisionModel
from PIL import Image

vision_model = VisionModel()

def get_macos_camera_list():
    # Retrieve actual hardware video capture devices from AVFoundation
    devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
    camera_info = []
    
    for idx, device in enumerate(devices):
        camera_info.append({
            "index": idx,
            "name": device.localizedName(),
            "unique_id": device.uniqueID()
        })
    return camera_info

# Print device list
cameras = get_macos_camera_list()
for cam in cameras:
    print(f"Index {cam['index']}: {cam['name']}")

import cv2

# Set the index of the camera you want to record from (e.g., 0, 1, 2)
SELECTED_CAMERA_INDEX = 0

cap = cv2.VideoCapture(SELECTED_CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {SELECTED_CAMERA_INDEX}")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0

# Define codec and create VideoWriter object (saves as output.mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

print(f"Recording from camera {SELECTED_CAMERA_INDEX}... Press 'q' to stop.")

not_image_normalized = True
out = None 
for _ in range(20):
    cap.read()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    if not_image_normalized:
        cropped,baseline_1,baseline_2 = vision_model.crop_based_angle(frame)
        not_image_normalized = False
    else:
        cropped = vision_model.crop_post_process(frame,baseline_1,baseline_2)
    # Write the frame to the video file
    if out is None:
        crop_h, crop_w = cropped.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (crop_w, crop_h))
    out.write(cropped)

    # Display live feed in a window
    cv2.imshow('Recording Feed', cropped)

    # Press 'q' to quit recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
print("Recording saved successfully.")


