import cv2
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Write the frame to the video file
    out.write(frame)

    # Display live feed in a window
    cv2.imshow('Recording Feed', frame)

    # Press 'q' to quit recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording saved successfully.")


