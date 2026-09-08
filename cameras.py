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