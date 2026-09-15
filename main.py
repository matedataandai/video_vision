import os
import queue
import subprocess
import threading
import time
import cv2
import sounddevice as sd
import soundfile as sf
import imageio_ffmpeg
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
from visionmodel import VisionModel2
import cv2
import os
from dotenv import load_dotenv
load_dotenv()
# Replace with your camera details
IP_ADDRESS = os.getenv("IP_ADDRESS")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Tapo RTSP stream URL structure (Standard definition stream usually works best)
# For HD stream, change /stream1 to /stream2 (varies by model)
rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:554/stream1"

# ---------------------------------------------------------
# Streamed Audio Recorder (Fixed RAM Usage)
# ---------------------------------------------------------
AUDIO_SAMPLE_RATE = 44100
audio_queue = queue.Queue()
is_recording = True

default_device_index = sd.default.device[0]
if default_device_index is None or default_device_index < 0:
    default_device_index = sd.query_devices(kind='input')['index']

temp_audio_path = "temp_audio.wav"

def record_audio_to_file():
    """Streams audio chunks directly to disk to prevent RAM accumulation."""
    def callback(indata, frames, time_info, status):
        if is_recording:
            audio_queue.put(indata.copy())

    # Open SoundFile writer directly to disk
    with sf.SoundFile(temp_audio_path, mode='x', samplerate=AUDIO_SAMPLE_RATE, channels=1) as file:
        with sd.InputStream(device=default_device_index, samplerate=AUDIO_SAMPLE_RATE, channels=1, callback=callback):
            while is_recording or not audio_queue.empty():
                try:
                    data = audio_queue.get(timeout=0.1)
                    file.write(data)
                except queue.Empty:
                    pass

# ---------------------------------------------------------
# OpenCV & Vision Model Setup
# ---------------------------------------------------------
vision_model2 = VisionModel2()

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

fps = 30.0
temp_video_path = "temp_video.mp4"
final_output_path = "output_with_audio.mp4"

# Remove existing temp audio file if it exists
if os.path.exists(temp_audio_path):
    os.remove(temp_audio_path)

# Start background audio file writer thread
audio_start_time = time.time()
audio_thread = threading.Thread(target=record_audio_to_file)
audio_thread.start()

# Warm up camera sensor
for _ in range(20):
    cap.read()

not_image_normalized = True
out = None 
angle = 0
video_start_time = None
frame_count = 0

print(f"Recording video + audio... Press 'q' to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        if video_start_time is None:
            video_start_time = time.time()

        # Crop & normalize alignment
        if not_image_normalized:
            cropped, angle = vision_model2.crop_based_angle(frame)
            not_image_normalized = False
        else:
            cropped = vision_model2.crop_post_angle(frame, angle)

        if out is None:
            crop_h, crop_w = cropped.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (crop_w, crop_h))
            
        out.write(cropped)
        frame_count += 1

        cv2.imshow('Recording Feed', cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_end_time = time.time()
    
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Stop background thread gracefully
    is_recording = False
    audio_thread.join()

    # Calculate actual real-world FPS to ensure zero sync-drift over long recordings
    actual_duration = video_end_time - video_start_time
    actual_fps = frame_count / actual_duration if actual_duration > 0 else fps
    print(f"Recorded {frame_count} frames over {actual_duration/60:.2f} mins (Effective FPS: {actual_fps:.2f})")

    # ---------------------------------------------------------
    # Merge Video and Audio using FFmpeg
    # ---------------------------------------------------------
    print("Merging video and audio...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    initial_delay = max(0.0, video_start_time - audio_start_time)

    cmd = [
        ffmpeg_exe, "-y",
        "-ss", f"{initial_delay:.3f}",   # Trim initial start delay
        "-i", temp_audio_path,
        "-r", str(actual_fps),          # Dynamic FPS calculation for sync over long sessions
        "-i", temp_video_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        final_output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"SUCCESS: Recording saved to {final_output_path}")
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
    else:
        print("FFmpeg merge error:")
        print(result.stderr)