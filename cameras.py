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
import uuid
from dotenv import load_dotenv
load_dotenv()

class VideoRecorder():
    def __init__(self, court:str, email:str, duration:int):
        self.court = court
        self.email = email
        self.duration = duration
        # Replace with your camera details
        self.IP_ADDRESS = os.getenv("IP_ADDRESS")
        self.USERNAME = os.getenv("USERNAME")
        self.PASSWORD = os.getenv("PASSWORD")
        self.uuid = str(uuid.uuid4())
    def test_camera(self):
        rtsp_url = f"rtsp://{self.USERNAME}:{self.PASSWORD}@{self.IP_ADDRESS}:554/stream1"
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return "NOT LIVE"
        else:
            return "LIVE"
    def start_recording(self):
        # Tapo RTSP stream URL structure (Standard definition stream usually works best)
        # For HD stream, change /stream1 to /stream2 (varies by model)
        rtsp_url = f"rtsp://{self.USERNAME}:{self.PASSWORD}@{self.IP_ADDRESS}:554/stream1"
        start_time = time.time()
        # ---------------------------------------------------------
        # Streamed Audio Recorder (Fixed RAM Usage)
        # ---------------------------------------------------------
        AUDIO_SAMPLE_RATE = 44100
        audio_queue = queue.Queue()
        is_recording = True
        default_device_index = sd.default.device[0]
        if default_device_index is None or default_device_index < 0:
            default_device_index = sd.query_devices(kind='input')['index']
        temp_audio_path = f"temp_audio_{self.uuid}.wav"
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
        temp_video_path = f"temp_video_{self.uuid}.mp4"
        final_output_path = f"{self.uuid}.mp4"
        # Remove existing temp audio file if it exists
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        # Start background audio file writer thread
        audio_start_time = time.time()
        audio_thread = threading.Thread(target=record_audio_to_file)
        audio_thread.start()
        # Warm up camera sensor
        for _ in range(20):
            try:
                cap.read()
            except cv2.error:
                pass
        not_image_normalized = True
        out = None 
        angle = 0
        video_start_time = None
        frame_count = 0
        print(f"Recording video + audio... Press 'q' to stop.")
        consecutive_failures = 0
        max_consecutive_failures = 30  # ~1 sec of dropped frames at 30fps before we give up
        vision_failures = 0
        max_vision_failures = 30  # give up if the vision model can't get a usable crop for ~1 sec straight
        try:
            while True:
                try:
                    ret, frame = cap.read()
                except cv2.error as e:
                    ret, frame = False, None
                    print(f"Warning: cap.read() raised {e}")
                if not ret or frame is None:
                    consecutive_failures += 1
                    print(f"Warning: Failed to grab frame ({consecutive_failures}/{max_consecutive_failures}).")
                    if consecutive_failures >= max_consecutive_failures:
                        print("Error: Too many consecutive failed reads, attempting to reconnect...")
                        cap.release()
                        time.sleep(1.0)
                        cap = cv2.VideoCapture(rtsp_url)
                        if not cap.isOpened():
                            print("Error: Could not reconnect to video stream. Stopping recording.")
                            break
                        consecutive_failures = 0
                    continue
                consecutive_failures = 0
                if video_start_time is None:
                    video_start_time = time.time()
                elif time.time() - video_start_time >= self.duration * 60:
                    break
                # Crop & normalize alignment
                try:
                    if not_image_normalized:
                        cropped, angle = vision_model2.crop_based_angle(frame)
                    else:
                        cropped = vision_model2.crop_post_angle(frame, angle)
                except Exception as e:
                    vision_failures += 1
                    print(f"Warning: vision model failed on this frame ({vision_failures}/{max_vision_failures}): {e}")
                    if vision_failures >= max_vision_failures:
                        print("Error: Vision model failed too many times in a row. Stopping recording.")
                        break
                    continue

                # Validate the crop before trusting its dimensions
                if cropped is None or cropped.size == 0 or cropped.shape[0] <= 0 or cropped.shape[1] <= 0:
                    vision_failures += 1
                    print(f"Warning: vision model returned an empty/invalid crop ({vision_failures}/{max_vision_failures}).")
                    if vision_failures >= max_vision_failures:
                        print("Error: Vision model failed too many times in a row. Stopping recording.")
                        break
                    continue

                vision_failures = 0
                not_image_normalized = False

                try:
                    if out is None:
                        crop_h, crop_w = cropped.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (crop_w, crop_h))
                        if not out.isOpened():
                            print("Error: Could not open VideoWriter with the detected crop size. Stopping recording.")
                            out = None
                            break
                    elif cropped.shape[:2] != (crop_h, crop_w):
                        # Crop size drifted frame-to-frame; skip rather than crash the writer
                        print("Warning: crop size changed mid-recording, skipping frame.")
                        continue
                    out.write(cropped)
                    frame_count += 1
                except cv2.error as e:
                    print(f"Warning: failed to write frame, skipping: {e}")
                    continue
                
        finally:
            video_end_time = time.time()
            cap.release()
            if out is not None:
                out.release()
            # Stop background thread gracefully
            is_recording = False
            audio_thread.join()
            # Calculate actual real-world FPS to ensure zero sync-drift over long recordings
            actual_duration = video_end_time - video_start_time if video_start_time is not None else 0
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
                "-movflags +faststart",
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