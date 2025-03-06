import cv2
import os
import uuid
import numpy as np
import time
from collections import deque
from motion import MotionDetection
from samurai import process_video

# Global Declarations
FRAME_QUEUE = deque(maxlen=10)
STOP_SIGNAL = False  # Signal to stop the frame reader thread
SAM2_PROCESSING = False  # Flag to indicate SAM2 processing
motion = MotionDetection()

def decode_fourcc(value):
    """Decode the FourCC codec value."""
    return "".join([chr((value >> 8 * i) & 0xFF) for i in range(4)])

def configure_camera(cap, width=1280, height=720, fps=90, codec="MJPG"):
    """Configure the camera with resolution, FPS, and codec."""
    if not cap or not cap.isOpened():
        return None

    fourcc = cv2.VideoWriter_fourcc(*codec)
    old_fourcc = decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))

    if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
        print(f"Codec changed from {old_fourcc} to {decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))}")
    else:
        print(f"Error: Could not change codec from {old_fourcc}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    print(f"Camera configured: FPS={cap.get(cv2.CAP_PROP_FPS)}, Width={cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Height={cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    return cap

def process_motion_session_with_sam2(video_path, bbox_points, model_path, output_folder):
    """Process recorded video with SAM2."""
    x_coords, y_coords = zip(*bbox_points)
    x1, y1 = min(x_coords), min(y_coords)
    w = max(x_coords) - x1
    h = max(y_coords) - y1

    output_path = os.path.join(output_folder, f"{uuid.uuid4()}.mp4")
    motion_type = process_video(
        video_path=video_path,
        coords=(x1, y1, w, h),
        model_path=model_path,
        save_video=True,
        output_path=output_path
    )
    return motion_type

def main(camera_index=0, width=1280, height=720, fps=90, codec="MJPG"):
    global SAM2_PROCESSING
    motion_session_started = False
    motion_video_writer = None
    motion_detector = MotionDetection()

    # Initialize Camera
    cap = cv2.VideoCapture(camera_index)
    cap = configure_camera(cap, width, height, fps, codec)
    if not cap or not cap.isOpened():
        print("Error: Camera not initialized.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read initial frame.")
        return

    # Hardcoded ROI points
    roi_points = [(477, 311), (1120, 310), (1277, 625), (298, 630)]
    if not roi_points:
        print("Error: ROI selection canceled.")
        return

    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_points)], 255)

    # Hardcoded bounding box points
    bbox_points = [(500, 320), (1100, 320), (1200, 600), (400, 600)]
    print(f"Initial Bounding Box Points: {bbox_points}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply ROI mask and detect motion
            motion_detector.update_motion_status(frame, roi_mask)

            if motion_detector.motion_detected:
                print("Motion Detected")
                if not motion_session_started:
                    if not os.path.exists("videos"):
                        os.makedirs("videos")
                    motion_video_path = os.path.join("videos", f"{uuid.uuid4()}.mp4")
                    motion_video_writer = cv2.VideoWriter(
                        motion_video_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    )
                    motion_session_started = True
                    print(f"Started motion session: {motion_video_path}")
                
                if motion_video_writer:
                    motion_video_writer.write(frame)

            elif motion_session_started:
                print("Motion Stopped")
                if motion_video_writer:
                    motion_video_writer.release()
                    print(f"Stopped motion session: {motion_video_path}")
                motion_session_started = False

                # Process the recorded video with SAM2
                print("Processing video with SAM2...")
                s_t = time.time()
                motion_type = process_motion_session_with_sam2(
                    motion_video_path, bbox_points, "./sam2/checkpoints/sam2.1_hiera_large.pt", "sam2_results"
                )
                print(f"Inference completed in {time.time() - s_t:.2f} seconds. Motion Type: {motion_type}")

            # Handle keyboard input
            key = input("Press 'q' to quit or Enter to continue: ")
            if key.lower() == 'q':
                break
    
    finally:
        STOP_SIGNAL = True
        if motion_video_writer:
            motion_video_writer.release()
        cap.release()
        print("Camera and resources released.")

if __name__ == "__main__":
    main(camera_index=0, width=1280, height=720, fps=90, codec="MJPG")
