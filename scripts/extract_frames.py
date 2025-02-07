import cv2
import os

def extract_frames(video_path, output_folder):
    """Extracts frames from a video and saves them as images."""
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)  # Save frame as image
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

if __name__ == "__main__":
    # Extract frames from original and generated videos
    # extract_frames("assets/input_video_with_audio.mp4", "frames/original")   Original video
    # extract_frames("assets/lipsynced.mp4", "frames/generated")   Generated video
    extract_frames("frames/mask_video.mp4", "frames/mask")
    

