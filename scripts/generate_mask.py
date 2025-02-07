import cv2
import numpy as np
import os

def generate_mask_video(original_folder, generated_folder, output_video_path, fps=25):
    """Creates a video highlighting differences between original and generated frames."""
    frame_files = sorted(os.listdir(original_folder))  # Ensure frames are aligned
    height, width = cv2.imread(os.path.join(original_folder, frame_files[0])).shape[:2]
    
    # Create video writer
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame_file in frame_files:
        orig_path = os.path.join(original_folder, frame_file)
        gen_path = os.path.join(generated_folder, frame_file)

        if not os.path.exists(gen_path):
            continue

        # Load frames
        orig_frame = cv2.imread(orig_path)
        gen_frame = cv2.imread(gen_path)

        # Compute absolute difference
        diff = cv2.absdiff(orig_frame, gen_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Apply threshold to highlight changes
        _, mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

        # Convert to color format for video writing
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out.write(mask_colored)

    out.release()
    print(f"Mask video saved as {output_video_path}")

if __name__ == "__main__":
    generate_mask_video("frames/original", "frames/generated", "frames/mask_video.mp4")
