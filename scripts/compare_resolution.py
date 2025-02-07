import cv2
import numpy as np
import os

def get_resolution_ratio(original_frame_path, generated_frame_path, mask_path):
    """Calculates the resolution ratio between original and generated subframes."""
    
    # Load images
    original = cv2.imread(original_frame_path)
    generated = cv2.imread(generated_frame_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load binary mask

    # Find the bounding box of the modified region
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    # Extract modified region
    mod_orig_res = original[y:y+h, x:x+w].shape[:2]  # (height, width)
    mod_gen_res = generated[y:y+h, x:x+w].shape[:2]

    # Calculate resolution ratio
    resolution_ratio = (mod_gen_res[0] / mod_orig_res[0], mod_gen_res[1] / mod_orig_res[1])

    print(f"Original Modified Region Resolution: {mod_orig_res}")
    print(f"Generated Modified Region Resolution: {mod_gen_res}")
    print(f"Resolution Ratio: {resolution_ratio}")

    # Determine if superresolution is needed
    needs_superres = resolution_ratio[0] < 1 or resolution_ratio[1] < 1

    return needs_superres, (x, y, w, h)

if __name__ == "__main__":
    frame_files = sorted(os.listdir("frames/mask"))
    
    for frame_file in frame_files:
        orig_frame = os.path.join("frames/original", frame_file)
        gen_frame = os.path.join("frames/generated", frame_file)
        mask_frame = os.path.join("frames/mask", frame_file)

        needs_superres, bbox = get_resolution_ratio(orig_frame, gen_frame, mask_frame)

        if needs_superres:
            print(f"Frame {frame_file}: Superresolution required.")
        else:
            print(f"Frame {frame_file}: Superresolution NOT required.")
