import cv2
from gfpgan import GFPGANer
from basicsr.archs.codeformer_arch import CodeFormer

# Load Superresolution Models
gfpgan = GFPGANer(model_path="checkpoints/GFPGANv1.pth", upscale=2, arch="clean", channel_multiplier=2)
codeformer = CodeFormer()

def apply_superresolution(image, method="GFPGAN"):
    """Applies GFPGAN or CodeFormer to enhance the given region of an image."""
    if method == "GFPGAN":
        _, _, enhanced_image = gfpgan.enhance(image, has_aligned=False, only_center_face=False)
    elif method == "CodeFormer":
        enhanced_image = codeformer(image)
    else:
        return image  # No enhancement applied
    
    return enhanced_image

def process_frame_with_superres(original_frame, generated_frame, mask_frame, superres_method):
    """Enhance the generated frame only in the modified region if superresolution is enabled."""
    
    mask = cv2.imread(mask_frame, cv2.IMREAD_GRAYSCALE)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    # Extract modified region
    modified_region = generated_frame[y:y+h, x:x+w]

    # Apply superresolution
    enhanced_region = apply_superresolution(modified_region, method=superres_method)

    # Replace the modified region in the generated frame
    generated_frame[y:y+h, x:x+w] = enhanced_region

    return generated_frame
