import cv2
import numpy as np

def domain_transform(image):
    # Edge-preserving smoothing for domain transformation
    return cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)

def image_fusion(original, transformed, a=0.5):
    # Fusion to enhance details by blending the original and transformed images
    return cv2.addWeighted(original, 1-a, transformed, a, 0)

def denoise_image(image):
    # First apply Gaussian blur for denoising
    temp = cv2.GaussianBlur(image, (5, 5), 0)
    # Then apply median blur
    return cv2.medianBlur(temp, 5)

# Load the image
image_path = 'ODIR/newimages/Male_4577_left.jpg'
original_image = cv2.imread(image_path)
assert original_image is not None, "Image not found."

# Ensure image is in the correct format (uint8)
if original_image.dtype != np.uint8:
    original_image = np.uint8(255 * (original_image - original_image.min()) / (original_image.max() - original_image.min()))

# Apply domain transform for background information
transformed_image = domain_transform(original_image)

# Fuse the original and the transformed image
fused_image = image_fusion(original_image, transformed_image)

# Denoise the fused image
final_image = denoise_image(fused_image)

# Save or display the final enhanced image
cv2.imwrite('ODIR/test.jpg', final_image)
cv2.imshow('Enhanced Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
