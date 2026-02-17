import cv2
import numpy as np

def show_image_with_mask(image, mask, title="Image with Mask"):
    """Display the image with the mask overlay."""
    overlay = image.copy()
    mask_color = np.zeros_like(image)
    mask_color[mask > 0] = [0, 255, 0]  # Green color for the mask

    # Blend the mask with the image
    alpha = 0.5
    cv2.addWeighted(overlay, 1 - alpha, mask_color, alpha, 0, overlay)

    cv2.imshow(title, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, path):
    """Save the image to the specified path."""
    cv2.imwrite(path, image)

def visualize_detection(image, mask, output_path):
    """Visualize the detection results by overlaying the mask on the image and saving the result."""
    show_image_with_mask(image, mask)
    save_image(image, output_path)