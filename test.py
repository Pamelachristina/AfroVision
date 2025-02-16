import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Set to evaluation mode

# Image Preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ])
    
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Run the model on an image
def segment_image(image_path):
    input_image = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(input_image)
    
    # Extract the segmentation mask
    segmentation_mask = output['out'][0]  # First item in the batch
    segmentation_mask = torch.argmax(segmentation_mask, dim=0)  # Find the most likely class for each pixel
    
    return segmentation_mask

# Segment the image and replace the background
def replace_background(image_path, segmentation_mask, background_path):
    # Convert segmentation mask to binary mask for "person" class (class 15)
    binary_mask = np.where(segmentation_mask == 15, 1, 0)  # 1 for "person", 0 for background

    # Replace the background with a virtual background (ensure background is the same size)
    background_image = Image.open(background_path).resize((segmentation_mask.shape[1], segmentation_mask.shape[0]))

    # Convert image and background to numpy arrays
    foreground = np.array(Image.open(image_path).convert("RGBA"))
    background = np.array(background_image.convert("RGBA"))

    # Combine the foreground (person) with the background using the binary mask
    final_image = foreground * binary_mask[..., None] + background * (1 - binary_mask[..., None])

    # Convert back to an image
    final_image = Image.fromarray(final_image.astype(np.uint8))

    return final_image

# Example usage
image_path = "/Users/pamelasanchezhernandez/AfroVision/sample.jpg"  # Update this path
 
background_path = "/Users/pamelasanchezhernandez/AfroVision/price_is_right.jpg" # Update to the actual background path

# Step 1: Segment the image
segmentation_mask = segment_image(image_path)

# Step 2: Replace background using the segmentation mask
final_image = replace_background(image_path, segmentation_mask, background_path)

# Show the final result
final_image.show()






