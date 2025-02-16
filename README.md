## AfroVision
Improving Hair Segmentation for Curly and Afro-Textured Hair

## Inspiration
Virtual backgrounds often distort or cut off curly, textured, and afro hair due to biased segmentation models trained mostly on straight and wavy hair. AfroVision enhances segmentation accuracy to ensure better representation across all hair types.

## What It Does
AfroVision improves virtual background rendering by enhancing segmentation for curly, coily, and afro-textured hair. We implemented DeepLabV3 with ResNet101 to refine hair detection and blending in virtual settings.

## How It Was Built
Data Collection: Curated datasets featuring diverse hair textures, supplemented by scraping additional images.
Segmentation Model: Used DeepLabV3-ResNet101 for precise hair segmentation.
Preprocessing & Augmentation: Normalized images and resized masks for consistency.
Testing & Evaluation: Applied segmentation to real-world images, analyzing effectiveness across different hair textures.

## Challenges
Dataset Gaps: Most datasets lack adequate representation of Black and curly hair.
File Management Issues: Google Drive access problems slowed workflow.
Processing Errors: Debugging segmentation outputs and image masks took time.
Time Constraints: While functional, further fine-tuning is necessary for real-time applications.

## Accomplishments
Built a working segmentation model tailored for diverse hair textures.
Demonstrated real-world improvements in virtual background blending.
Identified biases in existing models and laid the foundation for future fine-tuning.
What We Learned
AI Bias Affects Real-World Applications: Standard models struggle with afro-textured hair due to limited training data.
Fine-Tuning is Key: Pre-trained models help, but domain-specific data is essential.
Efficient Debugging Saves Time: Managing datasets, handling file paths, and optimizing preprocessing are crucial.

## Next Steps
🚀 Integrate with Zoom SDK for real-time segmentation
📊 Fine-tune on afro-textured hair datasets
📂 Expand dataset & annotation for better model generalization
⚡ Optimize performance for real-time applications
🛠 Open-source AfroVision to help address bias in AI

