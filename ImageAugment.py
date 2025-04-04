import cv2
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms

# Directories change based of Testing/Training and Posture classes     
input_dir = 'Original/Original Forward/Test'
output_dir = 'images/test/Forward-Head'

os.makedirs(output_dir, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((664, 428), antialias=True),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ColorJitter(brightness=0.3),
    transforms.RandomRotation(5),
    #transforms.RandomPerspective(0.5), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5]),
])

# Process images
for file in os.listdir(input_dir):
    if file.endswith(('.jpg')):
        # Load image
        img_path = os.path.join(input_dir, file)
        pil_img = Image.open(img_path).convert('RGB')
        # Apply transformations
        i = 0
        while i < 4: # or 10   
            transformed_img = transform(pil_img)
            #denormalize the colour range of the transform
            denormalize = transformed_img * 0.5 + 0.5
            
            # If transform outputs a tensor:
            if isinstance(transformed_img, torch.Tensor):
                # Convert tensor to numpy array [C,H,W] -> [H,W,C]
                np_img = (torch.clamp(denormalize, 0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            

            # Save with proper path joining
            prefix = str(i) + '_'
            filename = prefix + file
            output_path = os.path.join(output_dir, filename)
            success = cv2.imwrite(output_path, cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
            i+=1

            if not success:
                print(f"Failed to save {output_path}")
                
        
        
        



        
    
  



