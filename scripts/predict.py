
        
# import torch
# from .model import UNet
# from torchvision import transforms
# from PIL import Image
# import numpy as np

# class Predictor:
#     def __init__(self, model_path, device=None, DEBUG=False):
#         self.DEBUG = DEBUG
#         self.imgsize = (256, 256)
#         self.device = self.setup_device(device)
#         self.model = UNet(num_classes=1).to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
        
#         self.transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor()
#         ])

#     def setup_device(self, device):
#         if device is None:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             device = torch.device(device)    
#         print(f"Using device: {device}")
#         return device

#     def get_mask(self, image_):
#         image_ = image_.to(self.device)
#         with torch.no_grad():
#             output = self.model(image_)
#         output = torch.sigmoid(output)  # Apply sigmoid to get probability map
#         mask = (output > 0.5).float()  # Threshold at 0.5 to create binary mask
#         return mask  # Shape: (1, 1, 256, 256)

#     def blend_foreground_on_background(self, source_img, mask, background_img):
#         """
#         Crops the source image using the given mask and pastes it onto the background image.
        
#         Args:
#             source_img (torch.Tensor): Tensor of shape (1, 3, 256, 256) - Source image.
#             mask (torch.Tensor): Tensor of shape (1, 1, 256, 256) - Binary mask (1 for foreground, 0 for background).
#             background_img (torch.Tensor): Tensor of shape (1, 3, 256, 256) - Background image.

#         Returns:
#             torch.Tensor: The final composited image (1, 3, 256, 256).
#         """
#         # Ensure mask is broadcastable to (1, 3, 256, 256)
#         mask = mask.expand(-1, 3, -1, -1)  # Convert (1,1,256,256) → (1,3,256,256)

#         # Extract foreground
#         foreground = source_img * mask  # Keep only the masked region of the source

#         # Extract background where mask is 0
#         background = background_img * (1 - mask)  # Remove masked regions from background

#         # Combine foreground and background
#         blended_img = foreground + background

#         return blended_img

#     def background_changed_img(self, image, background_img):
#         image_ = self.transform(image).unsqueeze(0).to(self.device)
#         bg_image = self.transform(background_img).unsqueeze(0).to(self.device)

#         if self.DEBUG:
#             print("Image shape: ", image_.shape)
#             print("Background shape: ", bg_image.shape)    

#         mask = self.get_mask(image_).to(self.device)  # Output shape: (1, 1, 256, 256)

#         if self.DEBUG:
#             print("Mask shape: ", mask.shape)
#             print("Mask unique values: ", np.unique(mask.cpu().numpy()))

#         merged_img = self.blend_foreground_on_background(image_, mask, bg_image)
#         return merged_img

#     def background_chnage_video(self, video_path, background_img):
#         # Load video and process each frame
#         pass



import torch
from .model import UNet
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2  # OpenCV for video processing

class Predictor:
    def __init__(self, model_path, device=None, DEBUG=False):
        self.DEBUG = DEBUG
        self.imgsize = (256, 256)
        self.device = self.setup_device(device)
        self.model = UNet(num_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def setup_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)    
        print(f"Using device: {device}")
        return device

    def get_mask(self, image_):
        image_ = image_.to(self.device)
        with torch.no_grad():
            output = self.model(image_)
        output = torch.sigmoid(output)  # Apply sigmoid to get probability map
        mask = (output > 0.5).float()  # Threshold at 0.5 to create binary mask
        return mask  # Shape: (1, 1, 256, 256)

    def blend_foreground_on_background(self, source_img, mask, background_img):
        """
        Crops the source image using the given mask and pastes it onto the background image.
        
        Args:
            source_img (torch.Tensor): Tensor of shape (1, 3, 256, 256) - Source image.
            mask (torch.Tensor): Tensor of shape (1, 1, 256, 256) - Binary mask (1 for foreground, 0 for background).
            background_img (torch.Tensor): Tensor of shape (1, 3, 256, 256) - Background image.

        Returns:
            torch.Tensor: The final composited image (1, 3, 256, 256).
        """
        # Ensure mask is broadcastable to (1, 3, 256, 256)
        mask = mask.expand(-1, 3, -1, -1)  # Convert (1,1,256,256) → (1,3,256,256)

        # Extract foreground
        foreground = source_img * mask  # Keep only the masked region of the source

        # Extract background where mask is 0
        background = background_img * (1 - mask)  # Remove masked regions from background

        # Combine foreground and background
        blended_img = foreground + background

        return blended_img

    def background_changed_img(self, image, background_img):
        image_ = self.transform(image).unsqueeze(0).to(self.device)
        bg_image = self.transform(background_img).unsqueeze(0).to(self.device)

        if self.DEBUG:
            print("Image shape: ", image_.shape)
            print("Background shape: ", bg_image.shape)    

        mask = self.get_mask(image_).to(self.device)  # Output shape: (1, 1, 256, 256)

        if self.DEBUG:
            print("Mask shape: ", mask.shape)
            print("Mask unique values: ", np.unique(mask.cpu().numpy()))

        merged_img = self.blend_foreground_on_background(image_, mask, bg_image)
        return merged_img

    def background_chnage_video(self, video_path, background_img, output_video_path="output_video.mp4"):
        """
        Processes a video by replacing its background for each frame.
        
        Args:
            video_path (str): Path to the input video.
            background_img (PIL.Image): The background image to be used.
            output_video_path (str): Path to save the processed video.

        Returns:
            str: The output video path.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        # We'll use the same resolution as our transform (256x256)
        width, height = 256, 256

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame (BGR) to PIL Image (RGB) and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).resize((width, height))

            # Process the frame using the model
            processed_tensor = self.background_changed_img(frame_pil, background_img)
            # Convert tensor to image
            processed_frame = (processed_tensor.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            
            out.write(processed_frame)
            current_frame += 1

            if self.DEBUG:
                print(f"Processed frame {current_frame}/{frame_count}")

        cap.release()
        out.release()
        return output_video_path
