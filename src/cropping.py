import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import crop
from torchvision.transforms import transforms
from mpl_toolkits.axes_grid1 import ImageGrid

class GaussianCrops:
    def __init__(self, crop_percentage=0.4, seed=None, std_scale=1 ,
                 padding=False, regularised_crop=False, adaptive_center=False):
        """Initialization method for the cropping class.
            
            Args:
                crop_percentage (float, optional): Percentage of the image to crop. Defaults to 0.4.
                seed (int, optional): Seed for random number generation. Defaults to None.
                std_scale (float, optional): Scaling factor for the standard deviation of the Gaussian distribution. Defaults to 1.
                padding (bool, optional): Boolean flag indicating whether padding is applied. Defaults to False.
                regularised_crop (bool, optional): Boolean flag indicating whether regularized cropping is applied. Defaults to False.
            """
        self.crop_percentage = crop_percentage
        self.seed = seed
        self.std_scale = std_scale
        self.padding = padding
        self.regularised_crop = regularised_crop
        self.adaptive_center = adaptive_center
        
    def gcc(self, img):
        """
        Gaussian centered Cropping 
        Crop an image using isotropic Gaussian distributed cropping.
        Parameters:
        - image (np.array): Image to be cropped.
        - crop_percentage (float): Percentage of the original image to be cropped.
        - seed (int): Seed for reproducibility.
        - std_scale (float): Scaling factor for the standard deviation of the Gaussian distribution.
        
        Returns:
        - List of np.arrays: with two views of the cropped image.     
        """
        image_width, image_height = img.shape[1], img.shape[0]
        crop_width = int(image_width * np.sqrt(self.crop_percentage))
        crop_height = int(image_height * np.sqrt(self.crop_percentage))
        
        std_x = image_width  * self.std_scale
        std_y = image_height  * self.std_scale
        mean_x, mean_y = [image_width // 2, image_height // 2]
        
        if self.adaptive_center:
            mean_x = np.random.uniform(image_width * 0.25, image_width * 0.75)
            mean_y = np.random.uniform(image_height * 0.25, image_height * 0.75)
        
        if self.seed is not None:
            np.random.seed(self.seed)
        centers = np.random.multivariate_normal([mean_x,mean_y], [[std_x, 0], [0, std_y]], 2)
        
        crops = []
        for center in centers:
            center_x, center_y = int(center[0]), int(center[1])
            left = center_x - crop_width // 2
            top =  center_y - crop_height // 2
            bottom = top + crop_height 
            right = left + crop_width 
            
            
            if self.regularised_crop:
                if top < 0:
                    top = 0
                    bottom = crop_height
                if left < 0:
                    left = 0
                    right = crop_width
                if bottom > image_height:
                    bottom = image_height
                    top = bottom - crop_height
                if right > image_width:
                    right = image_width
                    left = right - crop_width             
                   
       
            if self.padding:
                if left < 0 and right> 0:
                    left = 0
                    pad_left = crop_width - right
                    pad_right = 0
                elif right > image_width and left < image_width:
                    right = image_width
                    pad_left = 0
                    pad_right = right - image_width
                elif left <= 0 and right <=0:
                    left = 0
                    right = 0
                    pad_left = crop_width
                    pad_right = 0 
                elif left >= image_width and right >= image_width:
                    left = 0
                    right = 0
                    pad_left = 0
                    pad_right = crop_width                   
                else:
                    pad_left = 0
                    pad_right = 0
                    
                if top < 0 and bottom > 0:
                    top = 0
                    pad_top = crop_height - bottom
                    pad_bottom = 0
                elif bottom > image_height and top < image_height:
                    bottom = image_height
                    pad_top = 0
                    pad_bottom = bottom - image_height
                elif top <= 0 and bottom <= 0:
                    top = 0
                    bottom = 0
                    pad_top = crop_height
                    pad_bottom = 0
                elif top >= image_height and bottom >= image_height:
                    top = 0
                    bottom = 0
                    pad_top = 0
                    pad_bottom = crop_height
                else:
                    pad_top = 0
                    pad_bottom = 0
                 
            if bottom - top < crop_height:
                pad_top = (crop_height - (bottom - top)) // 2
                pad_bottom = crop_height - (bottom - top) - pad_top   
            if right - left < crop_width:
                pad_left = (crop_width - (right - left)) // 2
                pad_right = crop_width - (right - left) - pad_left
                    
            cropped_image = img[top:bottom, left:right]
            cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom),
                (pad_left, pad_right), (0, 0)), mode='constant')
             
            assert cropped_image.shape[0] == crop_height
            assert cropped_image.shape[1] == crop_width         
            crops.append(cropped_image)
        return crops
    
class UniformCrops:
    def __init__(self, min_scale=-1, max_scale =1, crop_percentage=0.4, seed=None,
                padding=False, regularised_crop=False):
        """Initialization method for the cropping class.

            Args:
                min_scale (int, optional): Minimum scale factor for cropping. Defaults to -1.
                max_scale (int, optional): Maximum scale factor for cropping. Defaults to 1.
                crop_percentage (float, optional): Percentage of the image to crop. Defaults to 0.4.
                seed (int, optional): Seed for random number generation. Defaults to None.
                padding (bool, optional): Boolean flag indicating whether padding is applied. Defaults to False.
                regularised_crop (bool, optional): Boolean flag indicating whether regularized cropping is applied. Defaults to False.
        """
        
        self.crop_percentage = crop_percentage
        self.seed = seed
        self.padding = padding
        self.regularised_crop = regularised_crop
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    
    def ucc(self, img):
        """
        Uniform centered Cropping 
        Crop an image using isotropic uniform distributed cropping.
        Parameters:
        - image (np.array): Image to be cropped.
        - crop_percentage (float): Percentage of the original image to be cropped.
        - seed (int): Seed for reproducibility.
        
        Returns:
        - List of np.arrays: with two views of the cropped image.     
        """
        image_width, image_height = img.shape[1], img.shape[0]
        crop_width = int(image_width * np.sqrt(self.crop_percentage))
        crop_height = int(image_height * np.sqrt(self.crop_percentage))
        
        min_hight = image_height//2 - int(image_width * self.min_scale)
        max_hight=  image_height//2 + int(image_width * self.max_scale)
        
        min_width = image_width//2 - int(image_height * self.min_scale)
        max_width = image_width//2 + int(image_height * self.max_scale)
        
        if self.seed is not None:
            np.random.seed(self.seed)
        centers = np.random.uniform([image_width / 2, image_height / 2],
                                    [[min_hight, min_width], [max_hight, max_width]], (2, 2))
        crops = []
        for center in centers:
            center_x, center_y = int(center[0]), int(center[1])
            left = center_x - crop_width // 2
            top =  center_y - crop_height // 2
            bottom = top + crop_height 
            right = left + crop_width 
            
           
            if self.regularised_crop:
                if top < 0:
                    top = 0
                    bottom = crop_height
                if left < 0:
                    left = 0
                    right = crop_width
                if bottom > image_height:
                    bottom = image_height
                    top = bottom - crop_height
                if right > image_width:
                    right = image_width
                    left = right - crop_width             
                   
       
            if self.padding:
                if left < 0 and right> 0:
                    left = 0
                    pad_left = crop_width - right
                    pad_right = 0
                elif right > image_width and left < image_width:
                    right = image_width
                    pad_left = 0
                    pad_right = right - image_width
                elif left <= 0 and right <=0:
                    left = 0
                    right = 0
                    pad_left = crop_width
                    pad_right = 0 
                elif left >= image_width and right >= image_width:
                    left = 0
                    right = 0
                    pad_left = 0
                    pad_right = crop_width                   
                else:
                    pad_left = 0
                    pad_right = 0
                    
                if top < 0 and bottom > 0:
                    top = 0
                    pad_top = crop_height - bottom
                    pad_bottom = 0
                elif bottom > image_height and top < image_height:
                    bottom = image_height
                    pad_top = 0
                    pad_bottom = bottom - image_height
                elif top <= 0 and bottom <= 0:
                    top = 0
                    bottom = 0
                    pad_top = crop_height
                    pad_bottom = 0
                elif top >= image_height and bottom >= image_height:
                    top = 0
                    bottom = 0
                    pad_top = 0
                    pad_bottom = crop_height
                else:
                    pad_top = 0
                    pad_bottom = 0
                 
            if bottom - top < crop_height:
                pad_top = (crop_height - (bottom - top)) // 2
                pad_bottom = crop_height - (bottom - top) - pad_top   
            if right - left < crop_width:
                pad_left = (crop_width - (right - left)) // 2
                pad_right = crop_width - (right - left) - pad_left
                    
            cropped_image = img[top:bottom, left:right]
            cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom),
                (pad_left, pad_right), (0, 0)), mode='constant')
             
            assert cropped_image.shape[0] == crop_height
            assert cropped_image.shape[1] == crop_width         
            crops.append(cropped_image)
        return crops
    
    
if __name__ == "__main__":
    img = Image.open("g44yy1dz.bmp")
    img = np.array(img)

    num_samples = 12
    num_views = 2
    padding = True
    regularised_crop = False

    # Create a new figure
    crops1 = []
    for i in range(num_samples):
        crop_percentage = 0.4
        seed = None
        adaptive_center = True
        std_scale = 1
        crop = GaussianCrops(crop_percentage=crop_percentage,
                            seed=seed,
                            std_scale=std_scale,
                            padding=padding,
                            regularised_crop=regularised_crop,
                            adaptive_center=adaptive_center)
        cropped_images = crop.gcc(img)
        crops1.extend(cropped_images)
    num_images = len(crops1)
    spacing = 0.05
    num_cols = 6
    num_rows = (num_images + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(12, 12))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(num_rows, num_cols),
                     axes_pad=spacing,  # spacing between images
                     )

    for ax, image in zip(grid, crops1):
        ax.imshow(image)
        ax.axis("off")
    plt.title("Gaussian Cropping with adaptive center")
    plt.legend()
    plt.show()
    plt.savefig(f"gcc_adp_std_{std_scale}_cropsize_{crop_percentage}.jpg")
    crops2 = []
    for i in range(num_samples):
        crop_percentage = 0.4
        seed = None
        adaptive_center = False
        std_scale = 10
        crop = GaussianCrops(crop_percentage=crop_percentage,
                            seed=seed,
                            std_scale=std_scale,
                            padding=padding,
                            regularised_crop=regularised_crop,
                            adaptive_center=adaptive_center)
        cropped_images = crop.gcc(img)
        crops2.extend(cropped_images)
    num_images = len(crops2)
    spacing = 0.05
    num_cols = 6
    num_rows = (num_images + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(12, 12))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(num_rows, num_cols),
                     axes_pad=spacing,  # spacing between images
                     )

    for ax, image in zip(grid, crops2):
        ax.imshow(image)
        ax.axis("off")
    plt.title("Gaussian Cropping without adaptive center")
    plt.show()
    plt.savefig(f"gcc_not_adp_std_{std_scale}_cropsize_{crop_percentage}.jpg")